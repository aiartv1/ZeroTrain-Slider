"""Gradient slider training, warm-started from the closed-form solve. Model-agnostic.

The closed-form solve can only express what the text encoder already encodes, because it
only edits the text tower. That ceiling is why simple concepts (style, camera distance)
come out well and subtler ones stay weak. This module removes the ceiling: it runs real
gradient training through the *whole* denoiser, starting from the closed-form weights.

Objective (Gandikota et al., "Concept Sliders"), adapted to flow matching:

    frozen model, positive prompt  -> pred_pos
    frozen model, negative prompt  -> pred_neg
    frozen model, neutral prompt   -> pred_neutral
    delta      = (pred_pos - pred_neg) / 2
    LoRA @ +1, neutral prompt      -> minimise MSE(pred, pred_neutral + eta*delta)
    LoRA @ -1, neutral prompt      -> minimise MSE(pred, pred_neutral - eta*delta)

Only LoRA weights get gradients; the base model is frozen throughout, and the adapters
are applied as forward hooks (ComfyUI "bypass" adapters) rather than by touching the base
tensors -- which is what lets an FP8 checkpoint train with no dequantisation pass.

Why the warm start is not just "faster"
---------------------------------------
Ordinary LoRA training starts from noise-and-zeros: the adapter does nothing at step 0 and
most of the run is spent finding a direction at all. Here the text-tower adapter starts
from the closed-form solution -- already pointing the right way and already scaled to the
prompt difference -- while adapters on the denoiser blocks start at zero and learn only
what the text path could not reach. One LoRA, two origins, no merging: merging two
independently-built LoRAs risks them partly cancelling, and nothing here can cancel
because there is only ever one set of weights being refined.

Everything model-specific lives in the backend (``trainable_modules``, ``latent_shape``,
``sigmas``, ``checkpoint_types``, ``training_numerics``), so adding a model to the trainer
is the same one file that adds it to the solver.
"""

import logging
import random
import time

import torch

import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
from comfy_extras.nodes_train import patch as checkpoint_patch
from comfy_extras.nodes_train import unpatch as checkpoint_unpatch

from .zt_adapters import (
    build_adapters, clone_out_of_inference, make_lr_scheduler, make_optimizer,
    sigma_view, snapshot,
)

LOG = "[ZeroTrainSlider]"
GB = 1024 ** 3


class _Sample:
    """One precomputed training target: a latent, its sigma, and the slider axis there."""

    __slots__ = ("x", "sigma", "neutral", "delta", "scene")

    def __init__(self, x, sigma, neutral, delta, scene):
        self.x = x
        self.sigma = sigma
        self.neutral = neutral
        self.delta = delta
        self.scene = scene


def _encode(backend, clip, texts):
    out = {}
    for text in texts:
        comfy.model_management.throw_exception_if_processing_interrupted()
        out[text] = clone_out_of_inference(backend.encode(clip, text))
    return out


def _prepare(real_model, conds, texts, noise, device, seed):
    raw = {t: comfy.sampler_helpers.convert_cond([[conds[t].to(device), {}]]) for t in texts}
    return comfy.samplers.process_conds(
        real_model, noise, raw, device, latent_image=noise, denoise_mask=None, seed=seed)


def build_target_bank(backend, model_patcher, real_model, options, conds, triplets,
                      x_shape, sigmas, step_window, seed, salt=0, progress=True):
    """Precompute (latent, sigma, neutral prediction, slider axis) over real trajectories.

    Latents come from the model's own Euler rollout rather than ``sigma * noise``. The
    naive shortcut is off-manifold at low sigma -- the model never sees such latents at
    inference -- and a slider trained there comes out weak and noisy. The rollout also
    pays for itself: step k's neutral prediction *is* the target's centre at sigma_k, so a
    usable sample costs three forwards instead of four.
    """
    device = model_patcher.load_device
    noise_template = torch.zeros(x_shape, device=device, dtype=torch.float32)
    window = set(step_window)
    bank = []

    total = len(triplets) * (len(sigmas) - 1 + 2 * len(window))
    pbar = comfy.utils.ProgressBar(total) if progress else None
    done = [0]

    def tick(n=1):
        done[0] += n
        if pbar is not None:
            pbar.update_absolute(min(done[0], total), total, None)

    for index, (neutral, positive, negative) in enumerate(triplets):
        comfy.model_management.throw_exception_if_processing_interrupted()
        prepared = _prepare(real_model, conds, (neutral, positive, negative),
                            noise_template, device, seed)
        c_neu, c_pos, c_neg = prepared[neutral], prepared[positive], prepared[negative]

        generator = torch.Generator(device="cpu").manual_seed(
            (seed + salt * 100003 + index * 9973) % (2 ** 63 - 1))
        noise = torch.randn(x_shape, generator=generator, dtype=torch.float32).to(device)
        sigma0 = torch.full((x_shape[0],), float(sigmas[0]), device=device,
                            dtype=torch.float32)
        x = real_model.model_sampling.noise_scaling(
            sigma0, noise, torch.zeros_like(noise), True)

        for k in range(len(sigmas) - 1):
            if float(sigmas[k]) <= 1e-4:
                continue
            sigma_k = torch.full((x_shape[0],), float(sigmas[k]), device=device,
                                 dtype=torch.float32)
            with torch.no_grad():
                pred_neutral = comfy.samplers.calc_cond_batch(
                    real_model, [c_neu], x, sigma_k, options)[0].float()
            tick()

            if k in window:
                with torch.no_grad():
                    pred_pos, pred_neg = comfy.samplers.calc_cond_batch(
                        real_model, [c_pos, c_neg], x, sigma_k, options)
                tick(2)
                bank.append(_Sample(
                    x.detach().cpu(), float(sigmas[k]),
                    pred_neutral.cpu(),
                    ((pred_pos.float() - pred_neg.float()) * 0.5).cpu(),
                    neutral,
                ))
                del pred_pos, pred_neg

            # Euler: calc_cond_batch returns the denoised x0 estimate, so the flow
            # derivative is (x - denoised) / sigma.
            view = sigma_view(sigma_k, x.ndim)
            x = x + ((x - pred_neutral) / view) * (float(sigmas[k + 1]) - float(sigmas[k]))
            del pred_neutral

    comfy.model_management.soft_empty_cache()
    return bank


def train_slider(backend, model, clip, triplets, opts, warm_start=None, progress=True):
    """Run the slider training loop. Returns (lora_sd, loss_map, report)."""
    started = time.time()
    device = model.load_device
    steps = int(opts["steps"])
    seed = int(opts["training_seed"])

    x_shape = backend.latent_shape(model, int(opts["width"]), int(opts["height"]),
                                   batch=int(opts["batch_size"]))
    sigmas = backend.sigmas(model, int(opts["schedule_steps"]), opts["scheduler"])

    usable = [k for k in range(len(sigmas) - 1) if float(sigmas[k]) > 1e-4]
    first, last = int(opts["first_step"]), int(opts["last_step"])
    if last < 0:
        last = len(sigmas) - 2
    window = [k for k in usable if first <= k <= last]
    if not window:
        raise RuntimeError(
            "{} first_step={} / last_step={} left no usable timestep out of {}.".format(
                LOG, first, last, len(sigmas) - 1))

    scenes = triplets[: max(1, int(opts["train_scenes"]))]

    # ComfyUI runs nodes inside inference_mode(); tensors born there cannot be saved for
    # backward, so everything that touches autograd is re-materialised inside this block.
    with torch.inference_mode(False):
        texts = [t for triple in scenes for t in triple]
        conds = _encode(backend, clip, list(dict.fromkeys(texts)))

        # The text encoder is dead weight from here on, and evicting it is what buys the
        # denoiser its headroom on a 12GB card.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        mp = model.clone()
        mp.model.requires_grad_(False).train()

        targets = backend.trainable_modules(mp, opts["target_modules"], opts["blocks"])
        warm_modules = set()
        if warm_start:
            warm_modules = {k.rsplit(".lora_up.weight", 1)[0]
                            for k in warm_start if k.endswith(".lora_up.weight")}
            # The closed-form edit lives in the text tower; the training preset targets the
            # denoiser blocks. Union them, or the warm start would have no adapter to
            # initialise and would be silently discarded.
            known = {name for name, _ in targets}
            by_name = dict(mp.model.named_modules())
            for name in sorted(warm_modules - known):
                module = by_name.get(name)
                if module is not None and getattr(module, "weight", None) is not None:
                    targets.append((name, module))
        if not targets:
            raise RuntimeError(
                "{} target_modules={!r} with blocks={!r} matched no Linear on this "
                "model.".format(LOG, opts["target_modules"], opts["blocks"]))

        lora_sd, all_adapters, bypass_manager, warm_started = build_adapters(
            mp, targets, warm_start or {}, opts["algorithm"],
            getattr(torch, opts["lora_dtype"]), int(opts["rank"]), float(opts["alpha"]))
        trainable = [p for p in lora_sd.values() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        logging.info("%s %d adapters (%d warm-started), rank %d -> %.2fM trainable params.",
                     LOG, len(targets), len(warm_started), int(opts["rank"]), n_params / 1e6)

        comfy.model_management.load_models_gpu([mp])
        real_model = mp.model
        options = mp.model_options
        comfy.samplers.cast_to_load_options(options, device=device, dtype=mp.model_dtype())
        mp.pre_run()

        checkpointed = []
        injections = []
        loss_map = {"loss": []}

        try:
            # Targets and training must run under the same kernels -- see
            # backend.training_numerics().
            with backend.training_numerics():
                logging.info("%s Precomputing targets over %d scene(s) x %d timestep(s)...",
                             LOG, len(scenes), len(window))
                bank = build_target_bank(backend, mp, real_model, options, conds, scenes,
                                         x_shape, sigmas, window, seed, 0, progress)
                if not bank:
                    raise RuntimeError("{} The target bank came out empty.".format(LOG))
                logging.info("%s Target bank ready: %d samples.", LOG, len(bank))

                # Checkpointing and LoRA injection happen after the bank, so the frozen
                # passes above run on the bare model with no hooks and no recompute.
                if opts["gradient_checkpointing"]:
                    types = backend.checkpoint_types(mp)
                    if types:
                        for module in real_model.diffusion_model.modules():
                            if isinstance(module, types):
                                checkpoint_patch(module, offloading=False)
                                checkpointed.append(module)
                        logging.info("%s Gradient checkpointing %d blocks.",
                                     LOG, len(checkpointed))

                injections = bypass_manager.create_injections(mp.model)
                for injection in injections:
                    injection.inject(mp)

                def set_strength(value):
                    for hook in bypass_manager.hooks:
                        hook.adapter.multiplier = value

                optimizer = make_optimizer(opts["optimizer"], trainable,
                                           float(opts["learning_rate"]),
                                           float(opts["weight_decay"]))
                scheduler = make_lr_scheduler(optimizer, opts["lr_scheduler"], steps,
                                              int(opts["warmup_steps"]))

                eta = float(opts["eta"])
                accum = max(1, int(opts["grad_accumulation_steps"]))
                clip_norm = float(opts["max_grad_norm"])
                rng = random.Random(seed)
                pbar = comfy.utils.ProgressBar(steps) if progress else None
                micro = 0

                prepared_neutral = {}
                noise_template = torch.zeros(x_shape, device=device, dtype=torch.float32)
                for scene in {s.scene for s in bank}:
                    prepared_neutral[scene] = _prepare(
                        real_model, conds, (scene,), noise_template, device, seed)[scene]

                comfy.model_management.in_training = True
                for step in range(steps):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss, terms = 0.0, 0

                    for _ in range(accum):
                        sample = bank[rng.randrange(len(bank))]
                        x = sample.x.to(device)
                        sigma = torch.full((x_shape[0],), sample.sigma, device=device,
                                           dtype=torch.float32)
                        neutral = sample.neutral.to(device)
                        delta = sample.delta.to(device)
                        cond = [prepared_neutral[sample.scene]]

                        if opts["direction_mode"] == "both_per_step":
                            directions = (1.0, -1.0)
                        elif opts["direction_mode"] == "positive_only":
                            directions = (1.0,)
                        else:
                            directions = (1.0 if micro % 2 == 0 else -1.0,)
                        micro += 1

                        for direction in directions:
                            target = neutral + (direction * eta) * delta
                            set_strength(direction)
                            with torch.autocast(device.type, dtype=torch.bfloat16):
                                pred = comfy.samplers.calc_cond_batch(
                                    real_model, cond, x, sigma, options)[0]
                            pred = pred.float()

                            if opts["loss_space"] == "velocity":
                                # calc_cond_batch returns x0; converting both sides back to
                                # the flow velocity weights every timestep evenly, which is
                                # the Concept Sliders convention.
                                view = sigma_view(sigma, pred.ndim)
                                loss = torch.nn.functional.mse_loss(
                                    (x - pred) / view, (x - target) / view)
                            else:
                                loss = torch.nn.functional.mse_loss(pred, target)

                            (loss / (accum * len(directions))).backward()
                            total_loss += loss.item()
                            terms += 1
                        del x, neutral, delta, target

                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, clip_norm)
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is not None:
                                parameter.grad.data = parameter.grad.data.to(
                                    parameter.data.dtype)
                    optimizer.step()
                    scheduler.step()

                    loss_map["loss"].append(total_loss / max(1, terms))
                    if pbar is not None:
                        pbar.update_absolute(step + 1, steps, None)
        finally:
            comfy.model_management.in_training = False
            try:
                for hook in bypass_manager.hooks:
                    hook.adapter.multiplier = 1.0
            except Exception:  # noqa: BLE001 - teardown must not mask a training error
                pass
            for injection in injections:
                injection.eject(mp)
            for module in checkpointed:
                checkpoint_unpatch(module)
            mp.cleanup()

        for adapter in all_adapters:
            adapter.requires_grad_(False)
        comfy.model_management.soft_empty_cache()

        out = snapshot(lora_sd, getattr(torch, opts["save_dtype"]))

    losses = loss_map["loss"]
    report = {
        "steps": steps,
        "adapters": len(targets),
        "warm_started": len(warm_started),
        "warm_started_modules": ", ".join(sorted(warm_started)) or "(none)",
        "trainable_params_m": round(n_params / 1e6, 3),
        "bank_samples": len(bank),
        "scenes": len(scenes),
        "timesteps": len(window),
        "first_loss": round(losses[0], 6) if losses else None,
        "final_loss": round(losses[-1], 6) if losses else None,
        "mean_loss_last_20": (round(sum(losses[-20:]) / len(losses[-20:]), 6)
                              if losses else None),
        "train_minutes": round((time.time() - started) / 60.0, 2),
    }
    return out, loss_map, report
