"""Measure what the finished LoRA actually does to the denoiser, and scale it to match.

Everything in ``zt_core`` happens in text-conditioning space. That is where the concept
can be measured cleanly, but it is not where the user looks. A slider can score well on
``token_fit`` and still be invisible in an image, because how strongly the denoiser
responds to a given conditioning shift is a property of the other 12GB of the model that
the solve never touches.

So after the solve, this module runs the real thing: a short Euler rollout through all
of the model's blocks, then, at a few timesteps on that trajectory,

    D_true = denoise(x, positive prompt) - denoise(x, negative prompt)
    D_lora = denoise(x, neutral prompt, WITH the LoRA) - denoise(x, neutral prompt)

``D_true`` is the effect the user would get by editing the prompt. ``D_lora`` is the
effect they get from the slider at strength 1. Two numbers fall out:

* ``image_effect`` = cos(D_lora, D_true). Does the slider move the denoiser the same way
  the prompt pair does? This is the go/no-go number, and it is the only one measured
  where the image is actually decided.
* ``gain`` = argmin ||g D_lora - D_true||^2. The strength at which the slider matches the
  prompt difference. Baking it in makes strength 1.0 mean "as strong as writing the
  prompt", which is the only scale a user can reason about.

The LoRA is applied through ``comfy.sd.load_lora_for_models`` -- the same path the stock
Load LoRA node uses -- so this doubles as proof that the saved keys actually bind to the
model. A silently unbound LoRA looks exactly like a weak one otherwise.

The pass is optional and every failure is caught: a model whose backend cannot drive a
denoise falls back to reporting nothing rather than losing the build.
"""

import logging

import torch

import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.utils

LOG = "[ZeroTrainSlider]"


def _prepare_conds(real_model, texts, conds, noise, device, seed):
    """Conditioning tensors -> the processed form ``calc_cond_batch`` requires.

    ``process_conds`` is not optional: it runs the model's own ``extra_conds``, which is
    what turns a raw conditioning tensor into the ``model_conds`` entry the diffusion
    model reads. Skipping it hands the DiT no context at all.
    """
    raw = {t: comfy.sampler_helpers.convert_cond([[conds[t].to(device), {}]])
           for t in texts}
    return comfy.samplers.process_conds(
        real_model, noise, raw, device, latent_image=noise, denoise_mask=None, seed=seed)


def calibrate(backend, model, candidates, conds, triplets, width, height,
              probe_scenes=3, probe_steps=3, schedule_steps=8, seed=0, progress=True):
    """Measure what each candidate LoRA does to the denoiser. Never raises.

    ``candidates`` is {name: lora state dict}. They all share one frozen rollout -- by
    far the expensive part -- so comparing two objectives costs one extra forward per
    probe sample rather than a second full pass.

    ``conds`` is the {prompt text: cpu tensor} cache built during collection, so nothing
    is re-encoded and the text encoder stays unloaded.

    Returns {"ok": bool, "candidates": {name: metrics}, ...} or {"ok": False, "error"}.
    """
    try:
        return _calibrate(backend, model, candidates, conds, triplets, width, height,
                          probe_scenes, probe_steps, schedule_steps, seed, progress)
    except comfy.model_management.InterruptProcessingException:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed check must not lose the build
        logging.exception("%s Verification pass failed; the LoRAs are still valid.", LOG)
        return {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)}


def _calibrate(backend, model, candidates, conds, triplets, width, height,
               probe_scenes, probe_steps, schedule_steps, seed, progress):
    device = model.load_device
    probes = triplets[: max(1, int(probe_scenes))]

    x_shape = backend.latent_shape(model, width, height)
    sigmas = backend.sigmas(model, int(schedule_steps))

    # Pick timesteps spread over the usable part of the schedule. Structure is decided
    # early (high sigma) and surface late, so a slider of either kind gets seen.
    usable = [k for k in range(len(sigmas) - 1) if float(sigmas[k]) > 1e-4]
    if not usable:
        raise RuntimeError("schedule produced no usable sigma")
    n = max(1, min(int(probe_steps), len(usable)))
    picks = sorted({usable[round(i * (len(usable) - 1) / max(1, n - 1))]
                    for i in range(n)}) if n > 1 else [usable[len(usable) // 2]]

    total = (len(probes) * (len(usable) + 2 * len(picks))
             + len(candidates) * len(probes) * len(picks))
    pbar = comfy.utils.ProgressBar(total) if progress else None
    ticks = [0]

    def tick():
        ticks[0] += 1
        if pbar is not None:
            pbar.update_absolute(ticks[0], total, None)

    # ---- 1. frozen model: roll out, and record the true prompt-pair difference --------
    comfy.model_management.load_models_gpu([model])
    real = model.model
    options = model.model_options
    comfy.samplers.cast_to_load_options(options, device=device, dtype=model.model_dtype())
    # pre_run() assigns model.current_patcher, which calc_cond_batch dereferences for
    # prepare_state()/get_free_memory()/apply_hooks().
    model.pre_run()
    sampling = real.model_sampling
    noise_template = torch.zeros(x_shape, device=device, dtype=torch.float32)

    def denoise(patcher_model, opts, cond_list, x, sigma):
        with torch.no_grad():
            return comfy.samplers.calc_cond_batch(patcher_model, cond_list, x, sigma, opts)

    samples = []          # (x, sigma, D_true, pred_neutral, neutral text)
    for index, (neutral, positive, negative) in enumerate(probes):
        comfy.model_management.throw_exception_if_processing_interrupted()
        prepared = _prepare_conds(real, (neutral, positive, negative), conds,
                                  noise_template, device, seed)
        c_neu, c_pos, c_neg = prepared[neutral], prepared[positive], prepared[negative]

        generator = torch.Generator(device="cpu").manual_seed(seed + index * 9973)
        noise = torch.randn(x_shape, generator=generator, dtype=torch.float32).to(device)
        sigma0 = torch.full((x_shape[0],), float(sigmas[0]), device=device,
                            dtype=torch.float32)
        x = sampling.noise_scaling(sigma0, noise, torch.zeros_like(noise), True)

        for k in usable:
            sigma_k = torch.full((x_shape[0],), float(sigmas[k]), device=device,
                                 dtype=torch.float32)
            neutral_pred = denoise(real, options, [c_neu], x, sigma_k)[0].float()
            tick()

            if k in picks:
                pos_pred, neg_pred = denoise(real, options, [c_pos, c_neg], x, sigma_k)
                tick(); tick()
                samples.append((
                    x.detach().cpu(), float(sigmas[k]),
                    (pos_pred.float() - neg_pred.float()).cpu() * 0.5,
                    neutral_pred.cpu(), neutral,
                ))
                del pos_pred, neg_pred

            # Euler on the model's own trajectory: calc_cond_batch returns the denoised
            # x0 estimate, so the flow derivative is (x - denoised) / sigma. Probing on
            # latents the model actually visits matters -- sigma*noise is off-manifold at
            # low sigma and the measurement there is mostly noise.
            if k + 1 < len(sigmas):
                view = sigma_k.reshape(-1, *([1] * (x.ndim - 1)))
                x = x + ((x - neutral_pred) / view) * (float(sigmas[k + 1]) - float(sigmas[k]))
            del neutral_pred

    # ---- 2. same latents, neutral prompt, each candidate applied in turn -------------
    # load_lora_for_models is the stock Load LoRA path, so an unbound key shows up here
    # as a zero effect rather than as a mystery at generation time.
    results = {}
    for label, lora in candidates.items():
        patched, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0.0)
        bound = _count_bound(model, lora)
        comfy.model_management.load_models_gpu([patched])
        patched.pre_run()
        p_real = patched.model
        p_options = patched.model_options
        comfy.samplers.cast_to_load_options(p_options, device=device,
                                            dtype=patched.model_dtype())
        neutral_conds = _prepare_conds(p_real, {s[4] for s in samples}, conds,
                                       noise_template, device, seed)

        dots = den_l = den_t = 0.0
        per_sample = []
        for x_cpu, sigma_value, d_true, neutral_pred, text in samples:
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = x_cpu.to(device)
            sigma_k = torch.full((x.shape[0],), sigma_value, device=device,
                                 dtype=torch.float32)
            lora_pred = denoise(p_real, p_options, [neutral_conds[text]], x, sigma_k)[0]
            tick()
            d_lora = lora_pred.float().cpu() - neutral_pred

            a = d_lora.flatten().double()
            b = d_true.flatten().double()
            dots += float((a * b).sum())
            den_l += float((a * a).sum())
            den_t += float((b * b).sum())
            per_sample.append({
                "sigma": sigma_value,
                "cosine": float((a * b).sum() / (a.norm().clamp_min(1e-20)
                                                 * b.norm().clamp_min(1e-20))),
                "relative_size": float(a.norm() / b.norm().clamp_min(1e-20)),
            })
            del x, lora_pred

        results[label] = {
            "image_effect": dots / max((den_l ** 0.5) * (den_t ** 0.5), 1e-30),
            # The strength at which this candidate best matches the prompt difference.
            "gain": dots / max(den_l, 1e-30),
            # How big its effect is compared to the prompt's, before any scaling. A tiny
            # number here with a good cosine means the direction is right and the channel
            # is just narrow -- that is what the gain is for.
            "relative_size": (den_l / max(den_t, 1e-30)) ** 0.5,
            "bound_keys": bound,
            "total_keys": sum(1 for k in lora if k.endswith(".lora_up.weight")),
            "samples": per_sample,
        }
        del patched

    comfy.model_management.soft_empty_cache()
    return {
        "ok": True,
        "candidates": results,
        "probe_scenes": len(probes),
        "probe_sigmas": [float(sigmas[k]) for k in picks],
    }


def assert_binds(model, lora):
    """Fail the build if ComfyUI's key map resolves none of the LoRA's modules.

    This costs nothing and closes the worst failure mode the package has: a file that
    loads without complaint and does absolutely nothing, at any strength. It used to be
    reported only by the verification pass, which meant that when that pass could not run
    the user was left with a silent dud and no way to tell.
    """
    bound = _count_bound(model, lora)
    total = sum(1 for k in lora if k.endswith(".lora_up.weight"))
    if bound == 0 and total:
        modules = sorted({k.rsplit(".lora_up.weight", 1)[0]
                          for k in lora if k.endswith(".lora_up.weight")})
        raise RuntimeError(
            "{} None of this LoRA's modules ({}) are resolved by ComfyUI's LoRA key map, "
            "so the file would load and do nothing at any strength. The backend's site "
            "names must match the model's named_modules() exactly.".format(
                LOG, ", ".join(modules)))
    return bound, total


def _count_bound(model, lora):
    """How many of the LoRA's modules ComfyUI's key map actually resolves.

    Zero here means the file will load silently and do nothing -- the single most
    confusing failure mode a hand-built LoRA has.
    """
    try:
        import comfy.lora
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        modules = {k.rsplit(".lora_up.weight", 1)[0]
                   for k in lora if k.endswith(".lora_up.weight")}
        return sum(1 for m in modules if m in key_map)
    except Exception:  # noqa: BLE001 - diagnostic only
        return -1
