"""Trainable LoRA adapters, optimizers and schedulers. Model-agnostic.

Thin layer over ComfyUI's own weight-adapter machinery. Two things here are not in the
stock helpers and both matter:

  * adapters are built only on the modules asked for, rather than on every weight in the
    model. On a 12.8B denoiser the stock behaviour means an enormous optimizer state and
    a slider entangled with the timestep and text projections;
  * ``alpha`` is frozen. ``requires_grad_(True)`` unfreezes *every* parameter on the
    adapter, and ComfyUI's LoRA adapter stores alpha as a Parameter. alpha is a fixed
    scale hyperparameter (the effective scale is alpha/rank); letting the optimizer drift
    it -- with weight decay on top -- adds an uncontrolled global gain fighting the
    learning rate. ComfyUI's own trainer has this bug.
"""

import logging
import math

import torch

from comfy.weight_adapter import adapter_maps, adapters
from comfy.weight_adapter.bypass import BypassInjectionManager
from comfy_extras.nodes_train import BiasDiff

LOG = "[ZeroTrainSlider]"


def build_adapters(model_patcher, targets, existing_weights, algorithm, dtype, rank, alpha):
    """Attach a trainable LoRA to each target module.

    ``existing_weights`` is a plain LoRA state dict -- the closed-form solve's output
    drops straight in. Any module it covers starts from those weights instead of from
    zero; every other module starts fresh. That is the whole warm start: no merging, one
    LoRA, and the optimizer simply continues from a much better place than random.
    """
    if algorithm not in adapter_maps:
        raise ValueError(
            "{} Unknown adapter algorithm {!r}. ComfyUI offers: {}. Note the "
            "capitalisation.".format(LOG, algorithm, ", ".join(adapter_maps)))

    lora_sd = {}
    all_adapters = []
    bypass_manager = BypassInjectionManager()
    warm_started = []

    for name, module in targets:
        key = "{}.weight".format(name)
        module_alpha = alpha
        if existing_weights:
            # The saved key is "<module>.alpha", not "<module>.weight.alpha"; looking it
            # up under the wrong name silently resets a resumed run's alpha.
            module_alpha = float(existing_weights.get("{}.alpha".format(name), alpha))

        existing = None
        if existing_weights:
            for adapter_cls in adapters:
                existing = adapter_cls.load(name, existing_weights, module_alpha, None)
                if existing is not None:
                    break

        if existing is not None:
            train_adapter = existing.to_train().to(dtype)
            warm_started.append(name)
        else:
            train_adapter = adapter_maps[algorithm].create_train(
                module.weight, rank=rank, alpha=alpha).to(dtype)

        train_adapter = train_adapter.train().requires_grad_(True)
        for pname, parameter in train_adapter.named_parameters():
            if pname.rsplit(".", 1)[-1] == "alpha":
                parameter.requires_grad_(False)
            lora_sd["{}.{}".format(name, pname)] = parameter

        all_adapters.append(train_adapter)
        if isinstance(train_adapter, BiasDiff):
            model_patcher.add_weight_wrapper(key, train_adapter)
        else:
            bypass_manager.add_adapter(key, train_adapter, strength=1.0)

    if warm_started:
        logging.info("%s Warm-started %d module(s) from the closed-form solve: %s",
                     LOG, len(warm_started), ", ".join(warm_started[:4])
                     + (" ..." if len(warm_started) > 4 else ""))
    return lora_sd, all_adapters, bypass_manager, warm_started


def make_optimizer(name, params, lr, weight_decay):
    params = list(params)
    if name == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort a long run
            logging.warning("%s AdamW8bit unavailable (%s); using AdamW. Install "
                            "bitsandbytes to enable it.", LOG, exc)
            name = "AdamW"
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("{} Unknown optimizer {!r}.".format(LOG, name))


def make_lr_scheduler(optimizer, name, total_steps, warmup_steps):
    warmup_steps = max(0, min(int(warmup_steps), max(0, total_steps - 1)))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def snapshot(lora_sd, save_dtype):
    """Detached CPU copy of the live LoRA parameters."""
    out = {}
    for key, value in lora_sd.items():
        # alpha is a 0-dim float; keep it fp32 so alpha/rank stays exact.
        dtype = torch.float32 if key.endswith(".alpha") else save_dtype
        out[key] = value.detach().to(dtype).cpu().contiguous()
    return out


def clone_out_of_inference(obj):
    """Deep-copy tensors so none carry the inference-tensor flag.

    ComfyUI executes nodes inside ``torch.inference_mode()``. Tensors born there cannot be
    saved for backward, so anything entering an autograd graph must be re-materialised
    inside an ``inference_mode(False)`` block.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_out_of_inference(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [clone_out_of_inference(v) for v in obj]
        return tuple(cloned) if isinstance(obj, tuple) else cloned
    return obj


def sigma_view(sigma, ndim):
    """Broadcast a per-sample sigma vector against a (B, C, ...) tensor."""
    return sigma.reshape(-1, *([1] * (ndim - 1)))
