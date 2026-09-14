"""Krea 2 (K2) backend.

Krea 2 is a single-stream MMDiT: text tokens and patchified image tokens are
concatenated and run through 28 *shared* blocks. That normally makes weight editing
awkward -- ``attn.wq/wk/wv`` see image tokens as well as text ones, so any edit there
leaks into the image stream. Krea 2 happens to keep a clean seam before the join::

    context (B, seq, 12, 2560)          # 12-layer Qwen3-VL-4B tap
      -> txtfusion.layerwise_blocks     # 2 blocks, run per tap layer
      -> txtfusion.projector            # 12 taps -> 1
      -> txtfusion.refiner_blocks       # 2 blocks
      -> txtmlp = [RMSNorm, Linear(2560->6144), GELU, Linear(6144->6144)]
      -> concat with image tokens -> the 28 shared blocks

Everything above the concat is text-only, and ``txtmlp.3`` is the last linear in it.
An edit there is:

  * **exact** -- its output *is* the text conditioning the denoiser receives, with no
    non-linearity in between, so the least-squares target we solve for is achieved
    verbatim rather than approximately;
  * **isolated** -- image tokens never pass through it, so nothing else can drift.

That single layer is therefore the default site, and it is why this method needs no
sampling, no latents and no VAE on Krea 2. The whole build touches ~0.8GB of weights
(txtfusion + txtmlp) instead of the model's 12.8GB.

``text_mlp`` is offered as a second preset but is *not* exact: ``txtmlp.1`` sits behind
a GELU, so its contribution composes non-linearly with the ``txtmlp.3`` edit. The
builder splits eta across sites to keep the total magnitude honest, but expect the
achieved-vs-target diagnostics in the report to be worse than with ``text_out_proj``.
"""

import torch

from .base import SliderBackend

# Sequential indices inside SingleStreamDiT.txtmlp: 0 RMSNorm, 1 Linear, 2 GELU, 3 Linear.
_TXTMLP_OUT = r"^diffusion_model\.txtmlp\.3$"
_TXTMLP_IN = r"^diffusion_model\.txtmlp\.1$"


class Krea2Backend(SliderBackend):
    key = "krea2"
    display_name = "Krea 2 (K2)"
    model_hint = (
        "Load Diffusion Model -> krea2_turbo_fp8_scaled.safetensors (or any Krea 2 "
        "checkpoint), plus CLIPLoader with type 'krea2' for the Qwen3-VL-4B encoder"
    )

    # A slider built at eta = 1.0 moves the mean text feature by exactly the
    # positive-minus-negative prompt difference at LoRA strength 1.0. That is already a
    # large move on Krea 2; strength is the runtime dial, so there is no reason to bake
    # in more than 1.0 unless the concept is weak in the text prior.
    default_eta = 1.0
    default_rank = 16

    site_presets = {
        "text_out_proj": (_TXTMLP_OUT,),
        "text_mlp": (_TXTMLP_IN, _TXTMLP_OUT),
    }
    default_site_preset = "text_out_proj"
    exact_site_presets = ("text_out_proj",)

    notes = (
        "Krea 2 is a single-stream MMDiT, so only the pre-join text tower is safe to "
        "edit. txtmlp.3 is the last linear before text and image tokens are "
        "concatenated: editing it is exact and cannot affect image tokens."
    )

    # ---------------------------------------------------------------------------

    @classmethod
    def matches(cls, model_patcher):
        import comfy.model_base

        krea2 = getattr(comfy.model_base, "Krea2", None)
        if krea2 is None:
            return False
        return isinstance(model_patcher.model, krea2)

    @classmethod
    def describe(cls, model_patcher):
        try:
            cfg = dict(model_patcher.model.model_config.unet_config)
        except AttributeError:
            cfg = {}
        return (
            "Krea 2 single-stream MMDiT -- {} blocks, {} features, text tower "
            "{}x{} (Qwen3-VL-4B tap)".format(
                cfg.get("layers", "?"), cfg.get("features", "?"),
                cfg.get("txtlayers", "?"), cfg.get("txtdim", "?"),
            )
        )

    @classmethod
    def encode(cls, clip, text):
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        # encode_from_tokens_scheduled returns [[tensor, {pooled_output: ...}], ...];
        # Krea 2 conditions on the tensor alone. Detach and clone off the encoder's
        # device so the text encoder can be evicted before the tower runs.
        tensor = cond[0][0]
        return tensor.detach().to("cpu", copy=True)

    @classmethod
    def text_path_modules(cls, model_patcher):
        dit = model_patcher.model.diffusion_model
        return [dit.txtfusion, dit.txtmlp]

    @classmethod
    def run_text_path(cls, model_patcher, cond, device):
        dit = model_patcher.model.diffusion_model
        dtype = cls.compute_dtype(model_patcher)
        # _unpack_context also raises the "wrong text encoder" error for us, with a
        # message that names the expected CLIPLoader type.
        context = dit._unpack_context(cond.to(device=device, dtype=dtype))
        hidden = dit.txtfusion(context, mask=None, transformer_options={})
        return dit.txtmlp(hidden)

    @classmethod
    def compute_dtype(cls, model_patcher):
        dtype = super().compute_dtype(model_patcher)
        # RMSNorm in this model up-casts to fp32 internally, so bf16 elsewhere is fine;
        # fp16 is not -- the 12-tap context stack overflows it on long prompts.
        if dtype == torch.float16:
            dtype = torch.bfloat16
        return dtype

    # -- verification pass -------------------------------------------------------

    @classmethod
    def sigmas(cls, model_patcher, steps, scheduler="simple"):
        # Krea 2 Turbo is distilled to ~8 steps with ModelSamplingFlux shift 1.15. The
        # base class default already reads the schedule off the model, so this only
        # exists to keep the turbo step count from being overridden upwards by accident.
        return super().sigmas(model_patcher, max(4, int(steps)), scheduler)

    # -- gradient training -------------------------------------------------------
    # These are the *denoiser's* blocks, not the text tower. The closed-form edit is
    # confined to the text path by design; training has gradients through everything, so
    # it can reach the attention and MLP layers that decide structure and surface -- which
    # is exactly what the text path cannot express on its own.

    _ATTN_QKV = (r"^diffusion_model\.blocks\.(\d+)\.attn\.w[qkv]$",)
    _ATTN_REST = (r"^diffusion_model\.blocks\.(\d+)\.attn\.(gate|wo)$",)
    _MLP = (r"^diffusion_model\.blocks\.(\d+)\.mlp\.(gate|up|down)$",)

    train_site_presets = {
        "attn_mlp": _ATTN_QKV + _ATTN_REST + _MLP,
        "attn_only": _ATTN_QKV + _ATTN_REST,
        "attn_qkv": _ATTN_QKV,
        "mlp_only": _MLP,
    }
    default_train_preset = "attn_mlp"
    block_index_regex = r"^diffusion_model\.blocks\.(\d+)\."

    @classmethod
    def checkpoint_types(cls, model_patcher):
        from comfy.ldm.krea2.model import SingleStreamBlock, TextFusionBlock

        # TextFusionBlock is included because a warm start puts trainable weights in the
        # text tower too, and its activations are otherwise held for the whole backward.
        return (SingleStreamBlock, TextFusionBlock)
