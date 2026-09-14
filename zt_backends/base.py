"""Backend contract for the zero-training slider builder.

A *backend* teaches this package three things about one base model:

  1.  how to recognise it                      -> ``matches()``
  2.  how to turn text into conditioning       -> ``encode()``
  3.  how to run *only its text tower*, and    -> ``text_path_modules()``,
      which linears inside it may be edited       ``run_text_path()``, ``site_presets``

Nothing else. There is no training loop to port, no sampler, no latent format and no
VAE -- the whole method lives in text-conditioning space, so a new model is usually
60-100 lines. See ``ADDING_A_MODEL.md`` in the package root for a walkthrough.

Why "text tower only"
---------------------
The edit we solve for is a rank-r update to a Linear whose **input is a pure function
of the text conditioning**. That restriction is not cosmetic:

  * it is what makes the solve exact -- the layer sees text vectors and nothing else,
    so the least-squares problem written down here is the problem the model has;
  * it guarantees the LoRA cannot touch image/latent tokens, which on a single-stream
    MMDiT (Krea 2, Flux, SD3) is otherwise the main failure mode, because there the
    attention projections are shared between text and image tokens.

A site is *exact* when its output reaches the tower output through a linear path, i.e.
nothing non-linear sits between it and the conditioning handed to the denoiser. The
tower's last projection is always exact. Earlier layers are an approximation, and
backends should say so by leaving them out of ``exact_site_presets``.
"""

import contextlib
import re

import torch


class SliderBackend:
    """Subclass this, set the class attributes, implement the four classmethods."""

    # -- identity ----------------------------------------------------------------
    key = ""                     # short unique id, e.g. "krea2"
    display_name = ""            # shown in node tooltips and in the build report
    model_hint = ""              # what to wire in; quoted verbatim in error messages

    # -- defaults ----------------------------------------------------------------
    default_eta = 1.0            # sensible concept strength for this model
    default_rank = 16

    # -- edit sites --------------------------------------------------------------
    # preset name -> tuple of regexes matched against module names as returned by
    # ``model_patcher.model.named_modules()``, so they carry the "diffusion_model."
    # prefix that ComfyUI's generic LoRA key map expects.
    site_presets = {}
    default_site_preset = ""
    exact_site_presets = ()      # subset of site_presets that are mathematically exact

    notes = ""                   # free text appended to the build report

    # ---------------------------------------------------------------------------
    # required
    # ---------------------------------------------------------------------------

    @classmethod
    def matches(cls, model_patcher):
        """True if this backend can handle the given ComfyUI MODEL."""
        raise NotImplementedError

    @classmethod
    def encode(cls, clip, text):
        """Text -> one conditioning tensor of shape (1, seq, features), on CPU.

        Return the raw tensor the diffusion model consumes; whatever ComfyUI wraps
        around it (pooled output, attention mask) is the backend's business.

        Encode one prompt at a time. Batching introduces padding tokens, and every
        statistic in this package is a per-prompt token average -- padding would be
        counted as if it were content.
        """
        raise NotImplementedError

    @classmethod
    def text_path_modules(cls, model_patcher):
        """The nn.Modules making up the text tower.

        Only these get moved onto the compute device, which is the whole reason a
        12.8B model can be edited on a 12GB card in a couple of seconds.
        """
        raise NotImplementedError

    @classmethod
    def run_text_path(cls, model_patcher, cond, device):
        """Run the text tower on one conditioning tensor. Return value is ignored.

        Called under ``torch.no_grad()`` with forward hooks already installed on the
        selected sites, so this only has to make the data flow through them.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------------
    # optional
    # ---------------------------------------------------------------------------

    @classmethod
    def describe(cls, model_patcher):
        """One-line human description of the loaded model, for the report."""
        return cls.display_name

    # -- the verification pass ---------------------------------------------------
    # Only needed for the optional post-solve check that runs real denoiser forwards.
    # The defaults cover any model with a standard ComfyUI latent format; override
    # ``latent_shape`` for anything unusual (Krea 2's latents are 5D, for instance).

    @classmethod
    def latent_shape(cls, model_patcher, width, height, batch=1):
        """The noise shape the denoiser expects, aligned to the model's patch size."""
        model = model_patcher.model
        fmt = model.latent_format
        downscale = int(fmt.spacial_downscale_ratio)
        channels = int(fmt.latent_channels)

        patch = 1
        try:
            patch = int(model.model_config.unet_config.get("patch", 1)) or 1
        except Exception:  # noqa: BLE001 - not every config carries a patch size
            patch = 1

        align = downscale * patch
        width = max(align, (int(width) // align) * align)
        height = max(align, (int(height) // align) * align)
        shape = [batch, channels, height // downscale, width // downscale]

        # latent_dimensions == 3 means a temporal axis (Wan-style), so the tensor is 5D
        # with a single frame for a still image.
        if int(getattr(fmt, "latent_dimensions", 2)) == 3:
            shape.insert(2, 1)
        return tuple(shape)

    @classmethod
    def sigmas(cls, model_patcher, steps, scheduler="simple"):
        """The inference sigma schedule, so probes land on timesteps the model uses."""
        import comfy.samplers
        return comfy.samplers.calculate_sigmas(
            model_patcher.model.model_sampling, scheduler, int(steps))

    # -- gradient training (zt_train.py) ------------------------------------------
    # Only needed for the optional warm-start trainer. A backend that leaves these
    # alone still works for the closed-form solve and the verification pass.

    # preset name -> regexes for the modules that receive a trainable LoRA. Unlike
    # ``site_presets`` these are the *denoiser's* layers, not the text tower: training has
    # gradients through the whole model, so it can reach what the text path cannot.
    train_site_presets = {}
    default_train_preset = ""

    # A regex with exactly one capture group holding the block index, used to honour a
    # "blocks" filter like "8-20". Leave None on a model with no block numbering.
    block_index_regex = None

    @classmethod
    def trainable_modules(cls, model_patcher, preset, blocks="all"):
        """[(module_name, module)] to adapt during training."""
        import re

        if preset not in cls.train_site_presets:
            raise ValueError(
                "Backend {!r} has no training preset {!r}. Available: {}.".format(
                    cls.key, preset, ", ".join(sorted(cls.train_site_presets)) or "(none)"))
        patterns = [re.compile(p) for p in cls.train_site_presets[preset]]
        block_re = re.compile(cls.block_index_regex) if cls.block_index_regex else None
        wanted = parse_block_spec(blocks, cls.block_count(model_patcher))

        picked = []
        for name, module in model_patcher.model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            if not any(p.match(name) for p in patterns):
                continue
            if block_re is not None:
                found = block_re.match(name)
                if found is not None and int(found.group(1)) not in wanted:
                    continue
            picked.append((name, module))
        return picked

    @classmethod
    def block_count(cls, model_patcher):
        """How many transformer blocks the denoiser has, for the ``blocks`` filter."""
        try:
            cfg = model_patcher.model.model_config.unet_config
            for key in ("layers", "depth", "num_layers", "n_layers"):
                if key in cfg:
                    return int(cfg[key])
        except Exception:  # noqa: BLE001 - not every config exposes a depth
            pass
        return 0

    @classmethod
    def checkpoint_types(cls, model_patcher):
        """Module classes to wrap in gradient checkpointing.

        Default: the classes of the modules that own the trainable layers, which is the
        right granularity on any block-structured denoiser. Without checkpointing a model
        that streams from RAM keeps every dequantised block live for the whole backward
        pass and the run cannot fit.
        """
        types = set()
        for name, _module in cls.trainable_modules(
                model_patcher, cls.default_train_preset, "all"):
            owner = name.rsplit(".", 2)[0]
            for candidate_name, candidate in model_patcher.model.named_modules():
                if candidate_name == owner:
                    types.add(type(candidate))
                    break
        return tuple(types)

    @classmethod
    def training_numerics(cls):
        """Context manager wrapping the whole train loop, including target precompute.

        ComfyUI swaps in fused kernels whenever ``model_management.in_training`` is False.
        They are numerically close to the eager path but not identical, so building frozen
        targets outside this flag and training inside it gives every target a fixed
        kernel-mismatch offset -- an irreducible loss floor that can rival the signal on a
        subtle concept. Precompute and training must run under the same kernels.
        """
        import contextlib

        import comfy.model_management

        @contextlib.contextmanager
        def _numerics():
            previous = comfy.model_management.in_training
            comfy.model_management.in_training = True
            try:
                yield
            finally:
                comfy.model_management.in_training = previous

        return _numerics()

    @classmethod
    def compute_dtype(cls, model_patcher):
        """The dtype the tower should run in.

        Defaults to the model's manual-cast dtype -- what ComfyUI itself uses to run
        an FP8 or int8 checkpoint -- falling back to its storage dtype. Quantised
        storage dtypes are promoted to bf16 because they cannot be compute dtypes.
        """
        model = model_patcher.model
        dtype = getattr(model, "manual_cast_dtype", None)
        if dtype is None:
            try:
                dtype = model.get_dtype()
            except Exception:  # noqa: BLE001 - a backend need not be a comfy BaseModel
                dtype = torch.float32
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8):
            dtype = torch.bfloat16
        return dtype

    # ---------------------------------------------------------------------------
    # provided
    # ---------------------------------------------------------------------------

    @classmethod
    def select_sites(cls, model_patcher, site_preset):
        """[(module_name, module)] for the chosen preset, in model order."""
        if site_preset not in cls.site_presets:
            raise ValueError(
                "Backend {!r} has no edit-site preset {!r}. Available: {}.".format(
                    cls.key, site_preset,
                    ", ".join(sorted(cls.site_presets)) or "(none)",
                )
            )
        patterns = [re.compile(p) for p in cls.site_presets[site_preset]]
        picked = []
        for name, module in model_patcher.model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            if any(p.match(name) for p in patterns):
                picked.append((name, module))
        if not picked:
            raise ValueError(
                "Edit-site preset {!r} matched no 2D Linear in this model. The "
                "backend regexes are probably out of date with the model code.".format(
                    site_preset
                )
            )
        return picked


def parse_block_spec(spec, n_blocks):
    """'all' | '0-27' | '0,4,10-14' -> set of block indices."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*") or n_blocks <= 0:
        return set(range(max(n_blocks, 0))) or set(range(10000))

    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                low_i, high_i = int(low), int(high)
            except ValueError:
                raise ValueError("Could not parse block range {!r}.".format(part))
            if low_i > high_i:
                low_i, high_i = high_i, low_i
            out.update(range(max(0, low_i), min(n_blocks - 1, high_i) + 1))
        else:
            try:
                index = int(part)
            except ValueError:
                raise ValueError("Could not parse block index {!r}.".format(part))
            if 0 <= index < n_blocks:
                out.add(index)
    if not out:
        raise ValueError(
            "blocks={!r} selected nothing (model has {} blocks).".format(spec, n_blocks))
    return out


@contextlib.contextmanager
def modules_on_device(modules, device):
    """Temporarily relocate whole submodules, restoring their original device after.

    ComfyUI may have the diffusion model parked on the offload device -- that is the
    normal state for a checkpoint too large to stay resident. Rather than force a full
    load, the builder borrows just the text tower. Restoring in ``finally`` keeps the
    patcher's own view of where the model lives consistent.
    """
    origin = []
    for module in modules:
        where = next((p.device for p in module.parameters()), None)
        if where is None:
            where = next((b.device for b in module.buffers()), None)
        origin.append(where)
    try:
        for module in modules:
            module.to(device)
        yield
    finally:
        for module, where in zip(modules, origin):
            if where is not None:
                module.to(where)
