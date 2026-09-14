# Adding a model backend

[Back to README](README.md) · [Technical explanation](docs/HOW_IT_WORKS.md)

Only Krea 2 is supplied. A new Python file can register another backend, but identifying valid edit sites and validating conditioning, sampling and training can require substantial work. Automatic discovery is not automatic model compatibility.

## 1. Study the model's real conditioning path

Find a two-dimensional linear weight whose input depends only on text conditioning. Prefer the last linear before image/text interaction. Confirm the layer is reached by actual conditioning flow and recognised by ComfyUI's LoRA key map.

Do not assume architecture labels such as single-stream, dual-stream or cross-attention establish correctness. Inspect the exact model implementation and version. A text-only projection is isolated at its input; changing it can still alter all downstream image properties.

`exact_site_presets` should identify sites whose edit reaches the relevant text output through a linear path. Ridge, rank truncation and unreachable targets still cause approximation error. Do not describe an exact site as a guarantee of exact images.

The collection code rejects negligible aligned shared-prefix signal, even for the pooled objective. A purely position-wise path receiving an unchanged causal prefix can fail. Ensure the captured features actually carry the concept signal.

## 2. Implement the contract

Create `zt_backends/my_model.py`. The registry scans modules except `base` and names starting with `_`, discovers `SliderBackend` subclasses and records import errors for the Info node. Restart after adding a file. Use a unique backend key.

The following is a structural example only. `MyModel`, `text_adapter` and the model filenames must be replaced with real architecture-specific values:

```python
from .base import SliderBackend


class MyModelBackend(SliderBackend):
    key = "my_model"
    display_name = "My Model"
    model_hint = "Describe the exact loader and matching encoder here"
    site_presets = {
        "text_out_proj": (r"^diffusion_model\.text_adapter\.out$",),
    }
    default_site_preset = "text_out_proj"
    exact_site_presets = ("text_out_proj",)
    notes = "Explain the edit site and known restrictions."

    @classmethod
    def matches(cls, model_patcher):
        import comfy.model_base
        model_cls = getattr(comfy.model_base, "MyModel", None)
        return model_cls is not None and isinstance(model_patcher.model, model_cls)

    @classmethod
    def encode(cls, clip, text):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        return cond[0][0].detach().to("cpu", copy=True)

    @classmethod
    def text_path_modules(cls, model_patcher):
        return [model_patcher.model.diffusion_model.text_adapter]

    @classmethod
    def run_text_path(cls, model_patcher, cond, device):
        tensor = cond.to(device=device, dtype=cls.compute_dtype(model_patcher))
        return model_patcher.model.diffusion_model.text_adapter(tensor)
```

`encode` runs one prompt at a time. Preserve the actual tensor layout the model requires; the generic illustration above is not suitable for every encoder. `run_text_path` is called with forward hooks and no gradients; it must execute every selected site. Its return value is ignored by collection. `text_path_modules` lists the modules temporarily moved to the compute device.

Override `describe` and `compute_dtype` when appropriate. Backend `default_eta` and `default_rank` attributes exist but the current node widgets use their own hard-coded defaults; setting these attributes alone does not change widget defaults.

## 3. Check verification compatibility

`latent_shape` and `sigmas` have defaults reading ComfyUI model configuration, including spatial alignment and a single temporal frame when appropriate. Verify these assumptions rather than relying on them blindly.

Verification and training currently reconstruct conditioning as a tensor plus an empty metadata dictionary. Models requiring pooled embeddings, masks, multiple encoders or other conditioning metadata may need shared-pipeline changes as well as a backend. A single-file backend is not sufficient when the generic contract cannot represent the model's inputs.

Do not assume that support for an FP8 checkpoint establishes support for every quantised wrapper. Check actual module weights, forward hooks, device relocation and saved-key loading.

## 4. Optional training

Declare `train_site_presets`, `default_train_preset`, and optionally `block_index_regex` with one capture group for the block number. Copy the structure from `krea2.py`, substituting verified regexes for your architecture.

`checkpoint_types` should select suitable block classes for activation recomputation. Review `training_numerics`, the model's prediction convention, and whether the velocity conversion used in `zt_train.py` is valid for it. Solver compatibility does not prove gradient-training compatibility.

If training presets are empty, the trainer rejects the backend. Warm-start modules are unioned into the selected targets when found, but unresolved warm-start names can be skipped. Verify the actual warm-start count.

## 5. Validation before advertising support

1. Run ZT Slider Info with the loaded model. Check backend, site names, dimensions and latent shape.
2. Build an obvious supported concept and confirm all intended sites bind, not merely that the count is nonzero.
3. Run both objectives and full verification. Inspect magnitude, signed gain, agreement and leakage.
4. Load the saved file through stock ComfyUI loading. Generate neutral-prompt strength sweeps at fixed seeds.
5. Test unseen prompts, several seeds and unrelated preservation subjects; include failures.
6. If training is included, check from-zero and warm-start behaviour, loaded rank/alpha, loss finiteness and post-training images.
7. Test cleanup after cancellation/errors and run ordinary generation afterward.
8. Record exact ComfyUI/PyTorch versions, checkpoint/encoder, precision, GPU, RAM, peak memory and timing before making compatibility/performance claims.

Diagnostic success alone is not sufficient. Submit the backend with a minimal workflow and reproducible comparisons; see [CONTRIBUTING.md](CONTRIBUTING.md).
