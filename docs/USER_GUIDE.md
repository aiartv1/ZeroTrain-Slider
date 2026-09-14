# User guide

[Back to README](../README.md) · [Settings](NODE_REFERENCE.md) · [Troubleshooting](TROUBLESHOOTING.md)

## 1. What you are making

A slider LoRA is a small adapter that changes a model along a chosen concept axis. For the age preset, the negative end describes youth and the positive end describes older age. You build the file once, then change its loading strength during generation.

There are three different controls called strength or scale in this workflow:

| Control | Where | Purpose |
|---|---|---|
| `strength` | ZT Slider | Requested edit strength during construction; usually leave at 1 |
| `eta` | ZT Slider Trainer | Size of the positive/negative training target |
| `strength_model` | Generation-time LoRA loader | The slider you adjust when generating images; can be negative |

These controls interact but are not interchangeable. Rebuilding is unnecessary just to change generation strength.

## Important: expectations and experimental features

**Zero-training works best for relatively simple concepts the model already understands**, such as photorealistic ↔ anime appearance, adding detail, or changing colour and lighting. These are example targets, not guaranteed successes. Complex concepts and combinations of attributes can exceed what the closed-form edit can express; some will not work even with tuning. Higher weight cannot create a concept direction the edit did not capture. Realistic ↔ anime is an example custom axis, not a separately named built-in preset.

**There is no universal LoRA loading weight.** In the author's experiments, some sliders work around `+2` or `-2`, while others need around `+20` or `-20` to show a useful effect. The suitable range depends on the model, concept and resulting LoRA. Start low and increase gradually while comparing fixed-seed images. ±2 is a starting point, not a hard limit; ±20 is an observed example, not a recommended default or maximum. Stop increasing if distortion grows without improving the intended concept.

**The trainer is optional and experimental, and the `lora` → `warm_start` connection is entirely optional.** You can use the solver's saved LoRA directly without running the trainer. If you try the trainer, connect this wire only to initialise it from the solved weights; leave it disconnected to train from zero. The trainer still requires `plan`, MODEL and CLIP. Experimental refinement is not guaranteed to improve the slider or solve a complex concept.

**The author's separate weight-baking tool is not included.** The author sometimes uses that tool for sliders needing a large loading multiplier and may publish it later. This node's existing construction-strength and automatic-scaling controls are distinct from that separate utility. It is not required to use the saved LoRA with a suitable loading strength.

## 2. Requirements and compatibility

You need a working local ComfyUI installation, a compatible Krea 2 diffusion checkpoint and its matching Qwen3-VL-4B text encoder. The backend checks for ComfyUI's `Krea2` model class. It does not detect SD1.5, SDXL, Flux or other models.

The source imports `comfy.weight_adapter`, `comfy.weight_adapter.bypass` and `comfy_extras.nodes_train`, including checkpointing helpers and `BiasDiff`. Both solver and trainer modules are imported at package startup. Consequently, missing training APIs can prevent even the solver nodes from appearing.

There is no separate requirements file and the declared dependency list is empty. This means the default installation relies on ComfyUI's environment, not that it runs without dependencies. `AdamW8bit` optionally uses bitsandbytes and falls back to AdamW when unavailable.

No tested minimum version, GPU/RAM matrix or complete quantisation support matrix is supplied. The implementation accommodates ComfyUI-managed quantised weights, but do not assume every third-party loader or quantisation format works. First confirm that the checkpoint and encoder work in ordinary image generation.

The solver and trainer do not take a VAE input. A VAE is still needed by your usual image-generation workflow to decode images.

## 3. Installation

### Download ZIP

1. Download the repository using GitHub's **Code → Download ZIP**.
2. Extract it and place the repository contents in `ComfyUI/custom_nodes/ComfyUI-ZeroTrain-Slider/`.
3. Confirm that `__init__.py` sits directly inside that folder, alongside `nodes.py`, and that `zt_backends/krea2.py` exists.
4. Restart ComfyUI and search the node menu for `ZT Slider`.

Avoid an extra nested directory such as `custom_nodes/ComfyUI-ZeroTrain-Slider/ComfyUI-ZeroTrain-Slider-main/__init__.py`. On Desktop or machines with multiple installations, locate the active ComfyUI data/install directory before copying. See the [official installation guide](https://docs.comfy.org/installation/install_custom_node).

### Git installation

Run from your active installation's `custom_nodes` directory:

```shell
git clone https://github.com/aiartv1/ZeroTrain-Slider.git
```

To update a Git installation, open a terminal inside this node's folder and run `git pull --ff-only`, then restart ComfyUI. Save local edits first; if Git reports a conflict or divergence, resolve it rather than discarding changes. For ZIP installations, back up the old folder and replace it with the new complete folder. Keep only one active copy.

## 4. Load the example

Drag [example_workflow.json](../example_workflow.json) onto the canvas or use ComfyUI's workflow-opening control. It includes:

- A diffusion loader and CLIP loader.
- Solver and solver Advanced nodes.
- Trainer and trainer Advanced nodes.
- Two `PreviewAny` report displays, `LossGraphNode` and explanatory notes.

Its saved filenames are `krea2_turbo_fp8.safetensors` and `qwen3vl_4b_fp8_scaled.safetensors`. They are examples, not bundled downloads. Re-select files available in your installation. The diffusion loader's saved dtype is `fp8_e4m3fn`; choose a loader dtype appropriate to your actual checkpoint rather than copying that value blindly. The backend hint mentions another example filename, `krea2_turbo_fp8_scaled.safetensors`.

The example chooses the Style preset and explicit output names `style_zt` and `style_zt_trained`. If you change the concept, clear or change those names too.

Both stages are active in the JSON. For a solver-only first run, mute the trainer, its report preview and its loss graph through the canvas menu. The original workflow notes suggest `Ctrl+M` for muting selected nodes. If shortcut bindings differ, use the UI menu. Restore those nodes when ready to train.

The canvas notes have been updated to explain experimental training, optional warm start and variable loading weights. Model filenames, connections and execution settings remain those of the supplied workflow.

## 5. Build a solver-only slider

Connect the diffusion loader's MODEL and the CLIP loader's CLIP to ZT Slider. Set CLIP type to `krea2`. Advanced is optional; its defaults are already used when disconnected.

Start with 40 scenes, rank 16, strength 1, verification on and automatic saving on. Choose one preset, run, and wait through encoding, the solve and verification. Verification exercises the whole denoiser and can be the longest part.

Read the report before using the file. Check that the expected backend and concept were selected, keys bound, verification completed, and no major failure note was printed. A file may still save when verification fails; the report tells you whether the measurement succeeded.

## 6. Generate a controlled comparison

1. Open an image-generation workflow that already works with the same checkpoint.
2. Insert **Load LoRA (Model Only)** (`LoraLoaderModelOnly`) between the model loader and sampler/model-processing chain. Select the saved file.
3. Use a neutral prompt that leaves the concept unspecified. For an age test: `a portrait of a person beside a window`.
4. Keep the seed, prompt, dimensions, sampler, scheduler, steps and guidance fixed.
5. Generate at `-1`, `0`, `+1`. Add `-0.5` and `+0.5` for finer comparison; try ±2, then gradually higher magnitudes if needed. Values such as 5, 10 and 20 are exploratory examples, not required settings. Inspect the intended effect and unwanted changes at every step.
6. Test several different prompts and seeds, including subjects outside the concept's main domain.

The adapter edits layers in the diffusion model's text-conditioning path, not the separately loaded CLIP encoder. Use model-only loading. If using a general model+CLIP LoRA loader, leave the CLIP strength at zero.

No special trigger token is required. Avoid describing the opposite end of the concept strongly in the prompt during your initial test. Large strengths can distort composition, colour, anatomy or unrelated details. Different concepts and checkpoints may need different scales.

For a preliminary check, generate the two extreme prompts on the base model with the same seed. A visible difference shows the model has a signal to work with, but does not prove a low-rank slider will reproduce it.

## 7. Optional, experimental refinement

The solver's saved LoRA is usable on its own. The whole trainer stage is optional and experimental. If using it, connect the **same original model and encoder**, plus solver `plan` → trainer `plan`. **The `lora` → `warm_start` wire is optional:** connect it for warm-start refinement, or leave it disconnected for training from zero. When connected, do not first apply the solved LoRA to the trainer's MODEL input; `warm_start` already supplies its weights.

Begin with the defaults: 150 steps, learning rate 0.00005, rank 16, eta 2, resolution 512 × 512 and gradient checkpointing on. These are starting settings from the implementation, not measured guarantees of convergence.

Select `LoRA` explicitly in Train Advanced for ordinary slider refinement. The default algorithm is read from ComfyUI's adapter registry and can depend on that installation. Warm-started modules retain their loaded rank/alpha; new modules use the trainer settings.

The plan carries the resolved concept and all scene triplets. By default the trainer uses only the **first 3** triplets. Its target bank is computed once and reused. More steps do not add new scenes or fresh target trajectories. Increase `train_scenes` deliberately when broader coverage is needed.

With a warm start connected, the trainer saves one adapter containing the refined warm-start module and additional trained modules. Compare it against the solver file **separately**, using identical generation settings. Loading both together applies the original edit again. A falling training loss is not sufficient evidence that the new file is better.

The trainer does not automatically run the solver's verification on its finished file. It also does not apply the solver's preservation-prompt penalty. Recheck unrelated prompts after refinement.

For training from zero, leave `warm_start` disconnected; `plan` remains required. In the supplied graph the solver still runs to create that plan, even if its weights are not used. There is no standalone plan-builder node.

## 8. Custom concepts

Choose `Custom`, fill both phrases, and use a short descriptive name. For example:

```text
custom_name: fabric_texture
custom_positive: coarse woven fabric, clearly visible thick fibres, rough textile surface
custom_negative: fine smooth fabric, tightly woven delicate fibres, smooth textile surface
```

Use contrasting descriptions of one attribute, with similar specificity. Avoid changing the subject, camera, lighting and style at the same time. The negative phrase is the negative end of the slider, **not the sampler's negative-prompt field**.

Optional `custom_scenes` accepts one neutral scene per line:

```text
# Leave the fabric texture unspecified
a fabric bag resting on a table
a folded scarf on a shelf
a cloth cushion on a chair
```

Blank lines and lines beginning with `#` are ignored. Duplicate base prompts are removed. The concept phrases are appended automatically.

**Current UI limitation:** these scenes are appended to the generated set; they do not replace it. `scenes` cannot be set below 4. There is no exposed `hold_axes` widget. Custom concepts use no held axes, so generated prompts may mention the very attribute you want to control. Built-in presets hold specified generator axes automatically, but that does not remove concept words embedded in individual subjects or contexts.

Because probes and training take the first triplets, appended custom scenes can be omitted entirely at default settings. Increasing `probe_scenes` or `train_scenes` only helps if their limits (16 and 32) reach those appended entries. For a truly custom-only set or precise axis exclusions, a code change to scene construction is needed; this UI cannot promise it.

Choose a scene pack suited to the domain. For global appearance concepts, consider `bare_subjects` preservation; Custom otherwise defaults to `scenes_and_objects`. Read agreement and leakage diagnostics and inspect images rather than assuming an automatic choice is conflict-free.

## 9. Files, caching and sharing

Files are saved to the **first configured `loras` directory**, usually `ComfyUI/models/loras/`. If no LoRA directory is configured, the saver falls back to ComfyUI's output directory. `saved_path` is authoritative.

Names contain a sanitized stem and Unix timestamp in seconds:

```text
age_zt_slider_r16_<timestamp>.safetensors
age_zt_trained_r16_<timestamp>.safetensors
```

Enter a stem rather than a full path or extension in `save_name`. The saver appends the extension and timestamp. The timestamp is not a collision-proof counter; equal stems saved within one second can target the same path.

With `auto_save = false`, `saved_path` is empty and the adapter remains on the live `LORA_MODEL` output. That socket is an in-memory adapter, not a MODEL or filename. Connect it only to nodes accepting that type, or enable auto-save and load the resulting file normally.

ComfyUI may reuse cached node results for unchanged inputs. Re-queuing is not a guarantee of a new solve, training run or file. Change an intended input, such as pack/training seed, when you want a new experiment.

Share the LoRA with its base checkpoint name, node version/commit, concept phrases, build settings, recommended strength range, comparison images and workflow. Save the report separately; file metadata is not a complete reproducibility record. The solver metadata version currently says `2.0.0` while package metadata says `3.0.1`.

Pack seeds make prompt selection repeatable and training seeds control target noise/sample selection. Randomised SVD, adapter initialisation, hardware and kernels mean byte-identical rebuilds are not guaranteed.
