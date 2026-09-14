# Troubleshooting and FAQ

[Back to README](../README.md) · [User guide](USER_GUIDE.md) · [Settings](NODE_REFERENCE.md)

## Installation and loading

| Symptom | Check and next action |
|---|---|
| ZT nodes missing | Restart; verify the active `custom_nodes` path, directory nesting and `zt_backends/`. Read the first startup import traceback. |
| Missing `comfy.weight_adapter`, `BiasDiff`, or checkpoint helper | This package imports trainer APIs at startup. Use a compatible ComfyUI build; installing a random similarly named pip package will not replace those host APIs. |
| No installed backend recognises MODEL | Only Krea 2 is included. Connect the intended checkpoint and run ZT Slider Info. Another architecture needs a backend. |
| CLIP type `krea2` unavailable | Your active ComfyUI build does not expose the expected encoder loader support. Check the installation/version. |
| Conditioning shape or `_unpack_context` error | Use the matching Qwen3-VL-4B encoder and `krea2` type. File renaming cannot make an incompatible encoder compatible. |
| Model filename absent from dropdown | Re-select the real file. The workflow does not download weights. Confirm paths with the working model workflow. |
| `PreviewAny` or `LossGraphNode` missing | Check your ComfyUI version's display/training nodes. Temporarily remove optional displays and use the reports/log output. This does not fix missing Python APIs required by the node pack itself. |
| Saved file absent from LoRA loader | Read `saved_path`, check the first configured LoRA directory, and refresh the file list or restart. Verify `auto_save` was enabled. |

ComfyUI's [custom-node troubleshooting guide](https://docs.comfy.org/troubleshooting/custom-node-issues) covers identifying extension conflicts. Include the complete traceback when reporting a problem, not only the final line.

## Memory and speed

Encoding, text-module collection, the CPU solve, full-model verification and gradient training have different memory needs. A small saved LoRA does not imply a small construction working set.

- During covariance collection, try `stats_device = cpu`; it moves accumulation storage to RAM but does not eliminate model/activation memory or all device-side temporaries.
- For a CPU solve failure, check available system RAM and use `edit_site = auto`. Reducing rank does not remove the dense covariance matrices. Float32 uses less matrix storage than float64 but can be less numerically stable.
- For verification OOM, lower `probe_resolution`. Fewer probe scenes/steps reduce work and cached samples; they do not reduce the full model's weight size. If temporarily disabling verification, label the result unverified.
- For training OOM, keep gradient checkpointing on, `batch_size = 1`, and lower dimensions or use `attn_only`/`attn_qkv` and a narrower block range. Smaller adapter sets do not eliminate backward activation costs.
- `AdamW8bit` may fall back to AdamW. Read the log; do not assume 8-bit optimiser memory is in use.

No fixed GPU minimum or “seconds on any card” promise is supported. Offloading also needs RAM and can substantially change speed. Long prompts increase encoding and activation work.

## Solve failures

| Message or metric | Meaning and response |
|---|---|
| Every scene rejected / fewer than 3 shared tokens | Supply meaningful base scenes and check the encoder. |
| No concept signal | Shared-prefix targets are effectively zero. Check prompt phrases, encoder and backend text path. The current guard uses aligned signal even if you select pooled. |
| Sites never reached / no 2D Linear matched | Backend patterns and the actual model structure disagree. Restore `edit_site = auto`; report the model/version if still failing. |
| Cholesky failure | A fallback general solve is attempted. If it also fails, use float64 and increase regularisation; include the full error. |
| Zero keys bound | The builder refuses to save this detectable unbound edit. Confirm exact module names and compatible loader/model. |
| Binding `-1/N` | The diagnostic failed internally; binding is unknown, not verified. |
| Partial key binding | Only some modules resolve. It is not rejected automatically; investigate before sharing. |
| Verification failed | A solved file can still be saved with analytic scaling only. Read the error, address memory/compatibility and rerun verification. |

## Weak, reversed or entangled effects

**Zero-training works best for relatively simple concepts the model already understands**, such as photorealistic ↔ anime appearance, adding detail, or changing colour and lighting. These are example targets, not guaranteed successes. Complex concepts and combinations of attributes can exceed what the closed-form edit can express; some will not work even with tuning. Higher weight cannot create a concept direction the edit did not capture. Realistic ↔ anime is an example custom axis, not a separately named built-in preset.

**There is no universal LoRA loading weight.** In the author's experiments, some sliders work around `+2` or `-2`, while others need around `+20` or `-20` to show a useful effect. The suitable range depends on the model, concept and resulting LoRA. Start low and increase gradually while comparing fixed-seed images. ±2 is a starting point, not a hard limit; ±20 is an observed example, not a recommended default or maximum. Stop increasing if distortion grows without improving the intended concept.

**The trainer is optional and experimental, and the `lora` → `warm_start` connection is entirely optional.** You can use the solver's saved LoRA directly without running the trainer. If you try the trainer, connect this wire only to initialise it from the solved weights; leave it disconnected to train from zero. The trainer still requires `plan`, MODEL and CLIP. Experimental refinement is not guaranteed to improve the slider or solve a complex concept.

**The author's separate weight-baking tool is not included.** The author sometimes uses that tool for sliders needing a large loading multiplier and may publish it later. This node's existing construction-strength and automatic-scaling controls are distinct from that separate utility. It is not required to use the saved LoRA with a suitable loading strength.



First compare `-1`, `0`, `+1` at a fixed seed and neutral prompt on the same base checkpoint. Confirm that the sampler consumes the LoRA loader's MODEL output. Temporarily test the slider alone if other adapters are present.

If the two extreme prompts do not produce a useful difference on the base model, rewrite the axis. If they do, inspect concept agreement, relative size, verification completion and leakage. Lowering preservation weight or regularisation can strengthen the fit but increase unrelated changes. Increasing them can reduce drift but suppress the concept. Change one setting at a time.

Raw negative cosine can be corrected by a negative gain in the saved weights. The report and `effect_score` remain pre-gain, and verification is not repeated after gain. If generation still moves the wrong way, report the prompt pair, gain and image comparison. A sign claim is only an intention when verification is skipped, fails or has very weak signal.

If a custom concept misbehaves, remember generated scenes remain present and Custom has no held axes. Extra scenes at the end may not be used by verification or training. More training steps cannot fix an irrelevant target bank.

## Training issues

| Symptom | Response |
|---|---|
| Plan carries no scenes | Connect the solver's `plan`; a STRING/report cannot replace `ZTS_PLAN`. |
| No training presets | The selected backend has solver support only. Training needs explicit presets. |
| Empty timestep window | Use `first_step = 0`, `last_step = -1`, then narrow within actual schedule indices. |
| No modules selected | Restore `target_modules = auto`, `blocks = all`; inspect backend support. |
| Warm start reports zero modules | Confirm the `lora` → `warm_start` wire and use the same base checkpoint. Do not assume connection alone proves weights were loaded. |
| Loss flat | Compare images before tuning; then consider target signal, scene coverage and learning rate. A flat curve alone cannot diagnose convergence. |
| Loss falls but images worsen | Training may overfit its small fixed bank. Compare more prompts; consider fewer steps, a smaller learning rate, different scenes or targets. |
| NaN/Inf or backward error | Return to default FP32 adapters, compatible kernels and checkpointing; report the traceback and precision/loader details. |
| No file after cancellation | Final saving follows successful training. There is no automatic partial checkpoint or optimiser resume. |

## Frequently asked questions

**Is it really zero training?** Stage one uses no gradient training. The optional, experimental trainer explicitly does. Both are driven by text and model predictions.

**Does it change my checkpoint file?** Saved output is a separate adapter. The code does not save changes back to the checkpoint file. Temporary patches/hooks operate in memory during execution.

**Can I use it on SDXL or Flux?** Not with the included backend. A working backend needs model-specific development and validation.

**Can I use a Krea slider on any Krea checkpoint?** Start with the exact checkpoint used to build it. Matching architecture/keys does not guarantee matching semantics on another variant.

**Do I need a trigger word or negative sampler prompt?** No special trigger is required. The custom negative phrase defines the axis during construction, not a sampler input.

**Can it learn a person or new style?** No reference-image supervision exists here. Broad styles already in the model's text prior are possible targets, but new identity/style learning is a different task.

**Can I combine sliders?** You can test stacking compatible model adapters, but their effects can interact. Establish each alone first and use conservative strengths.

**Should I stack the solved and refined file?** For comparing stages, load each separately. The refined file already starts from the solved weights.

**Can I train only from custom scenes?** Not through the current widgets. They append to generated scenes; see the user guide.

**Does the plan guarantee the model matches?** No. It records a backend key, but the trainer does not enforce checkpoint identity against it. Supply the same model and encoder yourself.

**Can I resume training exactly?** There is no optimiser/scheduler-state resume. Reusing adapter weights, where a compatible loader supplies them, would start a new run and is not an exact continuation.

**What should I send in a bug report?** Node and ComfyUI versions/commits, install type, OS, GPU/VRAM and RAM, model/encoder filenames and precision, workflow, full traceback, report and reproducible steps. For visual problems include fixed-seed baseline and both poles. Remove private paths or content before sharing.
