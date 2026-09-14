# How it works

[Back to README](../README.md) · [User guide](USER_GUIDE.md)

## The idea

Suppose the model responds differently to `a portrait of a person, elderly...` and `a portrait of a person, youthful...`. The solver measures that difference across many base scenes and builds a small weight update that tries to reproduce the direction when given neutral prompts.

This is a model edit expressed as a LoRA. It is not conventional image-caption LoRA training, and no external image dataset enters either stage.

## Stage 1: collect a text-space target

For each base scene the code creates neutral, positive and negative prompts. The backend encodes each distinct prompt once and stores detached conditioning tensors on CPU. ComfyUI then unloads models before the selected diffusion-model text modules run.

Forward hooks record each chosen linear layer's input X and output Y. The target begins with **half the positive-minus-negative difference**. This treats the two poles as directions around a neutral reference; a real neutral prediction need not be their exact midpoint.

Two objectives are collected:

| Objective | Target | Tradeoff |
|---|---|---|
| `aligned` | Half the output difference over shared positions up to the shortest sequence length | Local token comparison; can have very weak signal when appended text barely affects the prefix |
| `pooled` | Half the difference between whole-sequence mean outputs | Includes appended tokens, but token counts and averaging can introduce unwanted components |

The code truncates by sequence length rather than proving token IDs match at every position. Both objectives are approximations to a reusable concept direction.

Before solving, the target's component parallel to the neutral output is removed:

```text
deflated_delta = delta - y * dot(delta, y) / dot(y, y)
```

Numerical guards handle very small denominators. This aims to reduce a generic conditioning-amplitude change that could look like contrast or guidance adjustment. It does not mathematically separate every possible nuisance attribute from a concept.

## Stage 1: solve and compress

The code accumulates input covariance over edit scenes, a separate covariance over preservation prompts, and a target/input cross matrix. Schematically:

```text
M  = C_edit + preservation_weight * C_pres
β  = regularization * max(mean(diagonal(M)), 1e-12)
ΔW = eta * D * inverse(M + β I)
```

Here `eta` is the solver's construction strength, divided across sites when multiple sites are selected. The implementation solves the linear system using Cholesky, falling back to a general solve; it does not explicitly form an inverse.

The preservation term penalises changing preservation inputs. Ridge regularisation discourages unstable large updates. Neither guarantees preservation on unseen prompts. Covariances are accumulated per prompt with token-count normalisation, then combined across prompts. Sites with matching inputs can share covariance storage.

A randomised low-rank SVD approximates the update:

```text
ΔW ≈ up @ down
up:   [output_features, rank]
down: [rank, input_features]
```

The factors become ordinary `.lora_up.weight` and `.lora_down.weight` keys. Solver alpha is saved equal to the actual rank, giving an alpha/rank factor of 1.

“Closed form” describes the full ridge solution. Rank truncation, regularisation, preservation and nonlinear downstream processing mean it does not perfectly reproduce all prompt differences. “Exact site” describes a linear relationship at the chosen layer, not perfect concept isolation or exact image matching.

## Why the Krea 2 backend uses txtmlp.3

The supplied backend runs `txtfusion` and `txtmlp`, then edits `diffusion_model.txtmlp.3`, the final text projection before joining image tokens. The separate Qwen text encoder is only used to produce conditioning; its weights are not edited.

The backend describes a text-fusion path followed by two linear layers with a GELU between them. `text_out_proj` selects the final linear. `text_mlp` selects both and is marked approximate because the earlier edit passes through that nonlinearity.

Image tokens do not pass through the selected text-only layer, but changed text conditioning still influences image content downstream. It would be incorrect to conclude that unrelated visual properties cannot change.

Backend architectural assumptions should be rechecked against the exact ComfyUI implementation when adding support or changing checkpoints. Only Krea 2 is included; references to other model families are not support declarations.

## Scaling and verification

First, the solver estimates the update's RMS effect on collected inputs and compares it with the pooled reference direction. It rescales the factors using a normalisation ratio clamped to 0.05–200, multiplied by the per-site construction strength. Degenerate signals can skip scaling. This is a text-space estimate, not an image-level identity.

Key binding is checked against ComfyUI's LoRA map. Zero matching modules raises an error when there are modules to check. Partial binding is not fatal. If the diagnostic itself raises an exception, the helper returns `-1`; this is unknown, not successful binding.

With verification enabled, the code rolls out neutral-prompt latents using Euler updates along the model's sigma schedule. On selected timesteps it compares:

```text
reference = (prediction_positive - prediction_negative) / 2
effect    = prediction_neutral_with_LoRA - prediction_neutral
```

The comparison uses denoised predictions from the actual denoiser. No VAE decoding or visual assessment occurs. The same frozen rollout is reused for candidate objectives.

Across the collected samples:

```text
image_effect  = dot(effect, reference) / (norm(effect) * norm(reference))
relative_size = norm(effect) / norm(reference)
gain          = dot(effect, reference) / dot(effect, effect)
```

Numerical guards are omitted here for clarity. Under `objective = auto`, the chosen candidate maximises:

```text
abs(image_effect) * min(1, relative_size * 4)
```

If the selected absolute cosine exceeds 0.08, the saved edit is multiplied by the signed gain, with magnitude clamped to 0.5–50. A negative gain flips the edit. The saved result is not probed again after this gain is applied, so the nonlinear denoiser can respond differently at the final scale.

If verification is off or fails, automatic selection falls back to the first built objective: `aligned` for `auto`. Text-space scaling remains, but there is no successful denoiser calibration. A caught verification failure does not prevent saving.

## Reading diagnostics accurately

| Field | Interpretation |
|---|---|
| `image_effect` | Signed cosine measured before the final verification gain; not a percentage of image quality |
| `effect_score` socket | Same pre-gain cosine when verification succeeds; otherwise average `token_fit`, a different quantity |
| `size vs prompt` / `relative_size` | Pre-verification-gain magnitude relative to half the positive-minus-negative prediction difference |
| `gain` / `calibrated_gain` | Proposed least-squares gain / gain actually applied to factors |
| `concept_agreement` | Agreement of per-scene target directions; inspect the worst scene too |
| `prompt_magnitude` | Fraction of aligned-target energy removed by deflation; not a general quality score for the pooled objective |
| `token_fit` | Residual-based fit for the low-rank edit before analytic scaling; can be negative |
| `direction_match` | Average directional agreement over scene means |
| `strength_ratio` | Achieved/requested magnitude before analytic scaling |
| `auto_scale` | Analytic factor applied after low-rank compression |
| `leakage` | Text-space preservation movement after analytic scaling, before final verification gain |
| `rank_energy_kept` | Fraction of full update energy retained by the approximation |
| `rank_error_curve` | Approximation residual over probed ranks; not a gradient-training loss |

The report's low/moderate thresholds are heuristics. Its code flags `image_effect < 0.15` as low and `< 0.4` as moderate, using the raw signed measurement. A strong negative measurement may therefore produce a pessimistic note even after a sign correction. Read the gain and compare images before deciding it failed.

Likewise, good cosine with tiny size is not sufficient, low cosine does not by itself mean no visible change, and low text-space leakage does not prove final image preservation. Default probes use only the first three construction scenes, so this is not held-out validation.

## Stage 2: gradient refinement

This stage is optional and experimental. The `lora` → `warm_start` connection is optional independently of using the trainer: without it, training starts from zero adapters; `plan`, MODEL and CLIP are still required. The solver's saved LoRA can be used directly without training.

The trainer freezes base-model weights and creates trainable adapters at backend-selected denoiser modules. Modules present in the warm-start LoRA are added to the target set when found. Existing adapter factors/ranks/alpha are loaded; new adapters are created from the trainer options. Alpha is held fixed.

A frozen target bank is constructed from the first `train_scenes` plan triplets, neutral Euler trajectories and the chosen inclusive timestep window. At each sampled latent:

```text
delta = (frozen_positive - frozen_negative) / 2
target_at_direction_s = frozen_neutral + s * eta * delta
```

`both_per_step` trains at s = +1 and -1 each step; `alternate` alternates per accumulation microstep; `positive_only` trains only +1. MSE is computed either directly on x0 predictions or after converting both sides with `(latent - prediction) / sigma` for the `velocity` option.

Only adapter parameters receive optimiser updates. Gradient checkpointing trades extra computation for activation memory. Matching the target-bank and training numerical modes reduces kernel mismatch. The bank is not refreshed during training.

This is one adapter set refined from a warm start, not a merge of independently trained files. Nevertheless, optimisation can weaken or undo earlier behaviour, and effects from different layers can oppose each other. Improvement is not guaranteed.

No preservation loss from stage one, held-out evaluation, final verification, intermediate save/resume of optimiser state, or automatic best-checkpoint selection is implemented. Final saving occurs after a successful training return. Cancelling a run should not be treated as saving its partial progress.

## Source map

| File | Responsibility |
|---|---|
| `__init__.py` | Registers solver and trainer nodes |
| `nodes.py` | Solver UI, scene/plan assembly, candidate choice, reports |
| `prompt_packs.py` | Concepts, generator axes, preservation data |
| `zt_core.py` | Collection, covariance solve, compression, scaling, saving |
| `zt_verify.py` | Binding checks and denoiser probes |
| `train_nodes.py` | Trainer UI and reports |
| `zt_train.py` | Target bank, gradient loop, cleanup |
| `zt_adapters.py` | Adapter initialisation, optimisers and snapshots |
| `zt_backends/base.py` | Shared backend contract and defaults |
| `zt_backends/krea2.py` | Included model detection and layer presets |
