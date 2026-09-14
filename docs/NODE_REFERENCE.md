# Node reference

## Practical tuning notes

Start with defaults and change one setting at a time. Bounds below describe what the UI accepts, not a promise that every combination fits memory or works well.

### Solver controls

- `scenes`: generated base-scene count before appended custom scenes and deduplication. More varied scenes increase coverage and encoding cost; 24–64 is a reasonable exploratory band from the source guidance.
- `rank`: factorisation capacity. Try 8–32 before the upper limits. It does not control dense covariance size. The rank curve reports the first selected site's approximation error.
- `strength`: construction scale. Leave at 1 and adjust the loader strength for image tests.
- `verify`: probes denoiser predictions, chooses candidates under `auto`, and can apply signed calibration. Off/failed verification leaves `auto` on `aligned`.
- `custom_positive`, `custom_negative`, `custom_name`: used only for the Custom preset. Both phrases are required there. `custom_scenes` can augment any concept.
- `preservation_weight`: increase to discourage changes on preservation inputs; decrease if preservation is suppressing the concept. Leakage and strength can trade off.
- `regularization`: ridge relative to covariance scale; increasing can stabilise the solve at the cost of fit. The source suggests exploring roughly 0.005–0.1.
- `scene_pack`, `preservation_pack`: `auto` follows the concept's data. A scene-pack override still uses that concept's declared held axes in the current implementation. Custom declares none.
- `preservation_count`: maximum requested prompts, capped by the finite pack. Zero or `off` disables that constraint.
- `pack_seed`: shuffles prompt selection and seeds probe noise; does not seed every numerical operation.
- `objective`: `aligned`, `pooled` or automatic candidate comparison. See the technical guide before forcing one.
- `edit_site`: Krea 2 `auto` resolves to `text_out_proj` (`diffusion_model.txtmlp.3`); `text_mlp` selects `.1` and `.3` and is approximate.
- `probe_scenes`: takes the first N scene triplets. `probe_steps`: number of selected usable timesteps per probe. `probe_resolution`: square probe size. These are diagnostics, not final image-output settings.
- `schedule_steps`: length of the probe rollout schedule; align it with your intended model usage. Verification uses the backend's default scheduler.
- `solve_precision`: CPU linear-algebra precision; float64 is the default for stability. `stats_device`: accumulation location, not the solve location. `save_dtype`: precision of saved factors.

### Trainer controls

The trainer is optional and experimental. `warm_start` is an optional input; connect the solver's `lora` only to initialise from its weights. `plan`, MODEL and CLIP remain required when training. Loading weights during generation can vary substantially: the author reports useful values around ±2 for some sliders and around ±20 for others. These are not the bounds of the solver's construction-strength widget.

- `steps`: optimiser updates, not a count of unique scenes. `learning_rate`: update size; high values can undo a warm start. `eta`: target displacement; increasing can strengthen the target but also entangle changes.
- `rank`: new-adapter rank. Existing warm-start ranks remain. `alpha` likewise configures new adapters while loaded alpha is preserved and frozen.
- `width`, `height`: target-bank/training dimensions, not a promised inference resolution. `gradient_checkpointing`: recompute activations to save memory. `training_seed`: target noise and bank-sample selection.
- `target_modules`: Krea 2 `auto` is `attn_mlp`. `attn_only` includes q/k/v, gate and output projections; `attn_qkv` selects q/k/v only; `mlp_only` selects gate/up/down. `attn_mlp` unions attention and MLP sites.
- `blocks`: zero-based inclusive indices, such as `8-20` or `0,4,10-14`, or `all`. Matched warm-start text modules are added independently of the denoiser block filter.
- `algorithm`: dynamically provided by ComfyUI; select `LoRA` for the intended workflow. Other families are exposed but not established as equivalent slider methods.
- `optimizer`: default AdamW; Adam, SGD (momentum 0.9) and RMSprop are alternatives. AdamW8bit attempts bitsandbytes, with AdamW fallback.
- `lr_scheduler`: cosine/linear decay or constant rate after warmup. `warmup_steps` is clamped below total steps. `weight_decay` is passed to the selected optimiser. `max_grad_norm = 0` disables clipping.
- `train_scenes`: first N plan triplets, capped by available scenes. `batch_size`: latent batch size. `grad_accumulation_steps`: microsteps per optimiser update, trading more work for accumulated gradients without increasing the latent batch itself.
- `direction_mode`: both poles per microstep, alternating poles across microsteps, or only the positive pole. Positive-only training does not enforce the negative end.
- `loss_space`: velocity uses sigma-scaled predictions; x0 uses direct denoised prediction MSE.
- `schedule_steps`, `scheduler`: target-rollout schedule. `first_step`/`last_step`: inclusive schedule indices; -1 means the last nonterminal index. There must be at least one usable nonzero-sigma timestep.
- `lora_dtype`: FP32 is the default to retain small updates. `save_dtype`: output factor precision; alpha remains FP32.

### Outputs and utility nodes

`saved_path` is blank when auto-save is off. Reports are STRING outputs and are also sent to node UI/logs. `effect_score` is not always the same metric: it is measured cosine on successful verification and average token fit otherwise. Trainer `total_steps` reports requested completed-loop steps. `loss_map` is the training loss history; solver `rank_error_curve` uses LOSS_MAP only for graph compatibility.

ZT Slider Info has no required inputs. Without MODEL it lists backends and import errors; with MODEL it adds detection, selected sites and latent shape. It is an inspection node and does not create a slider. The two Advanced nodes only construct option dictionaries; connect the matching one to its solver or trainer.

[Back to README](../README.md) · [User guide](USER_GUIDE.md)

Widget names, defaults and bounds below are extracted from the supplied `INPUT_TYPES` definitions. Tooltips are summarised in the usage notes below; claims of speed or guaranteed success in source comments are not validation results.

## Socket types

`MODEL` and `CLIP` are ComfyUI objects. `LORA_MODEL` is an in-memory adapter state, not a patched MODEL. `ZTS_PLAN` carries the resolved concept and scene triplets. `ZTS_OPTS` and `ZTS_TRAIN_OPTS` are different option dictionaries. `LOSS_MAP` holds a list under `loss`. Do not interchange these types.

## ZTSlider

| Input | Connection/widget | Default | Choices or range |
|---|---|---|---|
| `model` (required) | MODEL | — | (blank) |
| `clip` (required) | CLIP | — | (blank) |
| `concept` (required) | choice | Age (young <-> old) | installation/preset list; see notes |
| `scenes` (required) | INT | 40 | 4 to 256 |
| `rank` (required) | INT | 16 | 1 to 128 |
| `strength` (required) | FLOAT | 1.0 | 0.05 to 10.0; increment 0.05 |
| `verify` (required) | BOOLEAN | true | (blank) |
| `auto_save` (required) | BOOLEAN | true | (blank) |
| `save_name` (required) | STRING | (blank) | (blank) |
| `custom_positive` (optional) | STRING | (blank) | (blank) |
| `custom_negative` (optional) | STRING | (blank) | (blank) |
| `custom_name` (optional) | STRING | (blank) | (blank) |
| `custom_scenes` (optional) | STRING | (blank) | (blank) |
| `advanced` (optional) | ZTS_OPTS | — | (blank) |

Outputs: `lora` (LORA_MODEL), `plan` (ZTS_PLAN), `rank_error_curve` (LOSS_MAP), `saved_path` (STRING), `report` (STRING), `effect_score` (FLOAT).

## ZTSliderAdvanced

| Input | Connection/widget | Default | Choices or range |
|---|---|---|---|
| `preservation_weight` (required) | FLOAT | 1.0 | 0.0 to 20.0; increment 0.05 |
| `regularization` (required) | FLOAT | 0.02 | 0.0001 to 1.0; increment 0.0001 |
| `preservation_pack` (required) | choice | auto | installation/preset list; see notes |
| `preservation_count` (required) | INT | 24 | 0 to 256 |
| `scene_pack` (required) | choice | auto | installation/preset list; see notes |
| `pack_seed` (required) | INT | 0 | 0 to 4294967295 |
| `objective` (optional) | choice | auto | installation/preset list; see notes |
| `edit_site` (optional) | choice | auto | installation/preset list; see notes |
| `probe_scenes` (optional) | INT | 3 | 1 to 16 |
| `probe_steps` (optional) | INT | 3 | 1 to 16 |
| `probe_resolution` (optional) | INT | 512 | 256 to 1536; increment 64 |
| `schedule_steps` (optional) | INT | 8 | 4 to 50 |
| `solve_precision` (optional) | choice | float64 | float64, float32 |
| `stats_device` (optional) | choice | auto | auto, cpu |
| `save_dtype` (optional) | choice | fp32 | fp32, fp16, bf16 |

Outputs: `advanced` (ZTS_OPTS).

## ZTSliderInfo

| Input | Connection/widget | Default | Choices or range |
|---|---|---|---|
| `model` (optional) | MODEL | — | (blank) |

Outputs: `info` (STRING).

## ZTSliderTrainer

| Input | Connection/widget | Default | Choices or range |
|---|---|---|---|
| `model` (required) | MODEL | — | (blank) |
| `clip` (required) | CLIP | — | (blank) |
| `plan` (required) | ZTS_PLAN | — | (blank) |
| `steps` (required) | INT | 150 | 10 to 20000 |
| `learning_rate` (required) | FLOAT | 5e-05 | 1e-07 to 0.01; increment 1e-07 |
| `rank` (required) | INT | 16 | 1 to 128 |
| `eta` (required) | FLOAT | 2.0 | 0.1 to 20.0; increment 0.1 |
| `width` (required) | INT | 512 | 128 to 2048; increment 16 |
| `height` (required) | INT | 512 | 128 to 2048; increment 16 |
| `gradient_checkpointing` (required) | BOOLEAN | true | (blank) |
| `training_seed` (required) | INT | 0 | 0 to 18446744073709551615 |
| `auto_save` (required) | BOOLEAN | true | (blank) |
| `save_name` (required) | STRING | (blank) | (blank) |
| `warm_start` (optional) | LORA_MODEL | — | (blank) |
| `advanced` (optional) | ZTS_TRAIN_OPTS | — | (blank) |

Outputs: `lora` (LORA_MODEL), `loss_map` (LOSS_MAP), `total_steps` (INT), `saved_path` (STRING), `report` (STRING).

## ZTSliderTrainAdvanced

| Input | Connection/widget | Default | Choices or range |
|---|---|---|---|
| `target_modules` (required) | choice | auto | installation/preset list; see notes |
| `blocks` (required) | STRING | all | (blank) |
| `alpha` (required) | FLOAT | 1.0 | 0.01 to 128.0; increment 0.01 |
| `optimizer` (required) | choice | AdamW | AdamW, AdamW8bit, Adam, SGD, RMSprop |
| `lr_scheduler` (required) | choice | cosine | cosine, linear, constant |
| `warmup_steps` (required) | INT | 5 | 0 to 1000 |
| `weight_decay` (required) | FLOAT | 0.0 | 0.0 to 1.0; increment 0.001 |
| `max_grad_norm` (required) | FLOAT | 1.0 | 0.0 to 100.0; increment 0.1 |
| `train_scenes` (required) | INT | 3 | 1 to 32 |
| `grad_accumulation_steps` (required) | INT | 1 | 1 to 64 |
| `batch_size` (required) | INT | 1 | 1 to 8 |
| `direction_mode` (optional) | choice | both_per_step | both_per_step, alternate, positive_only |
| `loss_space` (optional) | choice | velocity | velocity, x0 |
| `schedule_steps` (optional) | INT | 8 | 4 to 50 |
| `scheduler` (optional) | choice | simple | installation/preset list; see notes |
| `first_step` (optional) | INT | 0 | 0 to 100 |
| `last_step` (optional) | INT | -1 | -1 to 100 |
| `algorithm` (optional) | choice | first installed adapter algorithm | installation/preset list; see notes |
| `lora_dtype` (optional) | choice | float32 | float32, bfloat16 |
| `save_dtype` (optional) | choice | float32 | float32, float16, bfloat16 |

Outputs: `advanced` (ZTS_TRAIN_OPTS).

## Concept presets

The phrases below are the exact supplied targets. Positive/negative are intended loading directions; verify them visually.

| Preset | Negative phrase | Positive phrase | Scene pack / held axes / preservation |
|---|---|---|---|
| Age (young <-> old) | very young, youthful, smooth unlined skin, fresh and unaged features | elderly, aged, deeply wrinkled skin, grey thinning hair, sagging features | `people_portrait` / none / `scenes_and_objects` |
| Body weight (slim <-> heavy) | slim, slender, lean thin body, narrow frame, low body fat | heavyset, overweight, full round body, thick torso and limbs, soft heavy build | `people_full_body` / none / `scenes_and_objects` |
| Muscularity (soft <-> muscular) | soft untoned body, no visible muscle definition, sedentary physique | very muscular, heavily defined muscles, athletic physique, visible definition | `people_full_body` / none / `scenes_and_objects` |
| Expression (neutral <-> big smile) | flat neutral expression, unsmiling, blank affect, expressionless | beaming wide smile, joyful laughing expression, bright happy eyes | `people_portrait` / none / `scenes_and_objects` |
| Hair length (short <-> long) | very short cropped hair, close-shorn, minimal hair length | very long flowing hair cascading well past the shoulders | `people_portrait` / none / `scenes_and_objects` |
| Detail (smooth <-> highly detailed) | smooth flat simplified shapes, minimal texture, soft low-detail rendering | extremely detailed, intricate fine micro-texture, razor sharp, rich surface detail | `mixed_general` / look / `bare_subjects` |
| Lighting (dim <-> bright) | dimly lit, deep shadow, low-key underexposed, murky darkness | brightly lit, strong luminous key light, high-key exposure, radiant illumination | `mixed_general` / context / `bare_subjects` |
| Depth of field (deep <-> shallow bokeh) | deep focus, everything sharp front to back, crisp detailed background | shallow depth of field, creamy bokeh, strongly blurred background, subject isolated | `mixed_general` / framing / `bare_subjects` |
| Camera distance (close-up <-> wide shot) | extreme close-up, subject filling the frame, tightly cropped macro framing | wide establishing shot, subject small in a large environment, distant framing | `mixed_general` / framing / `bare_subjects` |
| Colour (muted <-> vivid) | desaturated, muted washed-out palette, near-monochrome, faded colours | vividly saturated, intense punchy colours, high chroma, bold colour contrast | `mixed_general` / look / `bare_subjects` |
| Style (illustrated <-> photoreal) | stylised digital illustration, painterly rendering, obvious brushwork | photorealistic photograph, real camera optics, lifelike skin and materials | `mixed_general` / look / `bare_subjects` |
| Weather (clear <-> stormy) | clear bright sky, calm still air, cloudless and settled weather | stormy overcast sky, heavy dark clouds, rain and wind, dramatic weather | `landscape` / context / `bare_subjects` |
| Time of day (day <-> night) | in broad daylight, bright midday sun, full daytime illumination | at night, after dark, artificial lights against a black sky | `landscape` / context / `bare_subjects` |
| Age of scene (new <-> weathered) | brand new and pristine, spotless unworn surfaces, factory fresh | old and weathered, worn peeling surfaces, rust and patina, decayed with age | `objects` / none / `scenes_and_objects` |
| Crowding (empty <-> crowded) | empty and deserted, no people at all, silent and vacant | crowded and busy, packed with many people, dense bustling activity | `architecture` / none / `scenes_and_objects` |

## Pack choices

Scene packs: `people_portrait`, `people_full_body`, `landscape`, `objects`, `architecture`, `animals`, `anime_illustration`, `mixed_general`.

Preservation packs: `scenes_and_objects` (30 prompts), `bare_subjects` (32 prompts), `people_and_faces` (16 prompts), `style_and_medium` (14 prompts). `off` disables preservation; `auto` follows the concept. Requested preservation counts are capped by the available pack size, with no generated expansion.
