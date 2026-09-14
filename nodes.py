"""Node classes for the zero-training slider builder.

Two nodes, plus an optional third that only prints things:

    Load Diffusion Model ─┐
    CLIPLoader ───────────┴─► ZT Slider  ──► saves into loras/
                                 ▲
                        ZT Slider Advanced   (optional, solver knobs)

Everything the old Concept / Prompt Pack / Preservation nodes did is now widgets on the
one build node, because in practice all three were always wired the same way and the
choices interact — the concept determines which scenes and which preservation set make
sense, so splitting them across nodes mostly created ways to combine them wrongly.
"""

import logging
import time

import torch

from . import prompt_packs as packs
from . import zt_backends
from .zt_core import (
    LOG, OBJECTIVES, build_lora, collect_statistics, rescale_lora, sanitize_name,
    save_lora,
)
from .zt_verify import assert_binds, calibrate

CATEGORY = "training/zero-train slider"

DEFAULT_OPTS = {
    "edit_site": "auto",
    "preservation_pack": "auto",
    "preservation_count": 24,
    "preservation_weight": 1.0,
    "regularization": 0.02,
    "scene_pack": "auto",
    "pack_seed": 0,
    "solve_precision": "float64",
    "stats_device": "auto",
    "save_dtype": "fp32",
    "objective": "auto",
    "probe_scenes": 3,
    "probe_steps": 3,
    "probe_resolution": 512,
    "schedule_steps": 8,
}

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _score(metrics):
    """Rank a candidate objective by how much of the prompt difference it delivers.

    Alignment alone is not enough: an edit can point exactly the right way and still be
    too small to see, and one that is large but crooked is worse than useless. The size
    term is capped at 1.0 because overshooting the prompt difference is not a virtue --
    past that the gain handles scaling.
    """
    # Absolute value: an anti-correlated edit is not a failure, it is a sign error, and
    # the calibrated gain negates it. What disqualifies a candidate is being unrelated
    # (cosine near zero) or too small to see, not pointing backwards.
    return abs(metrics["image_effect"]) * min(1.0, metrics["relative_size"] * 4.0)


# --------------------------------------------------------------------------------------
# ZT Slider
# --------------------------------------------------------------------------------------

class ZTSlider:
    """Builds a concept-slider LoRA by solving for it instead of training it.

    Encodes each scene three ways -- neutral, concept-positive, concept-negative -- reads
    the concept off the model's own text tower, and solves one ridge least-squares
    problem for the weight update that adds it. No optimiser, no gradients.

    With ``verify`` on it then runs real denoiser passes through the whole model to
    measure what the finished LoRA actually does to an image, and scales it so that
    strength 1.0 is as strong as writing the concept into the prompt. That check takes a
    couple of minutes and is worth it: the solve happens in text-conditioning space, and
    without it there is no way to tell a working slider from an invisible one except by
    generating images and guessing.

    WHAT THIS CAN DO: anything reachable by editing the prompt -- age, weight,
    expression, detail, lighting, colour, framing, weather, wear.

    WHAT IT CANNOT: anything outside the text prior -- character identity, an artist's
    linework, pose, hand geometry, object count. The test that predicts it: write the two
    extreme prompts, generate both at one seed. If the model does not move, this will not
    move it either.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The diffusion model. Only its text tower is edited; the "
                               "rest is read but never modified.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "Its text encoder. Used for the prompt encodes, then "
                               "unloaded before anything else runs.",
                }),
                "concept": (
                    list(packs.CONCEPTS.keys()) + ["Custom"],
                    {"default": packs.DEFAULT_CONCEPT,
                     "tooltip": "The axis to build. Each preset also picks the scene "
                                "family and preservation set that suit it -- those pair "
                                "with the concept and choosing them by hand is the "
                                "easiest way to get a dead slider."},
                ),
                "scenes": ("INT", {
                    "default": 40, "min": 4, "max": 256,
                    "tooltip": "How many varied scenes the concept is measured on. This is "
                               "the generalisation knob: one scene gives an edit that "
                               "works on one prompt. 24-64 is the useful range; past that "
                               "you are mostly paying encode time.",
                }),
                "rank": ("INT", {
                    "default": 16, "min": 1, "max": 128,
                    "tooltip": "LoRA rank. 8-32 is sensible. Feed rank_error_curve to Plot "
                               "Loss Graph to see where the curve flattens.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05,
                    "tooltip": "Concept strength baked into the weights. With verify on, "
                               "1.0 is calibrated to mean 'as strong as writing the "
                               "concept into the prompt', so leave it here and use the "
                               "LoRA strength dial at generation time.",
                }),
                "verify": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run real denoiser passes afterwards to measure the effect "
                               "on the image and auto-scale to match the prompt "
                               "difference. Adds a couple of minutes and produces the only "
                               "number that actually predicts whether the slider works. "
                               "Turn off only when iterating on solver settings.",
                }),
                "auto_save": ("BOOLEAN", {"default": True}),
                "save_name": ("STRING", {
                    "default": "",
                    "tooltip": "Blank auto-names from the concept and rank.",
                }),
            },
            "optional": {
                "custom_positive": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Custom concept only. The +1 end, as a phrase that would be "
                               "appended to a prompt.",
                }),
                "custom_negative": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Custom concept only. The -1 end. Make it a real opposite, "
                               "not an absence -- the direction is the difference between "
                               "the two, so a vague negative halves the signal.",
                }),
                "custom_name": ("STRING", {
                    "default": "", "tooltip": "Custom concept only. Names the saved file.",
                }),
                "custom_scenes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One base scene per line, added to the generated set (# "
                               "comments ignored). Write them as plain scenes that do NOT "
                               "mention the attribute the slider controls -- the concept "
                               "phrases are appended automatically, and a scene that "
                               "already names the attribute contradicts them.",
                }),
                "advanced": ("ZTS_OPTS", {
                    "tooltip": "Optional ZT Slider Advanced node. Sensible defaults apply "
                               "when nothing is connected.",
                }),
            },
        }

    RETURN_TYPES = ("LORA_MODEL", "ZTS_PLAN", "LOSS_MAP", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("lora", "plan", "rank_error_curve", "saved_path", "report",
                    "effect_score")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    # ---------------------------------------------------------------------------------

    def build(self, model, clip, concept, scenes, rank, strength, verify, auto_save,
              save_name, custom_positive="", custom_negative="", custom_name="",
              custom_scenes="", advanced=None):
        started = time.time()
        opts = dict(DEFAULT_OPTS)
        if advanced:
            opts.update(advanced)

        backend = zt_backends.detect(model)
        spec = self._resolve_concept(concept, custom_positive, custom_negative, custom_name)
        triplets, scene_pack = self._build_scenes(spec, scenes, custom_scenes, opts)
        preserve, preserve_pack = self._build_preservation(spec, opts)

        site_preset = opts["edit_site"]
        if site_preset == "auto":
            site_preset = backend.default_site_preset
        if site_preset not in backend.site_presets:
            raise ValueError(
                "{} edit_site {!r} is not a preset of the {} backend. It offers: {}. Set "
                "it back to 'auto' unless you know why you changed it.".format(
                    LOG, site_preset, backend.display_name,
                    ", ".join(sorted(backend.site_presets))))
        exact = site_preset in backend.exact_site_presets

        stats_device = (torch.device("cpu") if opts["stats_device"] == "cpu"
                        else model.load_device)

        logging.info("%s Building '%s' on %s: %d scenes, %d preservation prompts, "
                     "site '%s'.", LOG, spec["name"], backend.display_name,
                     len(triplets), len(preserve), site_preset)

        stats, sites, conds = collect_statistics(
            backend, model, clip, triplets, preserve, site_preset,
            stats_device=stats_device)

        solve_dtype = (torch.float64 if opts["solve_precision"] == "float64"
                       else torch.float32)

        # Build every candidate objective. They share the collection pass and the
        # covariance, so a second one costs one extra linear solve.
        wanted = OBJECTIVES if opts["objective"] == "auto" else (opts["objective"],)
        built = {}
        for name in wanted:
            built[name] = build_lora(
                stats, sites, float(strength), int(rank), float(opts["regularization"]),
                preservation_weight=float(opts["preservation_weight"]), objective=name,
                solve_dtype=solve_dtype, save_dtype=_DTYPES[opts["save_dtype"]])

        # Cheap, unconditional, and fatal: a LoRA whose keys ComfyUI cannot resolve is a
        # silent dud at every strength, and that must never reach the loras folder.
        binding = assert_binds(model, built[wanted[0]][0])
        logging.info("%s LoRA key binding: %d/%d modules resolve.", LOG, *binding)

        # ---- measure the candidates against the real denoiser, and pick -------------
        check = None
        applied_gain = 1.0
        chosen = wanted[0]
        if verify:
            logging.info("%s Verifying %d candidate objective(s) against real denoiser "
                         "passes...", LOG, len(built))
            check = calibrate(
                backend, model, {k: v[0] for k, v in built.items()}, conds, triplets,
                int(opts["probe_resolution"]), int(opts["probe_resolution"]),
                probe_scenes=int(opts["probe_scenes"]),
                probe_steps=int(opts["probe_steps"]),
                schedule_steps=int(opts["schedule_steps"]),
                seed=int(opts["pack_seed"]))
            if check.get("ok"):
                # Rank by how much of the prompt difference each one actually delivers.
                # A candidate can win on cosine while being too small to see, so score on
                # the product of alignment and (capped) achievable size.
                chosen = max(check["candidates"], key=lambda k: _score(
                    check["candidates"][k]))
                metrics = check["candidates"][chosen]
                # The gain is signed on purpose. If the measurement says the edit moves
                # the denoiser *opposite* to the prompt pair, the honest correction is to
                # negate it -- a slider whose plus end is the minus end is a bug the user
                # should never have to discover by hand. Magnitude is clamped; sign is
                # not. |image_effect| is the trust condition, because an anti-correlated
                # edit is just as informative as a correlated one.
                # The edit already carries an analytic scale from the solve, so this is
                # a refinement, not the only thing standing between the user and an
                # invisible slider. The gate can therefore be loose: it only needs to
                # exclude measurements that are pure noise.
                if abs(metrics["image_effect"]) > 0.08:
                    gain = metrics["gain"]
                    applied_gain = float(min(max(abs(gain), 0.5), 50.0))
                    if gain < 0:
                        applied_gain = -applied_gain
                        logging.warning(
                            "%s The measured effect ran opposite to the prompt pair "
                            "(image_effect %.3f); the edit has been negated so +strength "
                            "is the positive concept.", LOG, metrics["image_effect"])
                    rescale_lora(built[chosen][0], applied_gain)

        lora, site_reports = built[chosen]
        primary = sites[0][0]
        fit = sum(r["token_fit"] for r in site_reports.values()) / len(site_reports)
        effect_score = (check["candidates"][chosen]["image_effect"]
                        if check and check.get("ok") else float(fit))

        elapsed = time.time() - started
        metadata = {
            "zt_slider_version": "2.0.0",
            "backend": backend.key,
            "concept": spec["name"],
            "concept_preset": spec["preset"],
            "positive_phrase": spec["positive"],
            "negative_phrase": spec["negative"],
            "strength": strength,
            "calibrated_gain": applied_gain,
            "rank": site_reports[primary]["rank"],
            "scenes": len(triplets),
            "scene_pack": scene_pack,
            "preservation_pack": preserve_pack,
            "preservation_prompts": len(preserve),
            "preservation_weight": opts["preservation_weight"],
            "regularization": opts["regularization"],
            "objective": chosen,
            "edit_site": site_preset,
            "exact_site": exact,
            "token_fit": round(fit, 4),
            "image_effect": (round(check["candidates"][chosen]["image_effect"], 4)
                             if check and check.get("ok") else "not measured"),
            "method": "closed-form text-conditioning edit (no gradient training)",
        }

        saved_path = ""
        if auto_save:
            saved_path = save_lora(lora, spec["name"], site_reports[primary]["rank"],
                                   save_name, metadata)

        report = self._format_report(
            backend, model, spec, scene_pack, preserve_pack, len(preserve), site_preset,
            exact, sites, site_reports, strength, int(rank), opts, len(triplets),
            check, chosen, applied_gain, binding, saved_path, elapsed, stats)
        logging.info("%s\n%s", LOG, report)

        curve = list(site_reports[primary]["rank_error_curve"])
        if len(curve) < 2:                     # Plot Loss Graph divides by (max - min)
            curve = (curve * 2) if curve else [0.0, 0.0]

        # The plan carries the resolved concept and the exact scenes it was measured on,
        # so ZT Slider Trainer refines this axis rather than a different one built from
        # the same widget values.
        plan = {
            "concept": spec,
            "triplets": triplets,
            "scene_pack": scene_pack,
            "preservation_pack": preserve_pack,
            "objective": chosen,
            "backend": backend.key,
        }
        return {
            "ui": {"text": [report]},
            "result": (lora, plan, {"loss": curve}, saved_path, report,
                       float(effect_score)),
        }

    # ---------------------------------------------------------------------------------

    @staticmethod
    def _resolve_concept(concept, positive, negative, name):
        if concept != "Custom":
            spec = dict(packs.concept_spec(concept))
            spec["preset"] = concept
            spec["name"] = sanitize_name(
                concept.split(" (")[0].strip().lower().replace(" ", "_"), "slider")
            return spec

        positive, negative = positive.strip(), negative.strip()
        if not positive or not negative:
            raise ValueError(
                "{} concept = Custom needs both custom_positive and custom_negative. The "
                "slider direction is measured as the difference between them, so both "
                "ends have to be described.".format(LOG))
        return {
            "preset": "Custom", "positive": positive, "negative": negative,
            "scenes": packs.DEFAULT_SUBJECT_PACK,
            # Nothing is known about a custom concept, so nothing is held out of the
            # generated scenes. If the slider controls framing, style or lighting, set
            # scene_pack + hold on the Advanced node or supply custom_scenes.
            "hold_axes": (),
            "preserve": packs.DEFAULT_PRESERVATION_PACK,
            "name": sanitize_name((name.strip() or "custom").lower().replace(" ", "_"),
                                  "custom"),
        }

    @staticmethod
    def _build_scenes(spec, count, custom_scenes, opts):
        pack = spec["scenes"] if opts["scene_pack"] == "auto" else opts["scene_pack"]
        extra = packs.split_lines(custom_scenes)
        bases = packs.expand_base_prompts(
            pack, count, int(opts["pack_seed"]), hold_axes=spec.get("hold_axes", ()))
        bases = list(dict.fromkeys(bases + extra))
        return packs.build_triplets(bases, spec["positive"], spec["negative"]), pack

    @staticmethod
    def _build_preservation(spec, opts):
        pack = (spec["preserve"] if opts["preservation_pack"] == "auto"
                else opts["preservation_pack"])
        return packs.expand_preservation(
            pack, int(opts["preservation_count"]), int(opts["pack_seed"])), pack

    # ---------------------------------------------------------------------------------

    @staticmethod
    def _format_report(backend, model, spec, scene_pack, preserve_pack, n_preserve,
                       site_preset, exact, sites, site_reports, strength, rank, opts,
                       n_scenes, check, chosen, applied_gain, binding, saved_path,
                       elapsed, stats):
        lines = []
        add = lines.append
        bar = "=" * 74

        add(bar)
        add("ZERO-TRAINING SLIDER  --  {}".format(spec["name"]))
        add(bar)
        add("model       : {}".format(backend.describe(model)))
        add("concept     : {}".format(spec["preset"]))
        add("  positive  : {}".format(spec["positive"]))
        add("  negative  : {}".format(spec["negative"]))
        held = ", ".join(spec.get("hold_axes", ())) or "none"
        prefix = stats.prefix_lengths or [0]
        add("scenes      : {} from '{}' (axes held out: {}), {}-{} aligned tokens".format(
            n_scenes, scene_pack, held, min(prefix), max(prefix)))
        add("preservation: {} prompts from '{}', weight {}".format(
            n_preserve, preserve_pack, opts["preservation_weight"]))
        add("edit site   : {} ({}) -> {}".format(
            site_preset, "exact" if exact else "APPROXIMATE",
            ", ".join(n for n, _ in sites)))
        add("key binding : {}/{} modules resolve in ComfyUI's LoRA key map".format(
            *binding))
        add("strength {} | rank {} | ridge {} | {:.0f}s total".format(
            strength, site_reports[sites[0][0]]["rank"], opts["regularization"], elapsed))
        add("")

        add("-- does it move the image? -------------------------------------------")
        if check is None:
            add("  verify was off, so this was never measured. The numbers below are")
            add("  from text-conditioning space only and cannot tell you whether the")
            add("  slider is visible. Turn verify on for the answer.")
        elif not check.get("ok"):
            add("  the check could not run: {}".format(check.get("error", "unknown")))
            add("  The LoRA is still valid -- test it by hand.")
        else:
            cands = check["candidates"]
            add("  Each candidate was applied through the stock Load LoRA path and run")
            add("  against the frozen model on the same latents.")
            add("")
            add("    objective   image_effect  size vs prompt  bound keys")
            for label in sorted(cands, key=lambda k: -_score(cands[k])):
                m = cands[label]
                add("    {:10s}  {:+8.3f}      {:8.4f}        {}/{}  {}".format(
                    label, m["image_effect"], m["relative_size"],
                    m["bound_keys"], m["total_keys"],
                    "<- kept" if label == chosen else ""))
            add("")
            add("  image_effect  cosine between what the LoRA does to the denoiser and")
            add("                what writing the concept into the prompt does. THE")
            add("                number. Above 0.5 works; near 0 is invisible.")
            add("  size vs prompt  how big its effect is before scaling. A good cosine")
            add("                with a tiny size means the direction is right and the")
            add("                channel is narrow -- that is what the gain fixes.")
            add("  bound keys    modules ComfyUI's key map resolves. 0 means the file")
            add("                loads and does nothing, whatever the strength.")
            add("")
            add("  kept '{}', scaled by {:.2f}x so strength 1.0 matches the prompt.".format(
                chosen, applied_gain))
            add("  probed {} scenes at sigma {}".format(
                check.get("probe_scenes", "?"),
                ", ".join("{:.2f}".format(x) for x in check.get("probe_sigmas", []))))
        add("")

        add("-- the solve ---------------------------------------------------------")
        for name, info in site_reports.items():
            add("{}  [{} -> {}]  objective '{}'".format(
                name, info["d_in"], info["d_out"], info.get("objective", "?")))
            add("   prompt_magnitude   {:.3f}   fraction of the raw measurement that was".format(
                info.get("contamination", 0.0)))
            add("                              conditioning STRENGTH, not direction, and was")
            add("                              projected out. A large number here is normal")
            add("                              and healthy -- left in, it makes every slider")
            add("                              behave as an inverted contrast knob.")
            add("   concept_agreement  {:+.3f}  do the scenes agree what the concept is?".format(
                info["concept_agreement"]))
            add("                              Low means the base prompts are fighting the")
            add("                              concept phrase -- a scene that already says")
            add("                              'wide shot' cannot measure a framing slider.")
            add("                              Worst scene: {:+.3f}".format(
                info["concept_agreement_min"]))
            add("   auto_scale         {:.2f}x   the solve returns the SMALLEST update that".format(
                info.get("auto_scale", 1.0)))
            add("                              fits, which after ridge and preservation can")
            add("                              be a faint fraction of the concept. The edit")
            add("                              was scaled by this so its conditioning shift")
            add("                              matches what writing the phrase does:")
            add("                              {:.4f} vs {:.4f}. Everything below describes".format(
                info.get("shift_rms", 0.0), info.get("prompt_shift_rms", 0.0)))
            add("                              the solve BEFORE this scaling.")
            add("   token_fit          {:+.3f}  how much of the measured concept the edit".format(
                info["token_fit"]))
            add("                              reproduces, per token. Above 0.4 is good.")
            add("   direction_match    {:+.3f}  the same thing per scene rather than per".format(
                info["direction_match"]))
            add("                              token, so always the kinder number.")
            add("   strength_ratio     {:.3f}   achieved / requested magnitude before".format(
                info["strength_ratio"]))
            add("                              calibration. Preservation and ridge cost this.")
            if info["preservation_leakage"]:
                add("   leakage            {:.4f}  movement of prompts meant to stay put,".format(
                    info["preservation_leakage"]["rms"]))
                add("                              relative to their size. Under 0.05 is clean.")
            add("   rank_energy_kept   {:.3f}   of the full edit, captured at rank {}.".format(
                info["rank_energy_kept"], info["rank"]))
            add("")

        add("-- notes -------------------------------------------------------------")
        for note in ZTSlider._verdict(site_reports, check, chosen, exact, sites,
                                      n_preserve):
            add("* " + note)
        if backend.notes:
            add("* " + backend.notes)
        add("")

        add("-- next --------------------------------------------------------------")
        if saved_path:
            add("saved: {}".format(saved_path))
        else:
            add("auto_save was off; the lora output is live for Save LoRA / Load LoRA Model.")
        add("Load with LoraLoaderModelOnly and sweep strength -2.0 .. +2.0.")
        add("Negative strength gives the other end of the axis.")
        add(bar)
        return "\n".join(lines)

    @staticmethod
    def _verdict(site_reports, check, chosen, exact, sites, n_preserve):
        out = []
        worst_agreement = min(r["concept_agreement"] for r in site_reports.values())
        worst_fit = min(r["token_fit"] for r in site_reports.values())
        weakest = min(r["strength_ratio"] for r in site_reports.values())
        leaks = [r["preservation_leakage"]["rms"] for r in site_reports.values()
                 if r["preservation_leakage"]]

        if worst_agreement < 0.35:
            out.append(
                "LOW concept_agreement ({:.2f}). The scenes disagree about what the "
                "concept is, which almost always means the base prompts mention the "
                "attribute the slider controls. Use a scene pack that does not describe "
                "it, or write custom_scenes that leave it out.".format(worst_agreement))
        if check and check.get("ok"):
            metrics = check["candidates"][chosen]
            if metrics.get("bound_keys") == 0:
                out.append(
                    "NO KEYS BOUND. ComfyUI's LoRA key map does not resolve any module in "
                    "this file, so it would load and do nothing. The backend's site names "
                    "do not match the model's module names.")
            elif metrics["image_effect"] < 0.15:
                out.append(
                    "LOW image_effect ({:.2f}). The slider barely moves the denoiser, and "
                    "raising strength will mostly add artefacts rather than effect. Either "
                    "the concept is not in the text prior -- check by generating your two "
                    "extreme prompts at one seed -- or concept_agreement above is the "
                    "cause.".format(metrics["image_effect"]))
            elif metrics["image_effect"] < 0.4:
                out.append(
                    "MODERATE image_effect ({:.2f}). It works but is entangled with other "
                    "changes. More scenes and a higher preservation_weight usually "
                    "sharpen it.".format(metrics["image_effect"]))
        if worst_fit < 0.15:
            out.append(
                "LOW token_fit ({:.2f}). The edit reproduces little of the measured "
                "concept. Try lowering regularization, or preservation_weight if leakage "
                "is already well under 0.05.".format(worst_fit))
        if weakest < 0.25 and n_preserve:
            out.append(
                "LOW strength_ratio ({:.2f}). Most of the requested shift was cancelled, "
                "usually by a preservation set that overlaps what the slider targets. "
                "Leave preservation_pack on 'auto' -- each concept ships with one chosen "
                "to complement it.".format(weakest))
        if leaks and max(leaks) > 0.15:
            out.append(
                "HIGH leakage ({:.2f}). This will disturb unrelated prompts. Raise "
                "preservation_weight or regularization.".format(max(leaks)))
        if not exact:
            out.append("This edit site is approximate; strength was split across {} "
                       "layers. 'auto' picks the exact site.".format(len(sites)))
        if not out:
            out.append("Numbers look healthy. Load it and sweep the strength.")
        return out


# --------------------------------------------------------------------------------------
# ZT Slider Advanced
# --------------------------------------------------------------------------------------

class ZTSliderAdvanced:
    """Solver and probe knobs. Every one has a working default on the build node.

    Only ``preservation_weight`` and ``regularization`` change results much; they trade
    the same thing against each other -- how hard the solve may push versus how much it
    may disturb.

    The pack overrides exist for custom concepts, where nothing is known about which
    scenes would collide with the axis. For a built-in concept, leave them on 'auto':
    each one ships with the scene family and preservation set chosen to complement it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preservation_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05,
                    "tooltip": "How hard the preservation prompts pull back. Raise if the "
                               "slider disturbs unrelated prompts; lower if it is too weak.",
                }),
                "regularization": ("FLOAT", {
                    "default": 0.02, "min": 0.0001, "max": 1.0, "step": 0.0001,
                    "round": False,
                    "tooltip": "Ridge term as a fraction of the mean eigenvalue of the "
                               "token covariance. Low = fits the measured scenes tightly "
                               "and generalises worse; high = gentler and more robust. "
                               "0.005-0.1 is the useful band.",
                }),
                "preservation_pack": (
                    ["auto"] + sorted(packs.PRESERVATION_PACKS) + ["off"],
                    {"default": "auto",
                     "tooltip": "'auto' uses the set the concept ships with. Whatever you "
                                "pick must NOT cover what the slider targets: preserving "
                                "framings while building a framing slider tells the solve "
                                "to move and not move the same thing."},
                ),
                "preservation_count": ("INT", {"default": 24, "min": 0, "max": 256}),
                "scene_pack": (
                    ["auto"] + sorted(packs.SUBJECT_PACKS),
                    {"default": "auto",
                     "tooltip": "'auto' uses the family the concept ships with, and holds "
                                "out the axes that would collide with it. Overriding "
                                "loses that protection.",
                     },
                ),
                "pack_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffff,
                    "tooltip": "Shuffle seed for scene and preservation selection. Same "
                               "seed rebuilds the identical slider.",
                }),
            },
            "optional": {
                "objective": (
                    ["auto"] + list(OBJECTIVES),
                    {"default": "auto",
                     "tooltip": "How the concept is targeted. 'aligned' compares the three "
                                "prompts over the tokens they share -- exactly reproducible "
                                "but only as strong as the tower's cross-token mixing "
                                "allows (on Krea 2 that is two refiner blocks, which is a "
                                "narrow channel for global attributes). 'pooled' targets "
                                "the whole-sequence mean difference: much larger, but only "
                                "matchable on average. 'auto' builds both and keeps "
                                "whichever actually moves the denoiser -- almost always "
                                "what you want, and it costs one extra forward per probe."},
                ),
                "edit_site": (
                    ["auto"] + zt_backends.all_site_presets(),
                    {"default": "auto",
                     "tooltip": "Which layers of the text tower to edit. 'auto' picks the "
                                "backend's exact site -- the last linear before text and "
                                "image tokens meet."},
                ),
                "probe_scenes": ("INT", {
                    "default": 3, "min": 1, "max": 16,
                    "tooltip": "Scenes used by the verification pass. Each costs one short "
                               "rollout plus a few extra denoiser passes.",
                }),
                "probe_steps": ("INT", {
                    "default": 3, "min": 1, "max": 16,
                    "tooltip": "Timesteps sampled per probe scene.",
                }),
                "probe_resolution": ("INT", {
                    "default": 512, "min": 256, "max": 1536, "step": 64,
                    "tooltip": "Resolution of the verification passes. 512 is plenty -- it "
                               "measures a direction, not image quality.",
                }),
                "schedule_steps": ("INT", {
                    "default": 8, "min": 4, "max": 50,
                    "tooltip": "Length of the sampler schedule the probes roll out along. "
                               "Match your usual generation steps (8 for a turbo model).",
                }),
                "solve_precision": (["float64", "float32"], {
                    "default": "float64",
                    "tooltip": "float64 for the linear solve. The covariance is "
                               "near-singular by construction, so this is not paranoia.",
                }),
                "stats_device": (["auto", "cpu"], {
                    "default": "auto",
                    "tooltip": "Where token covariances are accumulated. 'auto' uses the "
                               "GPU; switch to cpu if VRAM is tight.",
                }),
                "save_dtype": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            },
        }

    RETURN_TYPES = ("ZTS_OPTS",)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def build(self, **kwargs):
        opts = dict(DEFAULT_OPTS)
        opts.update(kwargs)
        return (opts,)


# --------------------------------------------------------------------------------------
# ZT Slider Info
# --------------------------------------------------------------------------------------

class ZTSliderInfo:
    """Prints how this works, which backends are installed, and -- with a MODEL
    connected -- which layers would be edited and whether that edit is exact.
    Changes nothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"model": ("MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    def info(self, model=None):
        lines = [
            "=" * 74,
            "ZERO-TRAINING SLIDER LoRA",
            "=" * 74,
            "",
            "Graph:  Load Diffusion Model + CLIPLoader  ->  ZT Slider  ->  loras/",
            "        (ZT Slider Advanced is optional)",
            "",
            "Solves a concept slider in closed form instead of training it. The concept",
            "is measured off the model's own text tower by encoding each scene three",
            "ways -- neutral, positive, negative -- and comparing them over the tokens",
            "all three share. Then one ridge least-squares solve for the weight update",
            "that adds it while a preservation set stays put.",
            "",
            "With verify on it also runs real denoiser passes to measure what the LoRA",
            "does to an image, and scales it so strength 1.0 equals writing the concept",
            "into the prompt. Read image_effect in the report: that is the go/no-go.",
            "",
            "CAN DO   : age, weight, expression, detail, lighting, colour, framing,",
            "           weather, time of day, wear -- anything a prompt can already reach.",
            "CANNOT   : identity, an artist's linework, pose, hands, object count.",
            "           It re-expresses what the model knows; it teaches nothing new.",
            "",
            "-- installed backends ------------------------------------------------",
        ]
        for key, backend in zt_backends.all_backends().items():
            lines.append("  {} ({})".format(backend.display_name, key))
            lines.append("     wire in : {}".format(backend.model_hint))
            lines.append("     sites   : {}".format(", ".join(
                "{}{}".format(p, " [exact]" if p in backend.exact_site_presets else "")
                for p in backend.site_presets)))
        errors = zt_backends.import_errors()
        if errors:
            lines.append("  backends that failed to import:")
            for name, err in errors.items():
                lines.append("     {}: {}".format(name, err))
        lines += ["", "Adding a model is one file in zt_backends/ -- see ADDING_A_MODEL.md."]

        if model is not None:
            lines += ["", "-- connected model ---------------------------------------------------"]
            try:
                backend = zt_backends.detect(model)
                preset = backend.default_site_preset
                sites = backend.select_sites(model, preset)
                lines.append("  handled by : {} ({})".format(backend.display_name, backend.key))
                lines.append("  detected   : {}".format(backend.describe(model)))
                lines.append("  would edit : {} ({})".format(
                    ", ".join(n for n, _ in sites),
                    "exact" if preset in backend.exact_site_presets else "approximate"))
                for name, module in sites:
                    lines.append("     {}  {} -> {}".format(
                        name, module.weight.shape[1], module.weight.shape[0]))
                lines.append("  probe latent: {}".format(
                    backend.latent_shape(model, 512, 512)))
            except Exception as exc:  # noqa: BLE001 - this node exists to report problems
                lines.append("  {}".format(exc))

        lines.append("=" * 74)
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "ZTSlider": ZTSlider,
    "ZTSliderAdvanced": ZTSliderAdvanced,
    "ZTSliderInfo": ZTSliderInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZTSlider": "ZT Slider (Zero-Training LoRA)",
    "ZTSliderAdvanced": "ZT Slider Advanced",
    "ZTSliderInfo": "ZT Slider Info / Backends",
}
