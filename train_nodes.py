"""The warm-start trainer nodes.

    ZT Slider ──lora──► ZT Slider Trainer ──► one refined LoRA
             └──plan───┘

``lora`` carries the closed-form weights, ``plan`` carries the concept and the scenes it
was measured on, so the trainer optimises the same axis rather than a different one that
happens to share a name. Both come straight off the ZT Slider node; no widgets have to be
re-entered.

Kept in its own file because it is a different kind of work: everything else in this
package is a closed-form solve that runs in seconds, and this is a real training loop with
an optimizer, gradient checkpointing and a model that streams from RAM.
"""

import json
import logging
import time

import torch

import comfy.samplers
from comfy.weight_adapter import adapter_maps

from . import zt_backends
from .zt_core import LOG, sanitize_name, save_lora
from .zt_train import train_slider

CATEGORY = "training/zero-train slider"

DEFAULT_TRAIN_OPTS = {
    "target_modules": "auto",
    "blocks": "all",
    # Taken from ComfyUI itself rather than hardcoded: the keys are "LoRA"/"LoHa"/
    # "LoKr"/"OFT", capitalised, and a hardcoded "lora" is a KeyError at adapter-build
    # time -- after the prompts have already been encoded.
    "algorithm": next(iter(adapter_maps)),
    "alpha": 1.0,
    "optimizer": "AdamW",
    "lr_scheduler": "cosine",
    "warmup_steps": 5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "batch_size": 1,
    "grad_accumulation_steps": 1,
    "train_scenes": 3,
    "schedule_steps": 8,
    "scheduler": "simple",
    "first_step": 0,
    "last_step": -1,
    "direction_mode": "both_per_step",
    "loss_space": "velocity",
    "lora_dtype": "float32",
    "save_dtype": "float32",
}


class ZTSliderTrainer:
    """Refines a zero-training slider with real gradient training. One LoRA out.

    The closed-form solve only edits the text tower, so it can only express what the text
    encoder already encodes -- that ceiling is why some concepts come out sharp and
    subtler ones stay weak. Training has gradients through the whole denoiser, so it
    reaches the attention and MLP layers the text path cannot.

    Connect ``lora`` and ``plan`` from a ZT Slider node. The text-tower adapter then
    starts from the closed-form weights -- already pointing the right way and already
    scaled -- while adapters on the denoiser blocks start at zero and learn only what was
    missing. That is a warm start, not a merge: there is one set of weights throughout, so
    nothing can cancel out the way two independently built LoRAs can.

    From a warm start 100-300 steps is usually enough, where training from scratch needs
    many times that. Leave ``warm_start`` unconnected to train a slider from zero.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The same model the plan was built on."}),
                "clip": ("CLIP", {
                    "tooltip": "Its text encoder. Encoded once, then evicted before "
                               "training starts -- that eviction is most of the headroom "
                               "on a 12GB card.",
                }),
                "plan": ("ZTS_PLAN", {
                    "tooltip": "From ZT Slider. Carries the concept and the exact scenes "
                               "it was measured on, so training refines the same axis.",
                }),
                "steps": ("INT", {
                    "default": 150, "min": 10, "max": 20000,
                    "tooltip": "From a warm start 100-300 is usually enough. From scratch "
                               "(no warm_start) expect 500+.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.00005, "min": 0.0000001, "max": 0.01,
                    "step": 0.0000001, "round": False,
                    "tooltip": "Lower than a from-scratch run on purpose: the warm start "
                               "is already close, and a high rate will simply throw that "
                               "away in the first few steps. Raise it if the loss is flat.",
                }),
                "rank": ("INT", {
                    "default": 16, "min": 1, "max": 128,
                    "tooltip": "Rank for the newly created adapters. Warm-started modules "
                               "keep the rank they were solved at.",
                }),
                "eta": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "How far the training target is pushed along the slider "
                               "axis. Guidance-distilled turbo models compress the "
                               "prompt-difference signal, so they want more than a base "
                               "model: 2-3 is a good range there.",
                }),
                "width": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "Training resolution. 512 is the sweet spot -- sliders are "
                               "semantic and transfer to 1024, while cost scales with the "
                               "square of this.",
                }),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 16}),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Leave on. A model too large to stay resident streams from "
                               "RAM, and without checkpointing every streamed block stays "
                               "live for the whole backward pass and the run OOMs.",
                }),
                # Deliberately not "seed": the frontend injects a hidden
                # control_after_generate widget beside any INT named seed/noise_seed.
                "training_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "auto_save": ("BOOLEAN", {"default": True}),
                "save_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "warm_start": ("LORA_MODEL", {
                    "tooltip": "The ZT Slider node's lora output. Leave unconnected to "
                               "train from scratch.",
                }),
                "advanced": ("ZTS_TRAIN_OPTS", {
                    "tooltip": "Optional ZT Slider Train Advanced node.",
                }),
            },
        }

    RETURN_TYPES = ("LORA_MODEL", "LOSS_MAP", "INT", "STRING", "STRING")
    RETURN_NAMES = ("lora", "loss_map", "total_steps", "saved_path", "report")
    FUNCTION = "train"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    def train(self, model, clip, plan, steps, learning_rate, rank, eta, width, height,
              gradient_checkpointing, training_seed, auto_save, save_name,
              warm_start=None, advanced=None):
        started = time.time()
        opts = dict(DEFAULT_TRAIN_OPTS)
        if advanced:
            opts.update(advanced)
        opts.update({
            "steps": steps, "learning_rate": learning_rate, "rank": rank, "eta": eta,
            "width": width, "height": height, "training_seed": training_seed,
            "gradient_checkpointing": gradient_checkpointing,
        })

        backend = zt_backends.detect(model)
        if opts["target_modules"] == "auto":
            opts["target_modules"] = backend.default_train_preset
        if not opts["target_modules"]:
            raise ValueError(
                "{} The {} backend declares no training presets, so it can solve sliders "
                "but not train them. Add train_site_presets to its backend "
                "file.".format(LOG, backend.display_name))

        triplets = plan.get("triplets") or []
        if not triplets:
            raise ValueError("{} The plan carries no scenes.".format(LOG))
        concept = plan.get("concept", {})
        name = concept.get("name", "slider")

        logging.info("%s Training '%s' on %s: %d steps, %s warm start.",
                     LOG, name, backend.display_name, steps,
                     "with" if warm_start else "no")

        lora, loss_map, report = train_slider(
            backend, model, clip, triplets, opts, warm_start=warm_start)

        report.update({
            "backend": backend.key,
            "concept": name,
            "concept_preset": concept.get("preset", ""),
            "positive_phrase": concept.get("positive", ""),
            "negative_phrase": concept.get("negative", ""),
            "eta": eta,
            "learning_rate": learning_rate,
            "rank": rank,
            "resolution": "{}x{}".format(width, height),
            "target_modules": opts["target_modules"],
            "blocks": opts["blocks"],
            "optimizer": opts["optimizer"],
            "loss_space": opts["loss_space"],
            "direction_mode": opts["direction_mode"],
            "method": ("closed-form warm start + gradient slider training"
                       if warm_start else "gradient slider training"),
        })

        saved_path = ""
        if auto_save:
            saved_path = save_lora(
                lora, name, rank, save_name or "{}_zt_trained_r{}".format(
                    sanitize_name(name), rank), report)

        text = self._format_report(backend, concept, report, loss_map, saved_path,
                                   bool(warm_start), time.time() - started)
        logging.info("%s\n%s", LOG, text)
        return {"ui": {"text": [text]},
                "result": (lora, loss_map, int(steps), saved_path, text)}

    @staticmethod
    def _format_report(backend, concept, report, loss_map, saved_path, warm, elapsed):
        bar = "=" * 74
        losses = loss_map.get("loss") or []
        lines = [
            bar,
            "ZT SLIDER TRAINER  --  {}".format(report.get("concept", "?")),
            bar,
            "model      : {}".format(backend.display_name),
            "concept    : {}".format(concept.get("preset", "custom")),
            "start      : {}".format(
                "warm -- {} module(s) initialised from the closed-form solve".format(
                    report["warm_started"]) if warm else "from scratch (zeros)"),
            "adapters   : {} modules, {:.2f}M trainable params".format(
                report["adapters"], report["trainable_params_m"]),
            "targets    : {} preset '{}', blocks {}".format(
                report["adapters"], report["target_modules"], report["blocks"]),
            "bank       : {} samples from {} scene(s) x {} timestep(s)".format(
                report["bank_samples"], report["scenes"], report["timesteps"]),
            "{} steps in {:.1f} min".format(report["steps"], elapsed / 60.0),
            "",
            "-- did it learn? -----------------------------------------------------",
        ]
        if losses:
            first, final = report["first_loss"], report["final_loss"]
            drop = (1.0 - final / first) if first else 0.0
            lines += [
                "  loss {:.6f} -> {:.6f}   ({:+.1%})".format(first, final, -drop),
                "  mean of last 20 steps: {:.6f}".format(report["mean_loss_last_20"]),
                "",
            ]
            if drop < 0.05:
                lines.append("* Loss barely moved. From a warm start that can mean the")
                lines.append("  closed-form solution was already near the optimum -- check")
                lines.append("  the images before assuming it failed. If they are unchanged")
                lines.append("  too, raise learning_rate or eta.")
            elif drop > 0.9:
                lines.append("* Loss collapsed. Usually eta is low enough that the target is")
                lines.append("  nearly the neutral prediction, so the slider learns very")
                lines.append("  little. Raise eta.")
            else:
                lines.append("* Healthy convergence.")
        else:
            lines.append("  no steps ran")

        lines += [
            "",
            "-- next --------------------------------------------------------------",
            "saved: {}".format(saved_path) if saved_path
            else "auto_save off; the lora output is live for Save LoRA.",
            "Load with LoraLoaderModelOnly and sweep strength -2.0 .. +2.0.",
            "Compare against the zero-training LoRA at the same strength: if this one",
            "is not better, the concept was already at the text prior's ceiling.",
            bar,
        ]
        return "\n".join(lines)


class ZTSliderTrainAdvanced:
    """Training knobs. Every one has a working default on the trainer node.

    ``eta`` and ``learning_rate`` on the main node matter far more than anything here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        presets = sorted({p for b in zt_backends.all_backends().values()
                          for p in b.train_site_presets})
        return {
            "required": {
                "target_modules": (["auto"] + presets, {
                    "default": "auto",
                    "tooltip": "Which denoiser layers get trainable adapters. 'auto' uses "
                               "the backend's default. attn_only is cheaper and often "
                               "enough for a slider.",
                }),
                "blocks": ("STRING", {
                    "default": "all",
                    "tooltip": "Which transformer blocks to adapt: 'all', '8-20', "
                               "'0,4,10-14'. Restricting to middle blocks trains faster "
                               "and often entangles less.",
                }),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 128.0, "step": 0.01}),
                "optimizer": (["AdamW", "AdamW8bit", "Adam", "SGD", "RMSprop"],
                              {"default": "AdamW"}),
                "lr_scheduler": (["cosine", "linear", "constant"], {"default": "cosine"}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 1000}),
                "weight_decay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                           "step": 0.001, "round": False}),
                "max_grad_norm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                            "step": 0.1}),
                "train_scenes": ("INT", {
                    "default": 3, "min": 1, "max": 32,
                    "tooltip": "How many of the plan's scenes to build targets from. More "
                               "scenes generalise better but each costs a full rollout.",
                }),
                "grad_accumulation_steps": ("INT", {"default": 1, "min": 1, "max": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "direction_mode": (["both_per_step", "alternate", "positive_only"], {
                    "default": "both_per_step",
                    "tooltip": "both_per_step trains +1 and -1 every step -- the most "
                               "stable, and what keeps the axis symmetric.",
                }),
                "loss_space": (["velocity", "x0"], {
                    "default": "velocity",
                    "tooltip": "velocity weights every timestep evenly and is the Concept "
                               "Sliders convention.",
                }),
                "schedule_steps": ("INT", {"default": 8, "min": 4, "max": 50,
                                           "tooltip": "Match your usual generation steps."}),
                # Read from ComfyUI rather than listed by hand, for the same reason as
                # the algorithm widget above.
                "scheduler": (list(comfy.samplers.SCHEDULER_NAMES), {"default": "simple"}),
                "first_step": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "Timestep window. High sigma (early) decides structure and "
                               "composition; low sigma (late) decides surface and texture. "
                               "Narrowing this is the main anti-entanglement lever.",
                }),
                "last_step": ("INT", {"default": -1, "min": -1, "max": 100}),
                "algorithm": (list(adapter_maps.keys()), {
                    "default": next(iter(adapter_maps)),
                    "tooltip": "Adapter family. LoRA is the right choice for a slider; "
                               "the others exist because ComfyUI supports them.",
                }),
                "lora_dtype": (["float32", "bfloat16"], {
                    "default": "float32",
                    "tooltip": "float32 recommended: at these learning rates bfloat16 "
                               "rounds small updates away.",
                }),
                "save_dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
            },
        }

    RETURN_TYPES = ("ZTS_TRAIN_OPTS",)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def build(self, **kwargs):
        opts = dict(DEFAULT_TRAIN_OPTS)
        opts.update(kwargs)
        return (opts,)


NODE_CLASS_MAPPINGS = {
    "ZTSliderTrainer": ZTSliderTrainer,
    "ZTSliderTrainAdvanced": ZTSliderTrainAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZTSliderTrainer": "ZT Slider Trainer (warm start)",
    "ZTSliderTrainAdvanced": "ZT Slider Train Advanced",
}
