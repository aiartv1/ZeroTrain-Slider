"""ComfyUI-ZeroTrain-Slider — concept-slider LoRAs, solved then optionally refined.

Two stages, either usable alone:

  ZT Slider          solves the slider in closed form from the model's own text prior,
                     verifies it against the real denoiser, and saves a LoRA. Seconds.
  ZT Slider Trainer  takes that LoRA as a warm start and refines it with real gradient
                     training through the whole denoiser, lifting the ceiling the
                     text-only solve runs into. Minutes.

Model support is one file in ``zt_backends/`` — discovered automatically, and the same
file serves both stages. Krea 2 ships here; see ADDING_A_MODEL.md for the rest.
"""

from .nodes import (
    NODE_CLASS_MAPPINGS as _SOLVE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _SOLVE_NAMES,
)
from .train_nodes import (
    NODE_CLASS_MAPPINGS as _TRAIN_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _TRAIN_NAMES,
)

NODE_CLASS_MAPPINGS = {**_SOLVE_CLASSES, **_TRAIN_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_SOLVE_NAMES, **_TRAIN_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
