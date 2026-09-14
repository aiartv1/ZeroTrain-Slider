# Contributing

Thank you for helping improve the node and its documentation.

For a bug, include the node commit/version, ComfyUI commit/version, installation type, OS, GPU/VRAM, RAM, model and encoder filenames, loader precision, minimal workflow and full traceback. For behaviour problems attach a solver/trainer report and fixed-seed baseline/negative/positive images.

For changes, explain the concrete problem, resulting behaviour and validation. Keep documentation and widget defaults consistent. Backend contributions should follow [ADDING_A_MODEL.md](ADDING_A_MODEL.md); a model-detection rule alone is not sufficient.

Validate changes in a compatible ComfyUI environment. For numerical changes, compare saved-file reloads, both strength directions and unrelated prompts. For training changes, check warm-start counts/ranks, finite loss and ordinary generation after cleanup. Do not replace visual validation with a lower training loss.

The project metadata declares GPL-3.0-or-later. Preserve applicable attribution and include only contributions you are entitled to share. The full license file is a release preparation item listed in [PUBLISHING.md](docs/PUBLISHING.md).
