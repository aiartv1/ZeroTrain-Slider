# Publishing this project on GitHub

[Back to README](../README.md)

This is a maintainer guide. The documentation package includes the supplied Python files and `pyproject.toml`, plus the example workflow with updated explanatory notes. It has not created a repository, published a release or run the nodes on a GPU.

## 1. Complete the release information

- Replace `YOUR_GITHUB_USERNAME` in the user guide with the real repository owner, and update the repository name if different.
- Add a full license file consistent with the existing `GPL-3.0-or-later` declaration and any applicable attribution. The supplied folder did not contain a LICENSE file. Do not silently choose a different license while preparing documentation.
- Add the real Repository URL under `[project.urls]` in `pyproject.toml`. Consider also linking Documentation and Issues once those URLs exist.
- Record a tested ComfyUI version/commit, PyTorch version, model/encoder sources and filenames, OS, GPU/VRAM and RAM. A successful local run is useful evidence; document its exact setup.
- Add genuine comparison images and their workflow/settings. Show a fixed-seed baseline with both poles, preferably on multiple prompts. Clearly distinguish solver-only and refined results.

The public README intentionally does not claim universal runtime, memory usage or success rates. Replace the compatibility gap with measured results when available, not guessed requirements.

## 2. Address source/documentation inconsistencies

These were found during source inspection; no Python behaviour was changed in this documentation package:

| Item | Current state | Suggested release action |
|---|---|---|
| Version metadata | Package `3.0.1`; solver LoRA metadata `2.0.0` | Decide whether these are intentionally different schemas; align or explain them |
| License | GPL declaration but no full license file | Supply full text and appropriate attribution |
| Registry identity | Blank `PublisherId` | Fill only if publishing to the Registry |
| Workflow notes | Updated with author guidance on optional experimental training and loading weights | Upload the revised workflow alongside the docs |
| Custom scenes | Append-only, generated scenes first | Consider a custom-only mode and exposed axis exclusions in a future code change |
| Metric sign | Metrics remain before signed gain | Consider a clearly labelled pre/post-calibration report and a second probe |
| Key-binding check | Partial/unknown results are not fatal | Consider stricter validation or prominent diagnostic status |
| Training validation | No final denoiser verification or preservation objective | Retain limitations and require image comparison |
| Startup compatibility | Trainer APIs imported even for solver-only use | Test clean imports and document minimum host version |

These are release review findings, not claims that every item caused a runtime failure.

## 3. Prepare a clean repository folder

Use a separate publishing checkout so personal installation files and model weights do not become repository content. Keep the Python modules, complete `zt_backends/`, docs, workflow, metadata, license and `.github` templates. The included `.gitignore` excludes caches, common weight files, environments and logs.

The node's `__init__.py` must be at the repository root, not under an extra package wrapper. Include the docs as ordinary Markdown files; GitHub renders them and the README without a separate website.

Keep legitimate third-party notices. Review staged changes before committing. Do not upload generated model files by accident; link separately hosted examples with the required model information if you decide to share them.

## 4. Create and push the repository

Create an empty GitHub repository named `ComfyUI-ZeroTrain-Slider`. For this existing-files route, do not initialise remote README/license/gitignore files. Authenticate Git on your machine, open a terminal in the prepared project root, and run the following only if it is not already a Git repository:

```shell
git init -b main
git add .
git diff --cached --stat
git status
```

Review what is staged, then commit and connect the real URL:

```shell
git commit -m "Prepare ZeroTrain Slider source and documentation"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ComfyUI-ZeroTrain-Slider.git
git remote -v
git push -u origin main
```

If a repository or remote already exists, inspect it rather than reinitialising or overwriting it. This sequence follows GitHub's [guide to adding locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github). GitHub Desktop is also an option if you prefer a graphical workflow.

## 5. Check the actual user experience

Before announcing it publicly:

1. Clone or download the published source into a clean compatible ComfyUI setup.
2. Confirm all five nodes import and ZT Slider Info detects Krea 2.
3. Open the example, re-select real model files and build a solver-only slider with verification.
4. Reload the saved file with the stock model-only LoRA loader; compare -1/0/+1 on held-out prompts and fixed seeds.
5. Run optional refinement. Check a nonzero warm-start count, saved-file loading and final visual comparisons.
6. Check a skipped/failed verification is reported honestly, and confirm normal generation works after cancellation/errors.
7. Read every Markdown link in GitHub, inspect the example's node names/settings, and make sure the public instructions do not depend on your local drive paths.

Static documentation review cannot substitute for these runtime checks. Record any untested environments explicitly.

## 6. Present the repository clearly

Suggested About description:

> Build concept-slider LoRAs in ComfyUI with a closed-form solver and optional warm-start gradient refinement. Krea 2 backend included.

Suggested topics: `comfyui`, `comfyui-custom-node`, `lora`, `concept-sliders`, `krea2`.

Enable Issues so the supplied bug-report template can be used. Include the tested environment and known limitations in release notes. Use a version tag matching your chosen package version; `v3.0.1` is appropriate only if that is the version you actually release and the tag is unused. Do not imply a prior public release history from the supplied version number alone.

## 7. Optional Comfy Registry publication

GitHub hosting and Comfy Registry publication are separate steps. Uploading source does not establish a Registry listing.

Create a Registry publisher, obtain its publishing API key, set `PublisherId` and repository metadata, and follow the current [official publishing instructions](https://docs.comfy.org/registry/publishing). They support Comfy CLI and GitHub Actions, and explain `.comfyignore` exclusions. Do not regenerate metadata over this project's existing file without reviewing it. Store publishing credentials through the documented secure mechanism, outside source control.

No Registry account, API key or publication workflow is configured in this package. Verify the resulting installation/listing before adding a Manager-installation claim to the README.

## 8. Future updates

For each release, state changed behaviour and relevant compatibility changes, update the version deliberately, and keep example workflows, docs and source defaults in sync. Recheck numerical changes with fixed-seed images rather than only solver metrics or training loss. Retain the exact commit and settings for published comparison images.
