"""The zero-training slider solve: statistics, closed form, factorisation, saving.

Model-agnostic. Everything model-specific is behind the backend interface in
``zt_backends/``.

The method
----------
Pick a Linear ``y = W x`` inside the text tower whose input is a pure function of the
text conditioning. Encode each scene three ways — neutral, concept-positive,
concept-negative — where the extremes are the neutral prompt with a phrase appended, so
all three share an identical prefix. Over those **shared prefix positions** the three
sequences line up token for token, and

    dy[p,t] = (y_positive[p,t] - y_negative[p,t]) / 2

is the concept, measured per token, in the layer's own output space. Solve for the
smallest weight update that adds it to the neutral prompt:

    for every shared prefix token t of scene p:   dW @ x[p,t] = eta * dy[p,t]

with a preservation set held at zero and a ridge term. Closed form:

    dW = eta * D @ M^-1
    D  = sum_p mean_t dy[p,t] x[p,t]^T
    M  = C_edit + lam * C_pres + beta * I

One symmetric solve, no optimiser, no sampler, no latents.

Two objectives, and why the choice is measured rather than argued
----------------------------------------------------------------
At inference the user's prompt contains no concept phrase, so the only thing a LoRA can
do is transform the tokens they did write. There are two ways to define what to
transform them into, and they trade off against each other:

**aligned** — the target above. Exactly reproducible (it is a per-token function of a
per-token input), but its size depends entirely on how much an appended phrase perturbs
the *shared* positions. On Krea 2 that is a narrow channel: the Qwen3-VL encoder is
causal, so the shared prefix is bit-identical across the three prompts, and
``txtfusion``'s ``layerwise_blocks`` attend over the 12 tap-layer axis *per token*, not
across tokens. Only the two ``refiner_blocks`` mix across the sequence. Everything the
aligned objective can see has to come through those two blocks — which is plenty for a
concept that recolours the whole sentence ("elderly") and far too little for a global
one ("vividly saturated").

**pooled** — target the whole-sequence mean difference, appended tokens included:
``dW x[p,t] = eta * dybar[p]`` for every token. Much larger, because it contains the
concept tokens' own contribution, but no linear map can send many different inputs to
one output, so it is only matched on average. It also carries a spurious term
proportional to the base prompt whenever the two extremes tokenise to different lengths.

Neither dominates: it depends on the concept and the architecture. So both are built —
they share one collection pass and one covariance — and ``zt_verify`` runs both against
the real denoiser and keeps whichever actually moves the image. That comparison appears
in the report.

*Halved.* Targeting half the difference makes LoRA strength +1 land on the positive
prompt and -1 on the negative, since the neutral sits between them.
"""

import logging
import os
import re
import time

import torch

import comfy.model_management
import comfy.utils
import folder_paths
import safetensors.torch

from .zt_backends import modules_on_device

LOG = "[ZeroTrainSlider]"
GB = 1024 ** 3

# Refuse a site preset whose matrices would not plausibly fit anywhere.
_COV_BUDGET = 8 * GB

_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_name(name, fallback="slider"):
    """Collapse anything that is not filename-safe. Never returns an empty string."""
    cleaned = _UNSAFE_NAME_RE.sub("_", (name or "").strip()).strip("._-")
    return cleaned or fallback


# --------------------------------------------------------------------------------------
# activation capture
# --------------------------------------------------------------------------------------

class _SiteCapture:
    """Forward hooks recording (input, output) at every edit site of one pass."""

    def __init__(self, sites):
        self.sites = sites
        self._handles = []
        self.current = {}

    def _make_hook(self, name):
        def hook(_module, args, output):
            if args:
                self.current[name] = (args[0].detach(), output.detach())
        return hook

    def __enter__(self):
        for name, module in self.sites:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.current = {}
        return False

    def take(self):
        """Pop this pass's captures, failing loudly if a site never fired."""
        out = self.current
        missing = [name for name, _ in self.sites if name not in out]
        if missing:
            raise RuntimeError(
                "{} These edit sites were never reached by the backend's text path: {}. "
                "The backend's run_text_path() and its site_presets disagree.".format(
                    LOG, ", ".join(missing))
            )
        self.current = {}
        return out


def _flat(tensor):
    """(1, seq, d) -> (seq, d) float32."""
    return tensor.reshape(-1, tensor.shape[-1]).float()


def _deflate(target, reference):
    """Remove the component of ``target`` parallel to ``reference``, row by row.

    This is the single most important line in the file, and leaving it out produces a
    slider that looks powerful and controls the wrong thing.

    A concept must change the *direction* of the conditioning, never its magnitude. Any
    component parallel to the prompt's own feature is not a concept at all -- it is a
    guidance-strength dial. Push it positive and the conditioning is amplified (vivid,
    contrasty); push it negative and it is weakened (dark, washed out). That reads as a
    dramatic slider on any concept whatsoever, which is exactly why it has to go.

    It creeps in because the two extreme phrases rarely tokenise to the same length. A
    whole-sequence mean divides the shared base content by a different token count on
    each side, so the difference retains a term proportional to the base prompt itself,
    scaled by the length mismatch. "brightly lit, strong luminous key light, high-key
    exposure, radiant illumination" is longer than "dimly lit, deep shadow, low-key
    underexposed, murky darkness", and that alone was enough to make a Lighting slider
    behave as an inverted brightness/contrast knob.

    Projecting the reference out removes it exactly, whatever caused it, and costs one
    dot product per row.
    """
    denom = (reference * reference).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    coefficient = (target * reference).sum(dim=-1, keepdim=True) / denom
    return target - coefficient * reference


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

class SliderStats:
    """Everything the solve needs, gathered in one pass over the prompt sets.

    Edit and preservation covariances are kept apart rather than pre-summed: it costs one
    extra matrix per group and makes preservation_weight a solve-time knob, so retuning
    it does not mean re-encoding every prompt.
    """

    def __init__(self):
        self.dims = {}            # site -> (d_in, d_out)
        self.group_of = {}        # site -> group id (sites sharing an input share a cov)
        self.cov_edit = {}        # group id -> (d_in, d_in) float32, neutral prompts
        self.cov_pres = {}        # group id -> (d_in, d_in) float32, preservation prompts
        # Two candidate objectives, gathered in the same pass. Which one wins is decided
        # by measurement in zt_verify, not by argument -- see OBJECTIVES below.
        self.cross = {}           # site -> (d_out, d_in), aligned: sum_p mean_t dy x^T
        self.target_energy = {}   # site -> float,         aligned: sum_p mean_t ||dy||^2
        self.cross_pooled = {}    # site -> (d_out, d_in), pooled:  sum_p dybar xbar^T
        self.pooled_energy = {}   # site -> float,         pooled:  sum_p ||dybar||^2
        self.signal_ratio = {}    # site -> float, pooled vs aligned magnitude
        self.x_edit = {}          # site -> list of (d_in,)  per-scene mean inputs
        self.dy_edit = {}         # site -> list of (d_out,) per-scene mean concept shift
        self.dy_pooled = {}       # site -> list of (d_out,) per-scene pooled shift
        self.x_pres = {}          # site -> list of (d_in,)  per-prompt mean inputs
        self.y_pres_energy = {}   # site -> float, sum_q mean_t ||y||^2
        self.y_edit_energy = {}   # site -> float, sum_p mean_t ||y_neutral||^2
        self.raw_energy = {}      # site -> float, concept energy before deflation
        self.kept_energy = {}     # site -> float, concept energy after deflation
        self.contamination = {}   # site -> float, fraction that was prompt magnitude
        self.n_pairs = 0
        self.n_preserve = 0
        self.prefix_lengths = []
        self.dropped_scenes = 0


def _assign_groups(stats, captured):
    """Sites fed the identical input tensor share one covariance.

    On a cross-attention model every ``to_k``/``to_v`` sees the same text embedding, so
    this collapses ~140 covariance matrices into one. Decided once, on the first prompt,
    by direct comparison — the sharing is structural and cannot change between prompts.
    """
    reps = []
    for name, (x, y) in captured.items():
        stats.dims[name] = (int(x.shape[-1]), int(y.shape[-1]))
        for gid, ref in reps:
            if ref.shape == x.shape and torch.equal(ref, x):
                stats.group_of[name] = gid
                break
        else:
            gid = len(reps)
            reps.append((gid, x))
            stats.group_of[name] = gid

    widths = {stats.group_of[n]: stats.dims[n][0] for n in stats.group_of}
    total = sum(2 * w * w * 4 for w in widths.values())
    total += sum(stats.dims[n][0] * stats.dims[n][1] * 4 for n in stats.dims)
    if total > _COV_BUDGET:
        raise RuntimeError(
            "{} This edit-site preset needs {:.1f}GB of solver matrices ({} independent "
            "input groups). Pick a narrower preset.".format(LOG, total / GB, len(widths))
        )


def _accumulate_cov(stats, captured, target, device):
    """target[group] += X^T X / tokens, once per group.

    Dividing by the token count makes a 12-token scene and an 80-token scene count
    equally. Without it the solve quietly optimises for whichever scenes were described
    at length.
    """
    done = set()
    for name, (x, _y) in captured.items():
        gid = stats.group_of[name]
        if gid in done:
            continue
        done.add(gid)
        flat = _flat(x)
        contribution = (flat.T @ flat).to(device) / max(1, flat.shape[0])
        if gid in target:
            target[gid] += contribution
        else:
            target[gid] = contribution


def collect_statistics(backend, model, clip, triplets, preservation, site_preset,
                       stats_device=None, progress=True):
    """Run the text tower over every prompt and gather the solve inputs.

    ``triplets`` is [(neutral, positive, negative), ...]; ``preservation`` is a list of
    prompts the slider must leave alone.

    Returns ``(stats, sites, conds)``. The conditioning cache comes back so the
    verification pass can reuse it -- otherwise it would have to reload the text encoder
    it just went to the trouble of evicting.
    """
    sites = backend.select_sites(model, site_preset)
    device = model.load_device
    stats_device = stats_device or device

    # ---- 1. encode everything first, then evict the text encoder ---------------------
    # The encoder (a 4B LLM on Krea 2) and the text tower do not need to be resident at
    # the same time; on a 12GB card insisting on both is the difference between a
    # two-second build and a swap storm.
    wanted = []
    for neutral, positive, negative in triplets:
        wanted.extend((neutral, positive, negative))
    wanted.extend(preservation)
    unique = list(dict.fromkeys(t for t in wanted if t and t.strip()))

    logging.info("%s Encoding %d unique prompts.", LOG, len(unique))
    pbar = comfy.utils.ProgressBar(len(unique)) if progress else None
    conds = {}
    for text in unique:
        comfy.model_management.throw_exception_if_processing_interrupted()
        conds[text] = backend.encode(clip, text)
        if pbar is not None:
            pbar.update(1)

    # Everything the encoder was needed for is a CPU tensor now; hand its VRAM back.
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    stats = SliderStats()
    stats.n_pairs = len(triplets)
    stats.n_preserve = len(preservation)

    # ---- 2. run the text tower ------------------------------------------------------
    total = len(triplets) * 3 + len(preservation)
    pbar = comfy.utils.ProgressBar(total) if progress else None
    modules = backend.text_path_modules(model)

    with torch.no_grad(), modules_on_device(modules, device), _SiteCapture(sites) as cap:

        def run(text):
            backend.run_text_path(model, conds[text], device)
            return cap.take()

        first = True
        for neutral, positive, negative in triplets:
            comfy.model_management.throw_exception_if_processing_interrupted()

            neu = run(neutral)
            if first:
                _assign_groups(stats, neu)
                first = False
            pos = run(positive)
            neg = run(negative)

            for name, (x_neu, y_neu) in neu.items():
                X = _flat(x_neu)
                Y_neu = _flat(y_neu)
                Y_pos, Y_neg = _flat(pos[name][1]), _flat(neg[name][1])

                # Shared-prefix alignment. The three prompts differ only by an appended
                # phrase, so positions below this bound describe the same tokens in all
                # three and can be compared one to one.
                prefix = min(X.shape[0], Y_pos.shape[0], Y_neg.shape[0])
                if prefix < 3:
                    stats.dropped_scenes += 1
                    continue
                X = X[:prefix]
                # Halved so that strength +1 lands on the positive prompt and -1 on the
                # negative, the neutral sitting between them.
                dY = (Y_pos[:prefix] - Y_neg[:prefix]) * 0.5
                raw_energy = float((dY * dY).sum())
                dY = _deflate(dY, Y_neu[:prefix])
                kept = float((dY * dY).sum())
                stats.raw_energy[name] = stats.raw_energy.get(name, 0.0) + raw_energy
                stats.kept_energy[name] = stats.kept_energy.get(name, 0.0) + kept

                contribution = (X.T @ X).to(stats_device) / prefix
                gid = stats.group_of[name]
                if gid in stats.cov_edit:
                    stats.cov_edit[gid] += contribution
                else:
                    stats.cov_edit[gid] = contribution

                cross = (dY.T @ X).to(stats_device) / prefix
                if name in stats.cross:
                    stats.cross[name] += cross
                else:
                    stats.cross[name] = cross

                stats.target_energy[name] = stats.target_energy.get(name, 0.0) + float(
                    (dY * dY).sum() / prefix)
                head = Y_neu[:prefix]
                stats.y_edit_energy[name] = stats.y_edit_energy.get(name, 0.0) + float(
                    (head * head).sum() / prefix)

                x_mean = X.mean(dim=0)
                stats.x_edit.setdefault(name, []).append(x_mean.to(stats_device))
                stats.dy_edit.setdefault(name, []).append(dY.mean(dim=0).to(stats_device))

                # --- the pooled candidate ---------------------------------------------
                # Whole-sequence means, so the concept's *own* tokens count. That target
                # is far larger than the aligned one -- on Krea 2 the shared prefix is
                # only reachable through two refiner-attention blocks -- but it is not
                # reproducible token for token, so the solve can only match it on average.
                # Bigger and blunter versus smaller and exact; the verification pass
                # decides which actually moves the image.
                dy_pooled = (Y_pos.mean(dim=0) - Y_neg.mean(dim=0)) * 0.5
                # This is where the length mismatch bites hardest -- the two means
                # divide the shared base content by different token counts -- so the
                # deflation matters most here.
                dy_pooled = _deflate(dy_pooled, Y_neu.mean(dim=0))
                pooled = torch.outer(dy_pooled, x_mean).to(stats_device)
                if name in stats.cross_pooled:
                    stats.cross_pooled[name] += pooled
                else:
                    stats.cross_pooled[name] = pooled
                stats.pooled_energy[name] = stats.pooled_energy.get(name, 0.0) + float(
                    (dy_pooled * dy_pooled).sum())
                stats.dy_pooled.setdefault(name, []).append(dy_pooled.to(stats_device))

            stats.prefix_lengths.append(
                min(int(conds[t].shape[1]) for t in (neutral, positive, negative)))
            if pbar is not None:
                pbar.update(3)

        for text in preservation:
            comfy.model_management.throw_exception_if_processing_interrupted()
            captured = run(text)
            if first:
                _assign_groups(stats, captured)
                first = False
            _accumulate_cov(stats, captured, stats.cov_pres, stats_device)
            for name, (x, y) in captured.items():
                stats.x_pres.setdefault(name, []).append(
                    _flat(x).mean(dim=0).to(stats_device))
                flat_y = _flat(y)
                stats.y_pres_energy[name] = stats.y_pres_energy.get(name, 0.0) + float(
                    (flat_y * flat_y).sum() / max(1, flat_y.shape[0]))
            if pbar is not None:
                pbar.update(1)

    if not stats.cross:
        raise RuntimeError(
            "{} Every scene was rejected: the three prompts of each triplet shared fewer "
            "than 3 tokens. Base prompts are probably empty.".format(LOG))

    # The measurement lives on the tokens the three prompts share, so it only exists if
    # the text tower mixes across positions. Krea 2's txtfusion blocks use bidirectional
    # self-attention, so an appended phrase changes how the base scene is represented --
    # that change is the signal. A tower that is purely position-wise (or a backend whose
    # run_text_path skips the attention part of it) produces an identical prefix for all
    # three prompts and therefore nothing to solve for. Say so, rather than shipping a
    # LoRA of zeros.
    for name in stats.cross:
        signal = stats.target_energy.get(name, 0.0)
        scale = stats.y_edit_energy.get(name, 0.0)
        if scale > 0.0 and signal / scale < 1e-8:
            raise RuntimeError(
                "{} Site {!r} shows no concept signal: the positive and negative prompts "
                "produce an identical shared prefix ({:.2e} relative). The text tower "
                "this backend runs appears to be position-wise, so an appended phrase "
                "cannot influence the tokens a LoRA is able to move. The edit site has to "
                "sit downstream of the tower's attention blocks.".format(
                    LOG, name, signal / max(scale, 1e-30)))

    for name in stats.cross:
        aligned = stats.target_energy.get(name, 0.0) ** 0.5
        pooled = stats.pooled_energy.get(name, 0.0) ** 0.5
        stats.signal_ratio[name] = pooled / max(aligned, 1e-30)
        raw = stats.raw_energy.get(name, 0.0)
        kept = stats.kept_energy.get(name, 0.0)
        stats.contamination[name] = 1.0 - (kept / raw if raw > 0 else 1.0)
        if stats.contamination[name] > 0.5:
            logging.info(
                "%s %.0f%% of the measured concept at %s was conditioning magnitude "
                "rather than direction, and was projected out. That component is a "
                "guidance dial, not a slider.",
                LOG, 100.0 * stats.contamination[name], name)

    comfy.model_management.soft_empty_cache()
    return stats, sites, conds


# --------------------------------------------------------------------------------------
# the closed form
# --------------------------------------------------------------------------------------

def _quadratic(up, down, cov):
    """trace(dW C dW^T) for dW = up @ down, without forming dW.

    The total squared movement the edit applies across every token summarised in ``C``.
    ``down @ C @ down.T`` is only r x r, so this costs one skinny matmul.
    """
    return float(((up @ (down @ cov @ down.T)) * up).sum())


def _solve_site(C_edit, C_pres, preservation_weight, D, X, DY, target_energy,
                eta, ridge, rank):
    """Return (lora_up, lora_down, diagnostics) for one edit site.

    All inputs are CPU tensors in the solve dtype. ``D`` is the (d_out, d_in) cross
    matrix, ``X``/``DY`` the per-scene means used for the readable diagnostics.
    """
    d_in = C_edit.shape[0]
    dtype = C_edit.dtype

    M = C_edit if C_pres is None else C_edit + C_pres * float(preservation_weight)
    # Ridge relative to the mean eigenvalue, so one number means the same thing across
    # models, layer widths and pack sizes.
    scale = torch.diagonal(M).mean().clamp_min(1e-12)
    M = M + torch.eye(d_in, dtype=dtype) * (float(ridge) * scale)

    try:
        chol = torch.linalg.cholesky(M)
        solved = torch.cholesky_solve(D.T.contiguous(), chol)      # (d_in, d_out)
    except Exception as exc:  # noqa: BLE001 - ridge too small to make M positive definite
        logging.warning("%s Cholesky failed (%s); using a general solve. Raising "
                        "regularization would be faster.", LOG, exc)
        solved = torch.linalg.solve(M, D.T.contiguous())
    del M
    dW = (solved.T * float(eta)).contiguous()                       # (d_out, d_in)
    del solved

    # Rank-r factorisation. Randomised SVD with oversampling: dW is 6144x6144 here and a
    # full SVD would dominate the runtime for no gain, since only the top few dozen
    # singular directions are ever kept.
    full_energy = float((dW * dW).sum())
    q = int(min(min(dW.shape), max(int(rank) + 12, int(rank) * 2)))
    U, S, V = torch.svd_lowrank(dW, q=q, niter=6)
    r = int(max(1, min(int(rank), S.numel())))
    up = (U[:, :r] * S[:r]).contiguous()                            # (d_out, r)
    down = V[:, :r].T.contiguous()                                  # (r, d_in)

    curve = []
    for k in range(1, S.numel() + 1):
        remaining = full_energy - float((S[:k] ** 2).sum())
        curve.append((max(0.0, remaining) / max(full_energy, 1e-30)) ** 0.5)

    # ---- diagnostics on the *truncated* edit, i.e. on what actually ships ------------
    target_energy = max(float(target_energy) * float(eta) ** 2, 1e-30)

    # Per token: the honest number. residual = sum ||dW x - eta dy||^2, expanded through
    # the covariance so no per-token data has to be kept around.
    cross_term = float(((down @ D.T.contiguous()) * up.T).sum()) * float(eta)
    token_fit = 1.0 - (_quadratic(up, down, C_edit) - 2.0 * cross_term
                       + target_energy) / target_energy

    # Per scene: readable, and always kinder than token_fit.
    achieved = (X @ down.T) @ up.T
    wanted = DY * float(eta)
    den = (achieved.norm(dim=1).clamp_min(1e-12)
           * wanted.norm(dim=1).clamp_min(1e-12))
    cosine = (achieved * wanted).sum(dim=1) / den
    strength = achieved.norm(dim=1) / wanted.norm(dim=1).clamp_min(1e-12)

    # Do the scenes even agree on what the concept is? If the base prompts collide with
    # the concept phrase, the measured directions point every which way and their
    # average is noise -- this catches that before an image is ever generated.
    unit = DY / DY.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mean_dir = unit.mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp_min(1e-12)
    agreement = unit @ mean_dir

    info = {
        "rank": r,
        "probed_rank": int(S.numel()),
        "token_fit": token_fit,
        "direction_match": float(cosine.mean()),
        "direction_match_min": float(cosine.min()),
        "strength_ratio": float(strength.mean()),
        "concept_agreement": float(agreement.mean()),
        "concept_agreement_min": float(agreement.min()),
        "rank_energy_kept": float((S[:r] ** 2).sum()) / max(full_energy, 1e-30),
        "rank_error_curve": curve,
    }
    return up, down, info


def _leakage(up, down, cov_pres, y_pres_energy, n_pres, x_pres):
    """How much the finished edit disturbs prompts it was told not to touch.

    ``rms`` is the headline: total squared movement over every preservation token,
    relative to the size of those tokens' own outputs. ``max`` is the worst single
    prompt, to catch a set where one prompt absorbs all the damage.
    """
    if cov_pres is None or not n_pres or y_pres_energy <= 0.0:
        return None
    rms = (max(0.0, _quadratic(up, down, cov_pres)) / max(y_pres_energy, 1e-30)) ** 0.5
    worst = None
    if x_pres is not None and x_pres.numel():
        moved = ((x_pres @ down.T) @ up.T).norm(dim=1)
        worst = float(moved.max()) / max((y_pres_energy / n_pres) ** 0.5, 1e-30)
    return {"rms": rms, "max": worst}


OBJECTIVES = ("aligned", "pooled")


def build_lora(stats, sites, eta, rank, ridge, preservation_weight=1.0,
               objective="aligned", solve_dtype=torch.float64,
               save_dtype=torch.float32):
    """Solve every site and assemble a ComfyUI-loadable LoRA state dict.

    Keys are ``<module>.lora_up.weight`` / ``.lora_down.weight`` / ``.alpha`` with module
    names straight from ``named_modules()``. ComfyUI's generic LoRA key map accepts
    exactly that, so the file loads in the stock Load LoRA nodes with no conversion --
    and can be handed to a gradient trainer as a warm start instead of zeros.
    """
    lora = {}
    report_sites = {}

    # Splitting eta across sites keeps the total shift at the requested magnitude when a
    # preset edits more than one layer.
    per_site_eta = float(eta) / max(1, len(sites))

    for name, _module in sites:
        gid = stats.group_of[name]
        C_edit = stats.cov_edit[gid].to("cpu", solve_dtype)
        C_pres = (stats.cov_pres[gid].to("cpu", solve_dtype)
                  if gid in stats.cov_pres else None)
        X = torch.stack(stats.x_edit[name]).to("cpu", solve_dtype)
        if objective == "pooled":
            D = stats.cross_pooled[name].to("cpu", solve_dtype)
            DY = torch.stack(stats.dy_pooled[name]).to("cpu", solve_dtype)
            energy = stats.pooled_energy[name]
        elif objective == "aligned":
            D = stats.cross[name].to("cpu", solve_dtype)
            DY = torch.stack(stats.dy_edit[name]).to("cpu", solve_dtype)
            energy = stats.target_energy[name]
        else:
            raise ValueError("{} Unknown objective {!r}; expected one of {}.".format(
                LOG, objective, ", ".join(OBJECTIVES)))

        up, down, info = _solve_site(
            C_edit, C_pres, preservation_weight, D, X, DY,
            energy, per_site_eta, ridge, rank,
        )

        # ---- put the edit on a scale that means something ---------------------------
        # The solve returns the *smallest* update consistent with the target, and after
        # ridge, preservation and rank truncation what survives can be a small fraction
        # of the concept. On Krea 2 the shipped edit shifted the conditioning by 1.5-3%,
        # which is invisible until LoRA strength ~20 -- the exact complaint this package
        # exists to avoid.
        #
        # So normalise: scale the edit until the conditioning shift it produces, in RMS
        # across the collected scenes, equals the shift that appending the concept phrase
        # itself produces. That makes strength 1.0 mean "as strong as writing it into the
        # prompt" by construction, needs no sampler, and cannot fail. The verification
        # pass then refines this and fixes the sign; it is no longer the only thing
        # standing between the user and an unusably faint slider.
        n_scenes = max(1, len(stats.x_edit[name]))
        achieved = (max(_quadratic(up, down, C_edit), 0.0) / n_scenes) ** 0.5
        reference = (max(float(stats.pooled_energy.get(name, 0.0)), 0.0) / n_scenes) ** 0.5
        # ``achieved`` already contains eta (it multiplied the target inside the solve),
        # so a bare reference/achieved ratio would divide eta straight back out and turn
        # the strength widget into a no-op. Multiply it back in.
        auto_scale = 1.0
        if achieved > 1e-12 and reference > 0.0:
            normalise = min(max(reference / achieved, 0.05), 200.0)
            auto_scale = float(normalise * per_site_eta)
            up = up * auto_scale
        info["auto_scale"] = auto_scale
        info["shift_rms"] = achieved * auto_scale
        info["prompt_shift_rms"] = reference * per_site_eta
        info["objective"] = objective
        info["signal_ratio"] = stats.signal_ratio.get(name, 1.0)
        info["contamination"] = stats.contamination.get(name, 0.0)

        x_pres = (torch.stack(stats.x_pres[name]).to("cpu", solve_dtype)
                  if stats.x_pres.get(name) else None)
        info["preservation_leakage"] = _leakage(
            up, down, C_pres, stats.y_pres_energy.get(name, 0.0),
            len(stats.x_pres.get(name, [])), x_pres)
        info["d_in"], info["d_out"] = stats.dims[name]
        info["eta"] = per_site_eta

        lora["{}.lora_up.weight".format(name)] = up.to(save_dtype).contiguous()
        lora["{}.lora_down.weight".format(name)] = down.to(save_dtype).contiguous()
        # alpha == rank makes ComfyUI's alpha/rank scale exactly 1.0, so LoRA strength is
        # the eta dial and nothing is silently rescaled.
        lora["{}.alpha".format(name)] = torch.tensor(float(info["rank"]),
                                                     dtype=torch.float32)
        report_sites[name] = info

    return lora, report_sites


def rescale_lora(lora, gain):
    """Multiply the whole edit by a scalar, in place, keeping rank and alpha."""
    if gain == 1.0:
        return lora
    for key in list(lora):
        if key.endswith(".lora_up.weight"):
            lora[key] = (lora[key] * float(gain)).contiguous()
    return lora


# --------------------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------------------

def save_lora(lora, concept_name, rank, save_name, metadata):
    """Write into the first configured ``loras`` folder and return the full path."""
    lora_dirs = folder_paths.get_folder_paths("loras")
    target_dir = lora_dirs[0] if lora_dirs else folder_paths.get_output_directory()
    os.makedirs(target_dir, exist_ok=True)

    if save_name and save_name.strip():
        name = sanitize_name(save_name)
    else:
        name = "{}_zt_slider_r{}".format(sanitize_name(concept_name), rank)
    full_path = os.path.join(target_dir, "{}_{}.safetensors".format(name, int(time.time())))

    safetensors.torch.save_file(
        {k: v.cpu().contiguous() for k, v in lora.items()},
        full_path,
        metadata={k: str(v) for k, v in metadata.items()},
    )
    logging.info("%s Saved zero-training slider LoRA to %s", LOG, full_path)
    return full_path
