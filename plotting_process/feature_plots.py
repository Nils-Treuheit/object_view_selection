"""
Per-feature dataset overview and bad-example plots.

Ported from nit_view_selection/select_best_views.py (plot_dataset_overview /
plot_bad_examples) and extended to cover every feature (all pre-filter raw
stats plus all quality component scores), not just the Vincent soft filters.

Output layout (raw stats in the pre-filter plots dir, quality scores in the
selection plots dir):
  <pre-filter>/data_set_overview/
    <feature>_filter_fixed.png       # goodness on a fixed coolwarm 0..1 scale (bounded features only)
    <feature>_filter_relative.png    # goodness on a data-relative viridis scale
  <selection>/data_set_overview/
    quality_score_<feature>_fixed.png
    quality_score_<feature>_relative.png
  bad_examples/                      # sibling of plots/, split by pipeline stage

Each data_set_overview image has a distribution histogram on the left (x-axis
fixed to [-0.05, 1.05] when the feature is bounded to [0, 1]) and the feature
value over the frame sequence on the right. Histogram bars are centred on the
bin values: bounded features use a fixed 0..1 grid so every bar is centre
aligned from 0.0 all the way to 1.0 (bars centred at 0.0, 0.025, ..., 1.0).

Reported values use a persistent meaning too: for the lower-is-better raw
stats (border ratio, edge ratio, hand overlap, artifact fraction) the plots
report ``1 - value`` — the "free" share — so a higher reported value is always
better, exactly like every other feature.

Coloring uses a *persistent* meaning across every plot: warm/bright = good,
cold/dark = bad, regardless of the feature. The two variants per feature
differ in the colourbar scale:

  *_fixed.png     colourbar pinned to 0..1, generated only for features whose
                  values are naturally bounded to [0, 1] (all quality scores
                  and the ratio stats). The colour is the absolute reported
                  value, so the dot colour matches the colourbar tick labels
                  exactly. Unbounded counting stats (Laplacian, Tenengrad,
                  boundary-blur variance) have no fixed 0..1 meaning and get
                  the relative plot only.
  *_relative.png  colourbar adjusted to this dataset's observed value range,
                  with the min rounded *down* and the max rounded *up* to the
                  second decimal place, and ticked in the reported units of
                  the feature (1 - value for the lower-is-better stats), at
                  most 3 decimal places, so it reads as "relative to this
                  dataset".

Neither title nor colourbar carries a "(fixed 0..1)" / "(relative …)" suffix;
the colourbar is just the numbers, written out in full (no e-notation). The
overview titles are ``<Label> (statistical pre-filter)`` for the raw stats and
``<Label> Score`` for the quality scores.

The pre-filter_stage images only ever show frames that were actually filtered
out for that feature's own reason. The worst frame is always included; the
rest are picked worst-first but must look visually distinct from the frames
already shown, so a run of near identical video frames never fills the whole
row. Placeholder tiles fill any empty slots. When a pre-filter's reason never
fired, and for every quality score, the image instead probability-samples the
lowest-quality *accepted* frames (worst = highest likelihood), so these plots
show what low quality looks like without pretending those frames were
filtered.

Every thumbnail border is coloured with the *viridis* colour of the relative
score over the min/max of all samples — the same scale as the relative
``data_set_overview`` colourbar — and each thumbnail is labelled with a
status line and ``#<frame id> | QS: <reported value>``: filtered frames show
``rejected - <reason>``, pre-filter lower-quality frames show
``accepted - <feature label>`` and selection-stage frames show
``accepted but not selected``.
"""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# matplotlib's coolwarm: cold (low values) = blue, warm (high values) = red.
# Used for the fixed 0..1 goodness scale, so 0.0 = bad (cold), 1.0 = good (warm).
WARM_COLD_CMAP = plt.cm.coolwarm
# viridis (dark = low, bright = high) for the data-relative goodness scale.
REL_CMAP = plt.cm.viridis

# Colourbar label is intentionally empty: the ticks (fully written numbers)
# carry all the information, and the fixed/relative distinction is in the
# filename, not in a textual label.
GOOD_FIXED_LABEL = ""
GOOD_RELATIVE_LABEL = ""

# Rejection reasons → histogram colors. "border" is the truncation filter
# (object cut off at the frame edge) and "occlusion" is hand / other-object
# coverage; both are kept as distinct categories everywhere. The default
# blur/artifact filters reject below a relaxed absolute floor ("_threshold")
# or as extreme bad outliers ("_outlier") — both modes share one color.
REJECT_COLORS = {
    "vincent_empty_mask": "dimgray",
    "vincent_border_pixel": "tab:purple",
    "border": "tab:blue",
    "truncation": "tab:blue",
    "area": "tab:orange",
    "small_object": "tab:orange",
    "low_confidence": "tab:green",
    "confidence": "tab:green",
    "blur": "tab:brown",
    "blur_laplacian": "tab:brown",
    "blur_laplacian_threshold": "tab:brown",
    "blur_laplacian_outlier": "tab:brown",
    "blur_tenengrad": "goldenrod",
    "blur_tenengrad_threshold": "goldenrod",
    "blur_tenengrad_outlier": "goldenrod",
    "vincents_artefacts": "tab:olive",
    "vincents_artefacts_threshold": "tab:olive",
    "vincents_artefacts_outlier": "tab:olive",
    "occlusion": "tab:pink",
    "motion_blur": "tab:red",
    "completeness": "tab:cyan",
    "incomplete_shape": "tab:cyan",
    "empty_mask": "dimgray",
}
FALLBACK_REJECT_COLOR = "tab:red"

# (metric attr, readable label, good_direction) where good_direction:
#   1  = higher is better
#   -1 = lower is better (reported inverted as 1 - value so a higher reported
#        value is always better)
RAW_FEATURES = [
    ("laplacian", "Laplacian", 1),
    ("tenengrad", "Tenengrad", 1),
    ("area_ratio", "Area Ratio", 1),
    ("border_ratio", "Border-Free Ratio", -1),
    ("edge_ratio", "Edge-Free Ratio", -1),
    ("hand_overlap", "Hand-Free Ratio", -1),
    ("completeness", "Completeness", 1),
    ("vincent_area_fraction", "Mask Area Fraction", 1),
    ("vincent_artifact_fraction", "Artifact-Free Fraction", -1),
    ("vincent_boundary_blur_variance", "Boundary Blur Variance", 1),
]

QUALITY_FEATURES = [
    ("blur", "Blur Quality", 1),
    ("area", "Area Quality", 1),
    ("vincents_artefacts", "Artifacts Quality", 1),
    ("centerness", "Centerness Quality", 1),
    ("confidence", "Confidence", 1),
    ("score", "Final Quality", 1),
]

# 5 example frames per feature in the bad_examples plots
BAD_EXAMPLES_PER_FEATURE = 5

# A candidate bad-example thumbnail must differ from every already-picked one
# by at least this much (mean abs diff on a 0..255 grayscale thumbnail) to be
# considered a distinct frame. Adjacent turntable frames typically differ by
# ~8-9, so 12.0 makes sure a run of near-identical consecutive video frames
# never fills the whole row.
BAD_EXAMPLE_MIN_IMG_DIFF = 12.0

# Fixed histogram x-limits for features bounded to [0, 1]
FIXED_HIST_XLIM = (-0.05, 1.05)

# Features whose raw values are naturally bounded to [0, 1]. For these the
# fixed 0..1 colorbar can colour by the absolute value itself (value, or
# 1 - value for lower-is-better features), so a dot at quality 0.99 is always
# warm and the colour matches the colourbar's tick labels. Unbounded features
# (laplacian, tenengrad, boundary-blur variance) have no absolute scale and
# fall back to a dataset-relative goodness.
BOUNDED_FEATURES = {
    "area_ratio", "border_ratio", "edge_ratio", "hand_overlap",
    "completeness", "vincent_area_fraction", "vincent_artifact_fraction",
    "blur", "area", "vincents_artefacts", "centerness",
    "confidence", "score",
}

# Overlay dispatch: which raw stat a quality feature maps to for highlighting.
_QUALITY_TO_RAW = {
    "blur": "laplacian",
    "area": "area_ratio",
    "vincents_artefacts": "vincent_artifact_fraction",
    "centerness": "area_ratio",
    "confidence": "score",
    "score": "score",
}


def _feature_value(obs, attr):
    if attr == "score":
        return float(getattr(obs, "quality", np.nan))
    v = getattr(obs.metrics, attr, None)
    return float(v) if v is not None else np.nan


def _feature_values(observations, attr):
    return np.array([_feature_value(obs, attr) for obs in observations])


# attr -> good_direction lookup shared by RAW_FEATURES and QUALITY_FEATURES
_FEATURE_DIRECTION = {attr: gd for attr, _, gd in RAW_FEATURES + QUALITY_FEATURES}

# Pre-filter raw stat -> the rejection reasons that feature detects. A frame is
# shown in ``<attr>_filtered.png`` only when it was rejected for one of these
# reasons (the truncation filters "border"/"truncation" and the border-pixel
# detector "vincent_border_pixel" all detect the same failure mode and share
# the border/edge stats). A feature with no reasons (the Vincent soft stats)
# never hard-rejects and always falls back to the prob-sampled
# ``lower_<attr>_quality.png`` form.
_FEATURE_REASONS = {
    "laplacian": ("blur_laplacian_threshold", "blur_laplacian_outlier", "blur",),
    "tenengrad": ("blur_tenengrad_threshold", "blur_tenengrad_outlier", "blur",),
    "area_ratio": ("small_object",),
    "border_ratio": ("border", "truncation", "vincent_border_pixel"),
    "edge_ratio": ("border", "truncation", "vincent_border_pixel"),
    "hand_overlap": ("occlusion",),
    "completeness": ("incomplete_shape", "completeness"),
    "vincent_area_fraction": ("vincent_empty_mask", "empty_mask"),
    "vincent_artifact_fraction": ("vincents_artefacts_threshold", "vincents_artefacts_outlier"),
    "vincent_boundary_blur_variance": (),
}


def _report_value(attr, value):
    """Feature value as reported in the plots.

    Lower-is-better features (border ratio, edge ratio, hand overlap, artifact
    fraction) are reported inverted as ``1 - value`` so that a higher reported
    value is always better, matching the persistent warm=good colouring.
    """
    if _FEATURE_DIRECTION.get(attr, 1) == -1:
        return 1.0 - value
    return value


def _format_tick(value):
    """Format a colourbar/axis tick fully written out (never e-notation)."""
    if value is None:
        return ""
    if value == 0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _round_range(lo, hi):
    """Round the data min down and max up to the second decimal place.

    The relative colourbar scale uses this clean outward-rounded range instead
    of the exact observed min/max, so the scale is stable across runs.
    """
    lo_r = np.floor(lo * 100.0) / 100.0
    hi_r = np.ceil(hi * 100.0) / 100.0
    if hi_r <= lo_r:
        lo_r = np.floor((lo - 0.01) * 100.0) / 100.0
        hi_r = np.ceil((hi + 0.01) * 100.0) / 100.0
    return float(lo_r), float(hi_r)


def _format_relative_tick(value):
    """Relative colourbar tick: fully written, at most 3 decimal places."""
    return _format_tick(round(value, 3))


def _goodness(value, lo, hi, good_direction):
    """Map a raw feature value to [0, 1] goodness where 1.0 = always good.

    ``good_direction`` is 1 when higher values are better and -1 when lower
    values are better (the inverted features: border ratio, hand overlap,
    artifact fraction). The mapping is relative to the dataset's observed
    min/max, so a bad frame always lands near 0 and a good frame near 1.
    """
    if hi > lo:
        frac = (value - lo) / (hi - lo)
    else:
        frac = 0.5
    return frac if good_direction == 1 else 1.0 - frac


def _abs_goodness(value, good_direction):
    """Absolute goodness for [0, 1]-bounded features: 1.0 is always good.

    Unlike ``_goodness`` (which is relative to the dataset's observed min/max),
    this maps the raw value directly onto the fixed 0..1 scale, so the colour
    of a dot matches the colourbar's tick labels exactly.
    """
    g = value if good_direction == 1 else 1.0 - value
    return float(np.clip(g, 0.0, 1.0))


def _load_image(obs):
    """Return (image_rgb, mask_u8) loading from disk if not already loaded."""
    image = getattr(obs, "image", None)
    mask = getattr(obs, "mask", None)
    if image is None:
        image_path = getattr(obs, "image_path", None)
        if image_path and Path(image_path).exists():
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mask is None:
        mask_path = getattr(obs, "mask_path", None)
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, mask
    return image, mask


def _mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy().astype(np.float32)
    foreground = mask > 0
    if foreground.any():
        overlay[foreground] = (
            (1 - alpha) * overlay[foreground] + alpha * np.array(color)
        ).astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _feature_overlay(obs, feature_attr):
    """Overlay a thumbnail emphasizing the given feature."""
    image, mask = _load_image(obs)
    if image is None or mask is None:
        return None
    foreground = (mask > 0).astype(np.uint8)
    overlay = _mask_overlay(image, mask, (0, 255, 0), alpha=0.45)

    if feature_attr == "vincent_artifact_fraction":
        from preprocessing.vincent_utils import compute_artifact_mask
        artifact = compute_artifact_mask(foreground, kernel_size=3)
        overlay[artifact] = (200, 60, 60)
    elif feature_attr == "vincent_boundary_blur_variance":
        from preprocessing.vincent_utils import compute_boundary_band
        band = compute_boundary_band(foreground, stroke_width=9)
        overlay = (image.astype(np.float32) * 0.5).astype(np.uint8)
        overlay[band] = image[band]
    elif feature_attr in ("border_ratio", "edge_ratio"):
        overlay[:2, :] = (220, 60, 60)
        overlay[-2:, :] = (220, 60, 60)
        overlay[:, :2] = (220, 60, 60)
        overlay[:, -2:] = (220, 60, 60)
    return overlay


# --------------------------------------------------------------------------- #
# Dataset overview
# --------------------------------------------------------------------------- #


def _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                     feature_attr, label, good_direction, cmap, norm,
                     colorbar_label, color_of=None, colorbar_ticks=None,
                     value_of=None, bins=None):
    sel_ids = {s.id for s in selected}

    if value_of is None:
        value_of = lambda v: v

    acc_vals = np.array([value_of(_feature_value(o, feature_attr)) for o in accepted])
    rej_vals = np.array([value_of(_feature_value(o, feature_attr)) for o in rejected])

    # shared axis range over the combined, non-NaN values
    combined = np.concatenate([acc_vals, rej_vals])
    combined = combined[~np.isnan(combined)]
    if combined.size == 0:
        ax_hist.axis("off")
        ax_scatter.axis("off")
        return
    lo, hi = float(combined.min()), float(combined.max())

    # bars are centred on the distribution bins: matplotlib's align='left'
    # places each bar centred on the provided bin edge, so the bar covering
    # value v is centred on v (e.g. the bar for 0.0 is centred at 0.0, not at
    # 0.0 + width/2). Bounded features pass fixed edges spanning the whole
    # 0..1 range (with one extra bin width past 1.0, so the trailing bar that
    # represents values in [0.975, 1.0] is centred exactly on 1.0) and the
    # bars line up on the 0.0, 0.025, ..., 1.0 grid; unbounded features fall
    # back to the observed data range.
    nbins = 40
    if bins is None:
        if hi > lo:
            bin_edges = np.linspace(lo, hi, nbins + 1)
        else:
            bin_edges = np.linspace(lo - 0.5, lo + 0.5, nbins + 1)
    else:
        bin_edges = bins

    if color_of is None:
        color_of = lambda v: _goodness(v, lo, hi, good_direction)

    # left: distribution histogram
    reasons = sorted({r for r in (o.rejection_reason for o in rejected) if r})
    if len(rej_vals[~np.isnan(rej_vals)]) > 0:
        ax_hist.hist(rej_vals, bins=bin_edges, align="left", alpha=0.35,
                     color="#d3d3d3", label="rejected")
    for reason in reasons:
        reason_vals = np.array([
            value_of(_feature_value(o, feature_attr)) for o in rejected
            if o.rejection_reason == reason
        ])
        reason_vals = reason_vals[~np.isnan(reason_vals)]
        color = REJECT_COLORS.get(reason, FALLBACK_REJECT_COLOR)
        ax_hist.hist(reason_vals, bins=bin_edges, align="left", alpha=0.6,
                     color=color, label=reason)
    if len(acc_vals[~np.isnan(acc_vals)]) > 0:
        ax_hist.hist(acc_vals, bins=bin_edges, align="left", alpha=0.6,
                     color="#2a9d8f", label="accepted")
    for s in selected:
        v = value_of(_feature_value(s, feature_attr))
        if not np.isnan(v):
            ax_hist.axvline(v, color="black", linestyle="--", linewidth=0.8)
    ax_hist.set_xlabel(label, fontsize=8)
    ax_hist.set_ylabel("num frames", fontsize=7)
    ax_hist.tick_params(labelsize=6)
    ax_hist.legend(fontsize=6, loc="upper right")
    if lo >= 0.0 and hi <= 1.0:
        ax_hist.set_xlim(*FIXED_HIST_XLIM)

    # right: feature value over the sequence, colored by persistent goodness
    all_obs = list(rejected) + list(accepted)
    all_obs = sorted(all_obs, key=lambda o: o.id)
    idx = np.arange(len(all_obs))
    vals = np.array([value_of(_feature_value(o, feature_attr)) for o in all_obs])

    # 1.0 = good, 0.0 = bad regardless of the raw feature's good_direction
    good = np.array([color_of(v) if not np.isnan(v) else np.nan for v in vals])

    valid = ~np.isnan(good)
    if valid.any():
        ax_scatter.scatter(
            idx[valid], vals[valid], s=16,
            c=good[valid], cmap=cmap, norm=norm,
            zorder=2,
        )
    for o, i, v in zip(all_obs, idx, vals):
        if o.id in sel_ids and not np.isnan(v):
            ax_scatter.scatter(i, v, s=60, color="gold", edgecolor="black", zorder=3)

    fig = ax_scatter.get_figure()
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_scatter, shrink=0.8, label=colorbar_label,
    )
    if colorbar_ticks:
        positions = [pos for pos, _ in colorbar_ticks]
        cb.set_ticks(positions)
        cb.set_ticklabels([str(tick) for _, tick in colorbar_ticks])
    ax_scatter.set_xlabel("frame index", fontsize=8)
    ax_scatter.set_ylabel(label, fontsize=8)
    ax_scatter.tick_params(labelsize=6)


def plot_dataset_overview(accepted, rejected, selected, dir_pre, dir_selection):
    """Two data_set_overview images per feature (fixed + relative goodness).

    Raw pre-filter stats land in ``dir_pre/data_set_overview`` under the
    ``<feature>_filter_{fixed,relative}.png`` names; quality-score overviews
    land in ``dir_selection/data_set_overview`` under
    ``quality_score_<feature>_{fixed,relative}.png``.
    """
    dir_pre = Path(dir_pre)
    dir_selection = Path(dir_selection)
    for group, features in [
        ("raw", RAW_FEATURES),
        ("quality", QUALITY_FEATURES),
    ]:
        out_dir = (dir_selection if group == "quality" else dir_pre) / "data_set_overview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, label, good_direction in features:
            # reported values: lower-is-better features are inverted (1 - value)
            # so a higher reported value is always better everywhere.
            combined = np.array([_report_value(attr, v) for v in _feature_values(accepted, attr)])
            if rejected:
                rej_reported = np.array([_report_value(attr, v) for v in _feature_values(rejected, attr)])
                combined = np.concatenate([combined, rej_reported])
            combined = combined[~np.isnan(combined)]
            if combined.size:
                lo, hi = float(combined.min()), float(combined.max())
            else:
                lo, hi = 0.0, 1.0
            bounded = attr in BOUNDED_FEATURES

            # relative colourbar scale: the data min is rounded down and the
            # max up to the second decimal place, and the ticks (at most 3
            # decimal places) label that clean range in reported units.
            rel_lo, rel_hi = _round_range(lo, hi)
            if rel_hi <= rel_lo:
                rel_norm = Normalize(rel_lo - 0.5, rel_lo + 0.5)
                rel_ticks = None
            else:
                rel_norm = Normalize(0.0, 1.0)
                rel_ticks = [
                    (0.0, _format_relative_tick(rel_lo)),
                    (0.5, _format_relative_tick(0.5 * (rel_lo + rel_hi))),
                    (1.0, _format_relative_tick(rel_hi)),
                ]

            title = (f"{label} (statistical pre-filter)" if group == "raw"
                     else f"{label} Score")
            # raw stats: <feature>_filter_{fixed,relative}.png
            # quality scores: quality_score_<feature>_{fixed,relative}.png
            if group == "raw":
                fixed_stem = f"{attr}_filter_fixed"
                rel_stem = f"{attr}_filter_relative"
            else:
                fixed_stem = f"quality_score_{attr}_fixed"
                rel_stem = f"quality_score_{attr}_relative"
            # fixed 0..1 grid for bounded features: 41 bins of width 0.025,
            # with one extra bin width past 1.0 so the final bar is centred on
            # 1.0 (align='left' centres bars on the leading edge of each bin)
            overview_bins = np.linspace(0.0, 1.0 + 0.025, 42) if bounded else None
            # fixed variant: only meaningful for [0, 1]-bounded features.
            # Unbounded counting stats (laplacian, tenengrad, boundary-blur
            # variance) have no fixed 0..1 scale and get the relative plot only.
            if bounded:
                fixed_color_of = lambda v: _abs_goodness(v, 1)
                fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 4.5))
                _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                                 attr, label, good_direction,
                                 WARM_COLD_CMAP, Normalize(0.0, 1.0), GOOD_FIXED_LABEL,
                                 color_of=fixed_color_of,
                                 value_of=lambda v: _report_value(attr, v),
                                 bins=overview_bins)
                fig.suptitle(title, fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                fixed_path = out_dir / f"{fixed_stem}.png"
                fig.savefig(fixed_path, dpi=150)
                plt.close(fig)
                print(f"  Saved {fixed_path}")

            # relative variant: colourbar adjusted to this dataset's observed
            # value range, shown in the reported units of the feature.
            fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 4.5))
            _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                             attr, label, good_direction,
                             REL_CMAP, rel_norm, GOOD_RELATIVE_LABEL,
                             color_of=lambda v: _goodness(v, rel_lo, rel_hi, 1),
                             colorbar_ticks=rel_ticks,
                             value_of=lambda v: _report_value(attr, v),
                             bins=overview_bins)
            fig.suptitle(title, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            rel_path = out_dir / f"{rel_stem}.png"
            fig.savefig(rel_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {rel_path}")


# --------------------------------------------------------------------------- #
# Bad examples (up to 5 filtered-out frames per feature)
# --------------------------------------------------------------------------- #


def _curated_bad_examples(rejected, feature_attr, good_direction, k):
    """Pick k representative filtered-out frames: worst-first but distinct.

    The single worst frame is always included. Subsequent picks are the worst
    remaining frames that look sufficiently different (thumbnail-level mean
    absolute difference >= BAD_EXAMPLE_MIN_IMG_DIFF) from every frame already
    chosen, so a run of near-identical consecutive video frames does not fill
    the whole row. When images are unavailable (mock observations) or no frame
    is distinct enough, the pick falls back to spreading across the feature
    value range.
    """
    scored = [(o, _feature_value(o, feature_attr)) for o in rejected]
    scored = [(o, v) for o, v in scored if not np.isnan(v)]
    if not scored:
        return []
    scored.sort(key=lambda t: t[1] * good_direction)  # worst first

    thumbs = {}

    def _thumb(o):
        if o.id not in thumbs:
            image, _ = _load_image(o)
            if image is None:
                thumbs[o.id] = None
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                thumbs[o.id] = cv2.resize(gray, (48, 48)).astype(np.float32)
        return thumbs[o.id]

    def _min_img_diff(o):
        t = _thumb(o)
        if t is None:
            return None
        best = None
        for s in picked:
            st = _thumb(s[0])
            if st is None:
                continue
            d = float(np.mean(np.abs(t - st)))
            if best is None or d < best:
                best = d
        return best

    picked = [scored[0]]
    rest = scored[1:]
    while len(picked) < k and rest:
        cand = None
        for o, v in rest:
            d = _min_img_diff(o)
            if d is not None and d >= BAD_EXAMPLE_MIN_IMG_DIFF:
                cand = (o, v)
                break
        if cand is None:
            # no remaining frame is distinct enough (images unavailable, or all
            # remaining frames look alike e.g. when every rejected frame shares
            # the same degenerate value): pick the frame that is farthest from
            # the ones already chosen, preferring image difference and falling
            # back to the feature-value distance when images are unavailable.
            def _sep(o, v):
                d = _min_img_diff(o)
                if d is not None:
                    return d
                return min((abs(v - pv) for _, pv in picked), default=0.0)

            cand = max(rest, key=lambda t: _sep(t[0], t[1]))
        picked.append(cand)
        rest.remove(cand)
    return picked


def _prob_sample_low_quality(pool, feature_attr, good_direction, k, rng=None):
    """Sample k frames with probability weighted by badness (worst most likely).

    ``pool`` is a list of observations. Frames are drawn without replacement
    and each frame's weight is its relative badness on ``feature_attr``
    (1.0 = the worst frame in the pool, 0.0 = the best), so the lowest-quality
    frames are picked with the highest likelihood. Images are not loaded, so
    near-duplicate consecutive frames may co-occur — that is the point of a
    probability sample.
    """
    scored = [(o, _feature_value(o, feature_attr)) for o in pool]
    scored = [(o, v) for o, v in scored if not np.isnan(v)]
    if not scored:
        return []
    vals = np.array([v for _, v in scored])
    lo, hi = float(vals.min()), float(vals.max())
    if hi > lo:
        good = np.array([_goodness(v, lo, hi, good_direction) for v in vals])
    else:
        good = np.full(len(vals), 0.5)
    badness = 1.0 - good
    weights = badness + 1e-6  # never a zero-probability pick
    weights /= weights.sum()
    if rng is None:
        rng = np.random.default_rng()
    k = min(k, len(scored))
    idx = rng.choice(len(scored), size=k, replace=False, p=weights)
    return [scored[i] for i in idx]


def _save_example_row(examples, n_per_feature, attr, title, out_path,
                      status_line, rel_range, reported_of=None):
    """Render one row of example thumbnails to ``out_path``.

    Each thumbnail gets the feature overlay, a frame whose border colour is
    the *viridis* colour of the relative score (goodness over the min/max of
    all samples, exactly like the ``data_set_overview`` relative plots), and a
    two-line label: ``<status>`` on top, ``#<id> | QS: <reported value>``
    below. Empty slots become placeholder tiles.
    """
    if reported_of is None:
        reported_of = lambda v: _report_value(attr, v)
    n_avail = len(examples)

    fig, axes = plt.subplots(1, n_per_feature, figsize=(3.4 * n_per_feature, 3.8))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for col in range(n_per_feature):
        ax = axes[col]
        ax.axis("off")
        if col >= n_avail:
            ax.imshow(np.full((64, 64, 3), 240, dtype=np.uint8))
            ax.set_title("no filtered\nframe", fontsize=8, color="gray")
            continue
        obs, value = examples[col]
        overlay = _feature_overlay(obs, attr)
        reported = reported_of(value)
        status = status_line(obs, reported)
        if overlay is None:
            ax.imshow(np.full((64, 64, 3), 210, dtype=np.uint8))
            ax.set_title(f"{status}\n#{obs.id}", fontsize=8)
            continue
        ax.imshow(overlay)

        # border colour = viridis relative score over the min/max of all
        # samples, matching the data_set_overview relative colourbar
        if rel_range is not None:
            lo, hi = rel_range
            color = REL_CMAP(_goodness(reported, lo, hi, 1))
        else:
            color = REL_CMAP(0.5)
        ax.add_patch(Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            fill=False, edgecolor=color, linewidth=4,
        ))
        ax.set_title(f"{status}\n#{obs.id} | QS: {_format_tick(reported)}", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _feature_rel_range(accepted, rejected, attr):
    """(lo, hi) of the reported values over all samples, or None when empty."""
    vals = list(_feature_values(accepted, attr)) + list(_feature_values(rejected, attr))
    vals = np.array([_report_value(attr, v) for v in vals])
    vals = vals[~np.isnan(vals)]
    if vals.size:
        return float(vals.min()), float(vals.max())
    return None


def plot_bad_examples(accepted, rejected, selected, output_dir, n_per_feature=BAD_EXAMPLES_PER_FEATURE):
    """Per-feature example-frame images, split by pipeline stage.

    ``bad_examples/pre-filter_stage/`` — raw pre-filter stats:
      ``<attr>_filtered.png``      up to 5 frames actually rejected *for that
                                   feature's reason* (worst-first, curated to
                                   stay visually distinct). Only produced when
                                   the reason fired at least once.
      ``lower_<attr>_quality.png`` when the reason never fired: the
                                   lowest-quality *accepted* frames per that
                                   stat, probability-sampled (worst = highest
                                   likelihood). Vincent soft stats that can
                                   never reject always take this form.
    ``bad_examples/selection_stage/`` — quality scores:
      ``lower_<attr>_quality.png`` lowest-quality accepted frames per quality
                                   score, probability-sampled.

    Every thumbnail border is coloured with the viridis relative score (same
    min/max-of-all-samples scale as the ``data_set_overview`` relative plots).
    """
    output_dir = Path(output_dir)
    rng = np.random.default_rng(0)

    pre_dir = output_dir / "bad_examples" / "pre-filter_stage"
    sel_dir = output_dir / "bad_examples" / "selection_stage"
    pre_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    for attr, label, good_direction in RAW_FEATURES:
        rel_range = _feature_rel_range(accepted, rejected, attr)
        reasons = _FEATURE_REASONS.get(attr, ())
        reason_pool = [o for o in rejected if o.rejection_reason in reasons] if reasons else []
        if reason_pool:
            examples = _curated_bad_examples(reason_pool, attr, good_direction, n_per_feature)
            if examples:
                _save_example_row(
                    examples, n_per_feature, attr,
                    f"Filtered-out examples: {label}",
                    pre_dir / f"{attr}_filtered.png",
                    status_line=lambda o, r: f"rejected - {o.rejection_reason or 'rejected'}",
                    rel_range=rel_range,
                )
        else:
            examples = _prob_sample_low_quality(
                accepted, attr, good_direction, n_per_feature, rng=rng)
            if examples:
                _save_example_row(
                    examples, n_per_feature, attr,
                    f"Lowest-quality accepted frames: {label}",
                    pre_dir / f"lower_{attr}_quality.png",
                    status_line=lambda o, r, lbl=label: f"accepted - {lbl}",
                    rel_range=rel_range,
                )

    for attr, label, good_direction in QUALITY_FEATURES:
        rel_range = _feature_rel_range(accepted, rejected, attr)
        examples = _prob_sample_low_quality(
            accepted, attr, good_direction, n_per_feature, rng=rng)
        if examples:
            _save_example_row(
                examples, n_per_feature, attr,
                f"Lowest-quality accepted frames: {label}",
                sel_dir / f"lower_{attr}_quality.png",
                status_line=lambda o, r: "accepted but not selected",
                rel_range=rel_range,
            )
