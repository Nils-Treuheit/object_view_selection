"""
Per-feature dataset overview and bad-example plots.

Ported from nit_view_selection/select_best_views.py (plot_dataset_overview /
plot_bad_examples) and extended to cover every feature (all pre-filter raw
stats plus all quality component scores), not just the Vincent soft filters.

Output layout (inside the pre-filter plots dir):
  data_set_overview/
    raw_filter_<feature>_fixed.png       # goodness on a fixed coolwarm 0..1 scale
    raw_filter_<feature>_relative.png    # goodness on a data-relative viridis scale
    quality_score_<feature>_fixed.png
    quality_score_<feature>_relative.png
  bad_examples/
    raw_filter_<feature>.png       # up to 5 filtered-out frames per raw stat
    quality_score_<feature>.png    # up to 5 filtered-out frames per quality score

Each data_set_overview image has a distribution histogram on the left (x-axis
fixed to [-0.05, 1.05] when the feature is bounded to [0, 1]) and the feature
value over the frame sequence on the right.

Coloring uses a *persistent* meaning across every plot: 1.0 = always good,
0.0 = always bad. For features where a high value is bad (border ratio, hand
overlap, artifact fraction) the value is inverted so a good frame is always
warm/bright, never cold. The two variants per feature are:

  *_fixed.png     colorbar pinned to the full 0..1 goodness range (coolwarm)
  *_relative.png  colorbar adjusted to this dataset's goodness min..max (viridis)

The bad_examples images only ever show frames that were actually filtered out
(worst-first per feature), with placeholder tiles filling any empty slots.
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

GOOD_FIXED_LABEL = "goodness (0=bad, 1=good)"
GOOD_RELATIVE_LABEL = "goodness (relative scale)"

# Rejection reasons → histogram colors. "border" is the truncation filter
# (object cut off at the frame edge) and "occlusion" is hand / other-object
# coverage; both are kept as distinct categories everywhere.
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
    "occlusion": "tab:pink",
    "motion_blur": "tab:red",
    "completeness": "tab:cyan",
    "incomplete_shape": "tab:cyan",
    "empty_mask": "dimgray",
}
FALLBACK_REJECT_COLOR = "tab:red"

# (metric attr, readable label, good_direction) where good_direction:
#   1  = higher is better
#   -1 = lower is better
RAW_FEATURES = [
    ("laplacian", "Laplacian", 1),
    ("tenengrad", "Tenengrad", 1),
    ("area_ratio", "Area Ratio", 1),
    ("border_ratio", "Border Ratio", -1),
    ("edge_ratio", "Edge Ratio", -1),
    ("hand_overlap", "Hand Overlap", -1),
    ("completeness", "Completeness", 1),
    ("vincent_area_fraction", "Mask Area Fraction", 1),
    ("vincent_artifact_fraction", "Artifact Fraction", -1),
    ("vincent_boundary_blur_variance", "Boundary Blur Variance", 1),
]

QUALITY_FEATURES = [
    ("blur", "Blur Quality", 1),
    ("area", "Area Quality", 1),
    ("occlusion", "Occlusion Quality", 1),
    ("vincents_area", "Vincent Area Quality", 1),
    ("vincents_artefacts", "Vincent Artifacts Quality", 1),
    ("vincents_motion_blur", "Vincent Motion Blur Quality", 1),
    ("confidence", "Confidence", 1),
    ("score", "Final Quality", 1),
]

# 5 example frames per feature in the bad_examples plots
BAD_EXAMPLES_PER_FEATURE = 5

# Fixed histogram x-limits for features bounded to [0, 1]
FIXED_HIST_XLIM = (-0.05, 1.05)

# Overlay dispatch: which raw stat a quality feature maps to for highlighting.
_QUALITY_TO_RAW = {
    "blur": "laplacian",
    "area": "area_ratio",
    "occlusion": "hand_overlap",
    "completeness": "completeness",
    "vincents_area": "vincent_area_fraction",
    "vincents_artefacts": "vincent_artifact_fraction",
    "vincents_motion_blur": "vincent_boundary_blur_variance",
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
                     feature_attr, label, good_direction, cmap, norm, colorbar_label):
    sel_ids = {s.id for s in selected}

    acc_vals = _feature_values(accepted, feature_attr)
    rej_vals = _feature_values(rejected, feature_attr)

    # shared axis range over the combined, non-NaN values
    combined = np.concatenate([acc_vals, rej_vals])
    combined = combined[~np.isnan(combined)]
    if combined.size == 0:
        ax_hist.axis("off")
        ax_scatter.axis("off")
        return
    lo, hi = float(combined.min()), float(combined.max())

    # left: distribution histogram
    reasons = sorted({r for r in (o.rejection_reason for o in rejected) if r})
    if len(rej_vals[~np.isnan(rej_vals)]) > 0:
        ax_hist.hist(rej_vals, bins=40, alpha=0.35, color="#d3d3d3", label="rejected")
    for reason in reasons:
        reason_vals = np.array([
            _feature_value(o, feature_attr) for o in rejected
            if o.rejection_reason == reason
        ])
        reason_vals = reason_vals[~np.isnan(reason_vals)]
        color = REJECT_COLORS.get(reason, FALLBACK_REJECT_COLOR)
        ax_hist.hist(reason_vals, bins=40, alpha=0.6, color=color, label=reason)
    if len(acc_vals[~np.isnan(acc_vals)]) > 0:
        ax_hist.hist(acc_vals, bins=40, alpha=0.6, color="#2a9d8f", label="accepted")
    for s in selected:
        v = _feature_value(s, feature_attr)
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
    vals = np.array([_feature_value(o, feature_attr) for o in all_obs])

    # 1.0 = good, 0.0 = bad regardless of the raw feature's good_direction
    good = np.array([_goodness(v, lo, hi, good_direction) for v in vals])

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
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax_scatter, shrink=0.8, label=colorbar_label,
    )
    ax_scatter.set_xlabel("frame index", fontsize=8)
    ax_scatter.set_ylabel(label, fontsize=8)
    ax_scatter.tick_params(labelsize=6)


def plot_dataset_overview(accepted, rejected, selected, output_dir):
    """Two data_set_overview images per feature (fixed + relative goodness)."""
    output_dir = Path(output_dir)
    for group, features, prefix in [
        ("raw", RAW_FEATURES, "raw_filter_"),
        ("quality", QUALITY_FEATURES, "quality_score_"),
    ]:
        out_dir = output_dir / "data_set_overview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, label, good_direction in features:
            combined = _feature_values(accepted, attr)
            if rejected:
                combined = np.concatenate([combined, _feature_values(rejected, attr)])
            combined = combined[~np.isnan(combined)]
            if combined.size:
                lo, hi = float(combined.min()), float(combined.max())
                goodness = np.array([_goodness(v, lo, hi, good_direction) for v in combined])
                gmin, gmax = float(goodness.min()), float(goodness.max())
            else:
                lo, hi, gmin, gmax = 0.0, 1.0, 0.0, 1.0
            if not (gmax > gmin):
                rel_norm = Normalize(gmin - 0.5, gmin + 0.5)
            else:
                rel_norm = Normalize(gmin, gmax)

            # fixed variant: colorbar pinned to the full 0..1 goodness range
            fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 4.5))
            _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                             attr, label, good_direction,
                             WARM_COLD_CMAP, Normalize(0.0, 1.0), GOOD_FIXED_LABEL)
            fig.suptitle(
                f"{label} — {'pre-filter raw stat' if group == 'raw' else 'quality score'} (fixed 0..1)",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fixed_path = out_dir / f"{prefix}{attr}_fixed.png"
            fig.savefig(fixed_path, dpi=150)
            plt.close(fig)

            # relative variant: colorbar adjusted to this dataset's goodness range
            fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 4.5))
            _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                             attr, label, good_direction,
                             REL_CMAP, rel_norm, GOOD_RELATIVE_LABEL)
            fig.suptitle(
                f"{label} — {'pre-filter raw stat' if group == 'raw' else 'quality score'} "
                f"(relative {gmin:.2f}..{gmax:.2f})",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            rel_path = out_dir / f"{prefix}{attr}_relative.png"
            fig.savefig(rel_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {fixed_path}")
            print(f"  Saved {rel_path}")


# --------------------------------------------------------------------------- #
# Bad examples (up to 5 filtered-out frames per feature)
# --------------------------------------------------------------------------- #


def _worst_rejected(rejected, feature_attr, good_direction, k):
    """Worst-first frames among the ones the pipeline filtered out."""
    scored = [(o, _feature_value(o, feature_attr)) for o in rejected]
    scored = [(o, v) for o, v in scored if not np.isnan(v)]
    if not scored:
        return []
    scored.sort(key=lambda t: t[1] * good_direction)
    return scored[:k]


def plot_bad_examples(accepted, rejected, selected, output_dir, n_per_feature=BAD_EXAMPLES_PER_FEATURE):
    """One bad_examples image per feature: frames that were filtered out.

    Only observations rejected by the pipeline are shown, worst-first for the
    feature. If fewer than n_per_feature were filtered out, the empty slots are
    drawn as placeholder tiles so the layout stays consistent.
    """
    output_dir = Path(output_dir)
    for group, features, prefix in [
        ("raw", RAW_FEATURES, "raw_filter_"),
        ("quality", QUALITY_FEATURES, "quality_score_"),
    ]:
        out_dir = output_dir / "bad_examples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, label, good_direction in features:
            examples = _worst_rejected(rejected, attr, good_direction, n_per_feature)
            n_avail = len(examples)
            if n_avail:
                vals = np.array([v for _, v in examples])
                lo, hi = float(vals.min()), float(vals.max())
            else:
                lo, hi = 0.0, 1.0

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
                reason = obs.rejection_reason or "rejected"
                if overlay is None:
                    ax.imshow(np.full((64, 64, 3), 210, dtype=np.uint8))
                    ax.set_title(f"#{obs.id}\n{reason}", fontsize=8)
                    continue
                ax.imshow(overlay)

                # frame border color tracks goodness: warm = good, cold = bad
                frac = _goodness(value, lo, hi, good_direction)
                color = WARM_COLD_CMAP(frac)
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    fill=False, edgecolor=color, linewidth=4,
                ))
                ax.set_title(f"#{obs.id} {value:.3g}\n{reason}", fontsize=8)

            fig.suptitle(
                f"Filtered-out examples: {label} — "
                f"{'pre-filter raw stat' if group == 'raw' else 'quality score'}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            path = out_dir / f"{prefix}{attr}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")
