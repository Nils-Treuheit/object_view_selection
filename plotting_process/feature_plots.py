"""
Per-feature dataset overview and bad-example plots.

Ported from nit_view_selection/select_best_views.py (plot_dataset_overview /
plot_bad_examples) and extended to cover every feature (all pre-filter raw
stats plus all quality component scores), not just the Vincent soft filters.

Output layout (inside the pre-filter plots dir):
  data_set_overview/
    raw_filter_<feature>.png       # one image per raw stat
    quality_score_<feature>.png    # one image per quality component
  bad_examples/
    raw_filter_<feature>.png       # worst accepted frames per raw stat
    quality_score_<feature>.png    # worst accepted frames per quality component

Each data_set_overview image has a distribution histogram on the left (x-axis
fixed to [-0.05, 1.05] when the feature is bounded to [0, 1]) and the feature
value over the frame sequence on the right, colored by matplotlib's coolwarm
(cold/blue = low value, warm/red = high value).
"""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# matplotlib's coolwarm: cold (low values) = blue, warm (high values) = red.
# Used for all feature coloring, so 0.0 = cold, 1.0 = warm.
WARM_COLD_CMAP = plt.cm.coolwarm

REJECT_COLORS = {
    "vincent_empty_mask": "dimgray",
    "vincent_border_pixel": "tab:purple",
    "border": "tab:blue",
    "area": "tab:orange",
    "confidence": "tab:green",
    "blur": "tab:brown",
    "occlusion": "tab:pink",
    "completeness": "tab:cyan",
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


def _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected, feature_attr, label, cmap):
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

    # right: feature value over the sequence, colored by warm_cold
    all_obs = list(rejected) + list(accepted)
    all_obs = sorted(all_obs, key=lambda o: o.id)
    idx = np.arange(len(all_obs))
    vals = np.array([_feature_value(o, feature_attr) for o in all_obs])

    # color by raw value: warm (red) = high, cold (blue) = low
    if hi > lo:
        frac = (vals - lo) / (hi - lo)
    else:
        frac = np.zeros_like(vals)

    valid = ~np.isnan(frac)
    if valid.any():
        ax_scatter.scatter(
            idx[valid], vals[valid], s=16,
            c=frac[valid], cmap=cmap, norm=Normalize(0.0, 1.0),
            zorder=2,
        )
    for o, i, v in zip(all_obs, idx, vals):
        if o.id in sel_ids and not np.isnan(v):
            ax_scatter.scatter(i, v, s=60, color="gold", edgecolor="black", zorder=3)

    fig = ax_scatter.get_figure()
    fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap),
        ax=ax_scatter, shrink=0.8,
    )
    ax_scatter.set_xlabel("frame index", fontsize=8)
    ax_scatter.set_ylabel(label, fontsize=8)
    ax_scatter.tick_params(labelsize=6)


def plot_dataset_overview(accepted, rejected, selected, output_dir):
    """One data_set_overview image per feature (histogram + warm_cold scatter)."""
    output_dir = Path(output_dir)
    for group, features, prefix in [
        ("raw", RAW_FEATURES, "raw_filter_"),
        ("quality", QUALITY_FEATURES, "quality_score_"),
    ]:
        out_dir = output_dir / "data_set_overview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, label, _ in features:
            fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 4.5))
            _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected, attr, label, WARM_COLD_CMAP)
            fig.suptitle(
                f"{label} — {'pre-filter raw stat' if group == 'raw' else 'quality score'}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            path = out_dir / f"{prefix}{attr}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")


# --------------------------------------------------------------------------- #
# Bad examples (5 samples per feature)
# --------------------------------------------------------------------------- #


def _worst_accepted(accepted, feature_attr, direction, k):
    scored = [(o, _feature_value(o, feature_attr)) for o in accepted]
    scored = [(o, v) for o, v in scored if not np.isnan(v)]
    if not scored:
        return []
    scored.sort(key=lambda t: t[1] * direction)
    return scored[:k]


def plot_bad_examples(accepted, rejected, selected, output_dir, n_per_feature=BAD_EXAMPLES_PER_FEATURE):
    """One bad_examples image per feature: its worst accepted frames.

    Each thumbnail overlays the mask (with feature-specific highlighting) and
    is framed by a warm_cold border proportional to its feature value.
    """
    output_dir = Path(output_dir)
    for group, features, prefix in [
        ("raw", RAW_FEATURES, "raw_filter_"),
        ("quality", QUALITY_FEATURES, "quality_score_"),
    ]:
        out_dir = output_dir / "bad_examples"
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, label, direction in features:
            examples = _worst_accepted(accepted, attr, direction, n_per_feature)
            if not examples:
                continue

            vals = np.array([v for _, v in examples])
            lo, hi = float(vals.min()), float(vals.max())

            fig, axes = plt.subplots(1, n_per_feature, figsize=(3.4 * n_per_feature, 3.8))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for col in range(n_per_feature):
                ax = axes[col]
                ax.axis("off")
                if col >= len(examples):
                    continue
                obs, value = examples[col]
                overlay = _feature_overlay(obs, attr)
                if overlay is None:
                    ax.imshow(np.full((64, 64, 3), 210, dtype=np.uint8))
                    ax.set_title(f"#{obs.id}\nno image", fontsize=8)
                    continue
                ax.imshow(overlay)

                # frame border color tracks the raw value: warm = high, cold = low
                frac = (value - lo) / (hi - lo) if hi > lo else 0.5
                color = WARM_COLD_CMAP(frac)
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    fill=False, edgecolor=color, linewidth=4,
                ))
                ax.set_title(f"#{obs.id} {value:.3g}", fontsize=8)

            fig.suptitle(
                f"Worst accepted examples: {label} — "
                f"{'pre-filter raw stat' if group == 'raw' else 'quality score'}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            path = out_dir / f"{prefix}{attr}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")
