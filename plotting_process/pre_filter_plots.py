from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# (metric attr, readable label, binary flag)
RAW_STATS = [
    ("laplacian", "Laplacian", False),
    ("tenengrad", "Tenengrad", False),
    ("area_ratio", "Area Ratio", False),
    ("border_ratio", "Border Ratio", False),
    ("edge_ratio", "Edge Ratio", False),
    ("hand_overlap", "Hand Overlap", False),
    ("completeness", "Completeness", False),
    ("vincent_pixel_count", "Mask Pixel Count", False),
    ("vincent_touches_border", "Touches Border (hard)", True),
    ("vincent_area_fraction", "Mask Area Fraction", False),
    ("vincent_artifact_fraction", "Artifact Fraction", False),
    ("vincent_boundary_blur_variance", "Boundary Blur Variance", False),
]

WEIGHTS = [
    ("vincents_area", "Mask Area Weight"),
    ("vincents_artefacts", "Artifacts Weight"),
    ("vincents_motion_blur", "Motion Blur Weight"),
]


def _metric_values(observations, attr):
    vals = []
    for obs in observations:
        v = getattr(obs.metrics, attr, None)
        if v is not None:
            vals.append(float(v))
    return np.array(vals)


def _hist_ax(ax, accepted_vals, rejected_vals, label, binary=False):
    if len(rejected_vals) > 0:
        ax.hist(rejected_vals, bins=30, alpha=0.6, color="#e9c46a",
                label="rejected", histtype="stepfilled")
    if len(accepted_vals) > 0:
        ax.hist(accepted_vals, bins=30, alpha=0.6, color="#2a9d8f",
                label="accepted", histtype="stepfilled")
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)
    if binary:
        ax.set_xticks([0, 1])
    if len(rejected_vals) > 0 or len(accepted_vals) > 0:
        ax.legend(fontsize=6)


def plot_pre_filter_distributions(accepted, rejected, output_dir_pre):
    """Histograms of every pre-filter element: raw stats (accepted vs
    rejected) and population-adapted soft weights (accepted)."""
    output_dir_pre = Path(output_dir_pre)

    n = len(RAW_STATS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (attr, label, binary) in zip(axes, RAW_STATS):
        _hist_ax(ax, _metric_values(accepted, attr),
                 _metric_values(rejected, attr), label, binary)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Pre-Filter Raw Stat Distributions (accepted vs rejected)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir_pre / "pre_filter_raw_stats.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    nw = len(WEIGHTS)
    fig, axes = plt.subplots(1, nw, figsize=(4 * nw, 3))
    if nw == 1:
        axes = [axes]
    for ax, (attr, label) in zip(axes, WEIGHTS):
        vals = _metric_values(accepted, attr)
        if len(vals) > 0:
            ax.hist(vals, bins=30, color="#4ecdc4")
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.tick_params(labelsize=7)
    fig.suptitle("Soft Pre-Filter Population Weights (accepted)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = output_dir_pre / "pre_filter_soft_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
