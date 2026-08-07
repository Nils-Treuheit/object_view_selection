import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


# Display labels for the raw filter reasons. The truncation filter ("border")
# and the border-pixel filter ("vincent_border_pixel") both detect the same
# failure mode — the object is cut off / touching the frame edge — so they are
# aggregated into one "truncation" bar. Occlusion (hand or other object
# covering the object) is always shown as its own separate bar so the two
# failure modes never get merged into one bucket. The default blur/artifact
# filters reject either below a relaxed absolute floor (``_threshold``) or as
# extreme bad outliers relative to the population (``_outlier``).
REASON_LABELS = {
    "vincent_empty_mask": "empty mask",
    "empty_mask": "empty mask",
    "vincent_border_pixel": "truncation (object out of frame)",
    "border": "truncation (object out of frame)",
    "truncation": "truncation (object out of frame)",
    "blur_laplacian": "blurred (low boundary sharpness)",
    "blur_laplacian_threshold": "blurred (below minimum sharpness)",
    "blur_laplacian_outlier": "blurred (extreme outlier)",
    "blur_tenengrad": "blurred boundary (low gradient)",
    "blur_tenengrad_threshold": "blurred boundary (below minimum gradient)",
    "blur_tenengrad_outlier": "blurred boundary (extreme outlier)",
    "vincents_artefacts": "mask artifacts",
    "vincents_artefacts_threshold": "mask artifacts (below minimum quality)",
    "vincents_artefacts_outlier": "mask artifacts (extreme outlier)",
    "small_object": "small object (low mask area)",
    "low_confidence": "low confidence",
    "blur": "blurred (low sharpness)",
    "motion_blur": "motion blur (smeared boundary)",
    "occlusion": "occlusion (hand / other object)",
    "incomplete_shape": "incomplete / non-compact mask",
    "completeness": "incomplete / non-compact mask",
    "unknown": "unknown",
}


def plot_rejection_reasons(results_dir, output_dir):
    rej_path = results_dir / "rejected.json"
    if not rej_path.exists():
        return

    with open(rej_path) as f:
        rej_data = json.load(f)

    display = {}
    for r in rej_data:
        reason = r.get("reason", "unknown")
        label = REASON_LABELS.get(reason, reason)
        display[label] = display.get(label, 0) + 1

    if not display:
        return

    # sorted by count, largest first; occlusion and truncation stay distinct
    ordered = sorted(display, key=display.get, reverse=True)
    labels = list(ordered)
    counts = [display[k] for k in ordered]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Rejection Reasons (occlusion and truncation kept separate)")
    for i, v in enumerate(counts):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "rejection_reasons.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'rejection_reasons.png'}")
