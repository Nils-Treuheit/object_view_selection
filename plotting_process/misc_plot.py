import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_rejection_reasons(results_dir, output_dir):
    rej_path = results_dir / "rejected.json"
    if not rej_path.exists():
        return

    with open(rej_path) as f:
        rej_data = json.load(f)

    reasons = {}
    for r in rej_data:
        reason = r.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    if not reasons:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(reasons.keys())
    counts = list(reasons.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Rejection Reasons")
    for i, v in enumerate(counts):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "rejection_reasons.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'rejection_reasons.png'}")
