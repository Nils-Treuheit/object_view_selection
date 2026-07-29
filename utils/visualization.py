from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_overview_grid(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    titles: list[str] | None = None,
    cols: int = 5,
    figsize=(16, 12),
) -> np.ndarray:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n):
        if masks is not None and i < len(masks):
            overlay = images[i].copy().astype(np.float32)
            mask_bool = (masks[i] > 0)
            overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.5 + 200
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            axes[i].imshow(overlay)
        else:
            axes[i].imshow(images[i])
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=8)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return buf


def save_overview_grid(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None,
    save_path: str,
    titles: list[str] | None = None,
):
    grid = create_overview_grid(images, masks, titles)
    cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))