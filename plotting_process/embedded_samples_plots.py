"""
Debug plots comparing the original frames against the embedding model's input.

Each figure is a 4x4 image matrix: one row per randomly sampled observation,
with the original frame, the original + mask overlay, the embedding model's
actual 224x224 input (``contrast_input``: grown-mask cut-out of the object on
the static maximum-contrast background) and that input with the original (not
grown) mask overlaid.

The static background (black for bright-border objects, white for dark-border
objects) is decided once over the whole pool with
``compute_contrast_background``; pass ``background`` to reuse the value the
embedding model was set to.

Output lands in ``<output_dir>/embedded_samples/samples_<n>.png`` (a sibling
of ``plots/`` and ``bad_examples/``).
"""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from embeddings.crop import (
    compute_contrast_background,
    contrast_input,
    contrast_mask,
)

GROW_PX = 5


def _get_image_mask(obs):
    """Return (image_rgb, mask_u8) loading from disk when not already loaded."""
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
    return image, mask


def _mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy().astype(np.float32)
    foreground = mask > 0
    if foreground.any():
        overlay[foreground] = (
            (1 - alpha) * overlay[foreground] + alpha * np.array(color)
        ).astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_embedded_samples(pool_obs, output_dir, n_figures=3, n_examples=4, random_state=0,
                          background=None):
    """Write ``embedded_samples/samples_<n>.png`` debug figures.

    ``pool_obs`` are the selection-pool observations (must carry image/mask or
    loadable ``image_path`` / ``mask_path``). Samples are drawn at random with
    a fixed seed so the output is reproducible. ``background`` (0 or 255) is
    the static contrast background; when None it is computed from ``pool_obs``
    itself.
    """
    pool = [o for o in pool_obs]
    if not pool:
        print("  embedded_samples: empty pool, skipping")
        return None

    if background is None:
        background = compute_contrast_background(pool)
    bg_name = "black" if background == 0 else "white"

    out_dir = Path(output_dir) / "embedded_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_state)

    written = []
    for figure_no in range(n_figures):
        if len(pool) <= n_examples:
            chosen = list(pool)
        else:
            chosen = [pool[i] for i in rng.choice(len(pool), n_examples, replace=False)]

        fig, axes = plt.subplots(n_examples, 4, figsize=(16, 4 * n_examples))
        if n_examples == 1:
            axes = axes[None, :]

        for r, obs in enumerate(chosen):
            image, mask = _get_image_mask(obs)
            if image is None:
                for c in range(4):
                    axes[r, c].axis("off")
                axes[r, 0].set_title(f"frame {obs.id} (unavailable)", fontsize=9)
                continue

            model_input = contrast_input(image, mask, background, grow=GROW_PX, size=224)
            square_m = contrast_mask(mask, grow=GROW_PX, size=224) if mask is not None else None

            cells = [
                (image, None, "original"),
                (image, mask, "original + mask"),
                (model_input, None, "embedding input 224x224"),
                (model_input, square_m, "input + original mask"),
            ]
            for c, (img, m, label) in enumerate(cells):
                ax = axes[r, c]
                if m is not None:
                    img = _mask_overlay(img, m)
                ax.imshow(img)
                ax.set_title(f"{label}\nframe {obs.id}", fontsize=9)
                ax.axis("off")

        fig.suptitle(f"Embedding model input vs original frame — static {bg_name} background, "
                     f"random sample {figure_no + 1}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / f"samples_{figure_no + 1:02d}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  Saved {path}")
        written.append(path)

    return written
