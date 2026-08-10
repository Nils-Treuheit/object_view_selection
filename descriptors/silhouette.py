"""Silhouette shape descriptor for view-diversity scoring.

A silhouette descriptor is a fixed-size, position-invariant encoding of the
object's outline (the binary mask downsampled onto a square grid). Two views
of the same object from different angles produce different silhouettes, so a
distance between silhouette descriptors is a cheap proxy for how visually
different two views are - independent of the semantic embedding space.

The descriptor normalises out translation (bounding-box crop), scale
(pad-to-square + resize) and intensity, so the divergence between two
silhouettes lives in a comparable [0, 1] cosine scale alongside the
embedding-space cosine distance used by the GQD selector.
"""

import cv2
import numpy as np


def silhouette_descriptor(mask: np.ndarray, size: int = 64) -> np.ndarray:
    """Return the L2-normalised ``size x size`` silhouette vector of ``mask``.

    The mask is binarised, cropped to its bounding box, padded to a square,
    resized onto the fixed grid and flattened. Empty masks yield the zero
    vector (distance to everything = 0).
    """
    mask = np.asarray(mask)
    binary = (mask > 0).astype(np.uint8) * 255

    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return np.zeros(size * size, dtype=np.float32)

    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = crop.shape
    side = max(hh, ww)
    padded = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - hh) // 2
    x0 = (side - ww) // 2
    padded[y0:y0 + hh, x0:x0 + ww] = crop

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float32).ravel() / 255.0

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def silhouette_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two silhouette descriptors ([0, 1])."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)
