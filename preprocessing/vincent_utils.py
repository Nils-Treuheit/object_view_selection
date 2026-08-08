"""
Shared mask/image helpers for the Vincent pre-filters.

The robust population-scoring helpers previously living here
(``robust_center_scale``, ``one_sided_weight``, ``fit_robust_scores``) moved to
:mod:`preprocessing.filter_utils` — see that module for the shared fit / reject
implementation used by every non-binary pre-filter.  They are re-exported here
for backwards compatibility.
"""

import cv2
import numpy as np

from .filter_utils import (  # noqa: F401  (re-exported for backwards compat)
    fit_robust_scores,
    one_sided_weight,
    robust_center_scale,
)

__all__ = [
    "compute_artifact_mask",
    "compute_boundary_band",
    "compute_boundary_blur_variance",
    "compute_boundary_tenengrad",
    "mask_to_foreground",
    "touches_border_pixels",
    "robust_center_scale",
    "one_sided_weight",
    "fit_robust_scores",
]


# --------------------------------------------------------------------------- #
# Mask stats
# --------------------------------------------------------------------------- #


def compute_artifact_mask(
    foreground: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Pixels where open(mask) and close(mask) disagree: speckles/holes/ragged edges.

    Opening drops small foreground specks/protrusions, closing fills small
    background holes/gaps; where the two disagree is unstable, noisy mask
    boundary.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return (opened ^ closed).astype(bool)


def compute_boundary_band(foreground: np.ndarray, stroke_width: int) -> np.ndarray:
    """Ring straddling the mask contour: dilate(mask) XOR erode(mask)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (stroke_width, stroke_width)
    )
    dilated = cv2.dilate(foreground, kernel)
    eroded = cv2.erode(foreground, kernel)
    return (dilated ^ eroded).astype(bool)


def compute_boundary_blur_variance(
    image_gray: np.ndarray, foreground: np.ndarray, stroke_width: int
) -> float:
    """Variance of the Laplacian restricted to the boundary band.

    Restricting to the band straddling the mask contour, rather than the whole
    image or whole mask, avoids background texture dominating a whole-image
    measure and a low-texture object interior diluting a whole-mask measure:
    the actual sharpness signal lives at the object/background transition.
    """
    band = compute_boundary_band(foreground, stroke_width)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=3)
    values = laplacian[band]
    return float(values.var()) if values.size else 0.0


def compute_boundary_tenengrad(
    image_gray: np.ndarray, foreground: np.ndarray, stroke_width: int
) -> float:
    """Mean Sobel-magnitude (Tenengrad) restricted to the boundary band.

    Companion to ``compute_boundary_blur_variance``: the Laplacian variance
    measures overall boundary sharpness while the boundary Tenengrad responds
    to structured gradients, so the two detect complementary blur modes.
    """
    band = compute_boundary_band(foreground, stroke_width)
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    values = gradient[band]
    return float(np.mean(values)) if values.size else 0.0


def mask_to_foreground(mask: np.ndarray) -> np.ndarray:
    """Normalize a possibly-multichannel mask to a uint8 foreground mask."""
    if mask.ndim == 3:
        mask = mask.any(axis=-1)
    return (mask > 0).astype(np.uint8)


def touches_border_pixels(mask: np.ndarray) -> bool:
    """True if any foreground pixel lies on the first/last row or column."""
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )
