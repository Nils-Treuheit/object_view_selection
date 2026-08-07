"""
Shared helpers ported from nit_view_selection/select_best_views.py.

Used by the Vincent pre-filters:
- hard: VincentEmptyMaskFilter, VincentBorderPixelFilter
- soft (population-adapted): VincentsAreaFilter, VincentsArtifactsFilter,
  VincentsMotionBlurFilter
"""

import cv2
import numpy as np


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


# --------------------------------------------------------------------------- #
# Robust population scoring
# --------------------------------------------------------------------------- #


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-derived robust scale (MAD * 1.4826 ~ std-equivalent)."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    robust_scale = float(np.median(np.abs(values - median))) * 1.4826
    return median, robust_scale


def one_sided_weight(
    values: np.ndarray,
    median: float,
    robust_scale: float,
    direction: str,
    softness: float,
) -> np.ndarray:
    """One-sided half-Gaussian decay from the robust center.

    Full weight on the "good" side of the median, smooth falloff on the "bad"
    side. `direction` is "low_bad" (values below median are penalized) or
    "high_bad" (values above median are penalized). `softness` is in
    robust-MADs and controls how quickly the falloff bites.
    """
    values = np.asarray(values, dtype=float)
    if direction == "high_bad":
        deviation = np.maximum(values - median, 0.0)
    elif direction == "low_bad":
        deviation = np.maximum(median - values, 0.0)
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    if robust_scale <= 0:
        return np.where(deviation <= 0, 1.0, 0.0)
    z = deviation / robust_scale
    return np.exp(-0.5 * (z / softness) ** 2)


def fit_robust_scores(
    observations,
    stat_attr: str,
    weight_attr: str,
    direction: str,
    softness: float,
) -> None:
    """Population pass: turn per-observation raw stats into robust (0,1] weights.

    Stores the computed weight on ``observation.metrics.<weight_attr>``.
    """
    values = np.array(
        [getattr(obs.metrics, stat_attr, 0.0) for obs in observations],
        dtype=float,
    )
    median, robust_scale = robust_center_scale(values)
    weights = one_sided_weight(values, median, robust_scale, direction, softness)
    for obs, weight in zip(observations, weights):
        setattr(obs.metrics, weight_attr, float(weight))
