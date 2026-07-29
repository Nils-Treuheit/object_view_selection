import numpy as np


def contour_area(mask: np.ndarray) -> float:
    return float(np.sum(mask > 0))


def bounding_box(mask: np.ndarray) -> tuple:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def mask_centroid(mask: np.ndarray) -> tuple:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))