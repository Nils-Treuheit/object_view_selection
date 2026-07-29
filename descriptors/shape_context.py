import cv2
import numpy as np


def shape_context_descriptor(mask: np.ndarray, num_points: int = 64, num_bins: tuple = (12, 5)) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros(num_bins[0] * num_bins[1])

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :].astype(np.float64)

    if len(contour) > num_points:
        idx = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        points = contour[idx]
    else:
        points = contour

    centroid = points.mean(axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    angles = np.arctan2(centered[:, 1], centered[:, 0])

    if dists.max() == 0:
        return np.zeros(num_bins[0] * num_bins[1])

    r_bins = np.linspace(0, dists.max(), num_bins[1] + 1)
    r_bins[1:] *= 1.05
    theta_bins = np.linspace(-np.pi, np.pi, num_bins[0] + 1)

    hist = np.zeros((num_bins[1], num_bins[0]), dtype=np.float64)
    for i in range(num_points):
        d = dists[i]
        a = angles[i]
        ri = np.searchsorted(r_bins[1:], d)
        ti = np.searchsorted(theta_bins[1:], a)
        if ri < num_bins[1] and ti < num_bins[0]:
            hist[ri, ti] += 1.0

    hist = hist.flatten()
    hist = hist / (np.sum(hist) + 1e-10)
    return hist