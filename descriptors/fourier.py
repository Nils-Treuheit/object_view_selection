import cv2
import numpy as np


def fourier_descriptors(mask: np.ndarray, num_descriptors: int = 32) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros(num_descriptors)

    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :].astype(np.float64)
    centroid = contour.mean(axis=0)
    contour = contour - centroid

    complex_pts = contour[:, 0] + 1j * contour[:, 1]
    fourier = np.fft.fft(complex_pts)
    fourier = np.abs(fourier)
    fourier = fourier[1:num_descriptors + 1]

    if len(fourier) < num_descriptors:
        fourier = np.pad(fourier, (0, num_descriptors - len(fourier)))

    total = np.sum(fourier)
    fourier = fourier / (total + 1e-10)
    return fourier