import cv2
import numpy as np


def hu_moments(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log(np.abs(hu) + 1e-10)
    return hu