import cv2
import numpy as np

from .base import BaseFilter


class BlurFilter(BaseFilter):

    def __init__(
        self,
        laplacian_threshold=120,
        tenengrad_threshold=35,
        enabled=True,
    ):

        super().__init__(enabled)

        self.laplacian_threshold = laplacian_threshold
        self.tenengrad_threshold = tenengrad_threshold

    def variance_of_laplacian(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

    def tenengrad(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        g = np.sqrt(gx ** 2 + gy ** 2)

        return np.mean(g)

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        lap = self.variance_of_laplacian(observation.image)

        ten = self.tenengrad(observation.image)

        observation.metrics["laplacian"] = lap
        observation.metrics["tenengrad"] = ten

        lap_score = min(lap / self.laplacian_threshold, 1.0)

        ten_score = min(ten / self.tenengrad_threshold, 1.0)

        score = 0.5 * lap_score + 0.5 * ten_score

        passed = (
            lap >= self.laplacian_threshold
            and ten >= self.tenengrad_threshold
        )

        return score, passed, "blur"
