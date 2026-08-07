import cv2

from .base import QualityMetric

from preprocessing.vincent_utils import (
    compute_boundary_blur_variance,
    mask_to_foreground,
)

DEFAULT_STROKE_WIDTH = 9


class BorderBlurQuality(QualityMetric):
    """Boundary-band sharpness quality from the raw boundary Laplacian variance.

    Uses ``metrics.laplacian`` when the pre-filter already computed it
    (``BorderLaplacianBlurFilter``), otherwise computes the same boundary-band
    variance from the image + mask directly so the score is self-contained.
    Anchored at a fixed global max variance so it is comparable across
    datasets.
    """

    name = "blur"

    def __init__(self, max_variance=10000.0, stroke_width=DEFAULT_STROKE_WIDTH):
        self.max_variance = max_variance
        self.stroke_width = stroke_width

    def compute(self, observation):
        variance = getattr(observation.metrics, "laplacian", 0.0)
        if variance <= 0.0:
            if observation.image is None or observation.mask is None:
                return 0.0
            foreground = mask_to_foreground(observation.mask)
            gray = cv2.cvtColor(observation.image, cv2.COLOR_RGB2GRAY)
            variance = compute_boundary_blur_variance(
                gray, foreground, self.stroke_width
            )
        return min(variance / self.max_variance, 1.0)
