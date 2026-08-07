import cv2

from .base import BaseFilter
from .vincent_utils import (
    compute_boundary_blur_variance,
    compute_boundary_tenengrad,
    mask_to_foreground,
)

BORDER_BLUR_STROKE_WIDTH = 9


class BorderLaplacianBlurFilter(BaseFilter):
    """Hard boundary-band sharpness pre-filter (Laplacian variance).

    Stat: variance of the Laplacian restricted to the band straddling the
    mask contour (``compute_boundary_blur_variance``), stored on
    ``metrics.laplacian``. Higher = sharper object/background transition.

    The filter itself never hard-rejects: it always passes and returns a
    (0, 1] goodness score anchored at ``max_variance``. Rejection is layered
    on top by ``FilterVariant`` with a very relaxed absolute floor
    (``threshold_min``) and a population-relative extreme-bad-outlier pass
    (``outlier_z``), so only awful-quality samples are dropped.
    """

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_variance: float = 10000.0,
        enabled=True,
    ):

        super().__init__(enabled)

        self.stroke_width = stroke_width
        self.max_variance = max_variance

    def evaluate(self, observation):
        if not self.enabled:
            return 1.0, True, ""

        if observation.image is None:
            return 1.0, True, ""

        foreground = mask_to_foreground(observation.mask)
        gray = cv2.cvtColor(observation.image, cv2.COLOR_RGB2GRAY)
        variance = compute_boundary_blur_variance(gray, foreground, self.stroke_width)

        observation.metrics.laplacian = variance
        observation.metrics.vincent_boundary_blur_variance = variance

        score = min(variance / self.max_variance, 1.0)
        return score, True, "blur_laplacian"


class BorderTenengradBlurFilter(BaseFilter):
    """Hard boundary-band sharpness pre-filter (Tenengrad).

    Stat: mean Sobel magnitude restricted to the boundary band
    (``compute_boundary_tenengrad``), stored on ``metrics.tenengrad``.
    Higher = sharper structured gradients at the object contour.

    Like ``BorderLaplacianBlurFilter`` it always passes and returns a (0, 1]
    goodness score anchored at ``max_tenengrad``; ``FilterVariant`` layers
    the relaxed-floor / extreme-outlier rejection.
    """

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_tenengrad: float = 100.0,
        enabled=True,
    ):

        super().__init__(enabled)

        self.stroke_width = stroke_width
        self.max_tenengrad = max_tenengrad

    def evaluate(self, observation):
        if not self.enabled:
            return 1.0, True, ""

        if observation.image is None:
            return 1.0, True, ""

        foreground = mask_to_foreground(observation.mask)
        gray = cv2.cvtColor(observation.image, cv2.COLOR_RGB2GRAY)
        tenengrad = compute_boundary_tenengrad(gray, foreground, self.stroke_width)

        observation.metrics.tenengrad = tenengrad

        score = min(tenengrad / self.max_tenengrad, 1.0)
        return score, True, "blur_tenengrad"
