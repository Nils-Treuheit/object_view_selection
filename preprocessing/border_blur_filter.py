import cv2
import numpy as np

from .base import BaseFilter
from .vincent_utils import (
    compute_boundary_blur_variance,
    compute_boundary_tenengrad,
    mask_to_foreground,
    robust_center_scale,
)

BORDER_BLUR_STROKE_WIDTH = 9


class BorderLaplacianBlurFilter(BaseFilter):
    """Boundary-band sharpness pre-filter (Laplacian variance).

    Stat: variance of the Laplacian restricted to the band straddling the mask
    contour (``compute_boundary_blur_variance``), stored on ``metrics.laplacian``
    and ``metrics.vincent_boundary_blur_variance``. Higher = sharper object/background
    transition.

    Implements both rejection criteria mandated by ``BaseFilter``: an absolute
    threshold-based garbage floor (``hard_min_variance``) on the raw stat, and a
    population-based extreme-bad-outlier removal (``outlier_z``) using robust
    median/MAD z-scores over the raw stat. Returns a (0, 1] goodness score
    anchored at ``max_variance`` when the observation passes both criteria.
    """

    BORDER_BLUR_STROKE_WIDTH = 9
    MAX_VARIANCE = 10000.0
    DEFAULT_HARD_MIN_VARIANCE = 100.0

    stat_attr = "laplacian"
    weight_attr = "vincent_boundary_blur_variance"
    softness = 0.3
    direction = "low_bad"
    reason = "blur_laplacian"

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_variance: float = MAX_VARIANCE,
        hard_min_variance: float = DEFAULT_HARD_MIN_VARIANCE,
        outlier_z: float | None = None,
        threshold_min: float | None = None,
        enabled=True,
    ):

        super().__init__(enabled)

        self.stroke_width = stroke_width
        self.max_variance = max_variance
        self.hard_min_variance = hard_min_variance
        self.outlier_z = outlier_z
        self.threshold_min = threshold_min
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of the raw boundary-blur stat.

        Skipped unless ``outlier_z`` is set.
        """
        if self.outlier_z is None:
            return
        values = []
        for obs in observations:
            if not self.enabled:
                continue
            values.append(float(self.compute_stat(obs)))
        if values:
            median, robust_scale = robust_center_scale(np.array(values, dtype=float))
            if robust_scale <= 0:
                robust_scale = 1.0
            self._robust = (median, robust_scale)

    def evaluate(self, observation):
        """Compute the raw boundary-blur stat and apply both rejection criteria.

        The score is the stat scaled against a fixed global anchor
        (``max_variance``, default 10000) so it is comparable across datasets
        and doubles as the pre-filter goodness score.
        """
        if not self.enabled:
            return -1.0, True, ""

        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.stat_attr, stat)
        setattr(observation.metrics, self.weight_attr, stat)

        score = min(stat / self.max_variance, 1.0)

        # Absolute threshold-based garbage rejection: a boundary smeared below
        # hard_min_variance is unusable regardless of the population (motion blur).
        if self.hard_min_variance > 0.0 and stat < self.hard_min_variance:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based outlier rejection: robust median/MAD z-score of the
        # raw stat — the "low_bad" (blurred) tail at z <= -outlier_z is rejected
        # as a noticeably bad outlier.
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        return score, True, self.reason

    def compute_stat(self, observation) -> float:
        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)


class BorderTenengradBlurFilter(BaseFilter):
    """Boundary-band sharpness pre-filter (Tenengrad).

    Stat: mean Sobel magnitude restricted to the boundary band
    (``compute_boundary_tenengrad``), stored on ``metrics.tenengrad``.
    Higher = sharper structured gradients at the object contour.

    Companion to ``BorderLaplacianBlurFilter`` implementing both rejection
    criteria mandated by ``BaseFilter``: an absolute threshold-based garbage
    floor (``hard_min_tenengrad``) on the raw stat, and a population-based
    extreme-bad-outlier removal (``outlier_z``) using robust median/MAD z-scores.
    Returns a (0, 1] goodness score anchored at ``max_tenengrad`` when passed.
    """

    BORDER_BLUR_STROKE_WIDTH = 9
    MAX_TENEGRAD = 100.0
    DEFAULT_HARD_MIN_TENEGRAD = 25.0

    stat_attr = "tenengrad"
    weight_attr = "bound_tenengrad"
    softness = 0.3
    direction = "low_bad"
    reason = "blur_tenengrad"

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_tenengrad: float = MAX_TENEGRAD,
        hard_min_tenengrad: float = DEFAULT_HARD_MIN_TENEGRAD,
        outlier_z: float | None = None,
        threshold_min: float | None = None,
        enabled=True,
    ):

        super().__init__(enabled)

        self.stroke_width = stroke_width
        self.max_tenengrad = max_tenengrad
        self.hard_min_tenengrad = hard_min_tenengrad
        self.outlier_z = outlier_z
        self.threshold_min = threshold_min
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of the raw Tenengrad stat.

        Skipped unless ``outlier_z`` is set.
        """
        if self.outlier_z is None:
            return
        values = []
        for obs in observations:
            if not self.enabled:
                continue
            values.append(float(self.compute_stat(obs)))
        if values:
            median, robust_scale = robust_center_scale(np.array(values, dtype=float))
            if robust_scale <= 0:
                robust_scale = 1.0
            self._robust = (median, robust_scale)

    def evaluate(self, observation):
        """Compute the raw Tenengrad stat and apply both rejection criteria.

        The score is the stat scaled against a fixed global anchor
        (``max_tenengrad``, default 100) so it is comparable across datasets
        and doubles as the pre-filter goodness score.
        """
        if not self.enabled:
            return -1.0, True, ""

        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.stat_attr, stat)
        setattr(observation.metrics, self.weight_attr, stat)

        score = min(stat / self.max_tenengrad, 1.0)

        # Absolute threshold-based garbage rejection: a boundary with insufficient
        # Tenengrad below hard_min_tenengrad is unusable regardless of the
        # population (motion blur, defocus).
        if self.hard_min_tenengrad > 0.0 and stat < self.hard_min_tenengrad:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based outlier rejection: robust median/MAD z-score of the
        # raw stat — the "low_bad" (blurred) tail at z <= -outlier_z is rejected
        # as a noticeably bad outlier.
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        return score, True, self.reason

    def compute_stat(self, observation) -> float:
        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return compute_boundary_tenengrad(gray, foreground, self.stroke_width)
