import cv2
import numpy as np

from ..vincent_utils import (
    compute_boundary_blur_variance,
    mask_to_foreground,
    robust_center_scale,
)
from ..vincents_base import VincentSoftFilter


class VincentsMotionBlurFilter(VincentSoftFilter):
    """Soft motion-blur pre-filter, ported from score_motion_blur in
    nit_view_selection/select_best_views.py.

    Stat = variance of the Laplacian restricted to the boundary band
    (dilate XOR erode of the mask). Blurred boundaries are penalized
    (direction "low_bad"). Boundary sharpness is a continuous spectrum, so
    the softness is deliberately small (0.3 robust-MADs) to discriminate.

    Besides deriving the (0, 1] selection weight this filter is also a
    working pre-filter: ``evaluate`` reports a quality-scaled stat score and
    implements both rejection criteria from ``BaseFilter`` — an absolute
    threshold-based garbage floor (``hard_min_variance`` on the raw stat) and
    a population-based extreme-bad-outlier removal (``outlier_z``, fit once
    over the population via robust median/MAD).
    """

    BLUR_SCORE_SOFTNESS = 0.3
    BLUR_STROKE_WIDTH = 9
    BLUR_MAX_VARIANCE = 10000.0

    stat_attr = "vincent_boundary_blur_variance"
    weight_attr = "vincents_motion_blur"
    direction = "low_bad"
    softness = BLUR_SCORE_SOFTNESS

    # rejection reason base; threshold/outlier modes append _threshold/_outlier
    reason = "vincents_motion_blur"

    def __init__(
        self,
        softness: float = BLUR_SCORE_SOFTNESS,
        stroke_width: int = BLUR_STROKE_WIDTH,
        hard_min_variance: float = 0.0,
        max_variance: float = BLUR_MAX_VARIANCE,
        threshold_min: float | None = None,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness
        self.stroke_width = stroke_width
        self.hard_min_variance = hard_min_variance
        self.max_variance = max_variance
        self.threshold_min = threshold_min
        self.outlier_z = outlier_z
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of the raw boundary-blur stat.

        Mirrors ``FilterVariant.fit``: the robust statistics are fit over the
        population before the per-observation pass so ``evaluate`` can compare
        each stat against the distribution. Skipped unless ``outlier_z`` is set.
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
        # return default invalid value, pass=true and no reason when disabled
        if not self.enabled:
            return -1.0, True, ""

        # calculate raw stat and publish it
        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.stat_attr, stat)

        # quality scaled stat: sharpness score in (0, 1] anchored at max_variance
        score = min(stat / self.max_variance, 1.0)

        # Threshold-based Filter
        # absolute garbage floor: a boundary smeared below hard_min_variance is
        # unusable regardless of the population (motion blur)
        if self.hard_min_variance > 0.0 and stat < self.hard_min_variance:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based Filter
        # robust median/MAD z-score of the raw stat: the "low_bad" (blurred)
        # tail at z <= -outlier_z is rejected as a noticeably bad outlier
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        # Pass Filter Criteria
        # report the quality scaled stat score
        return score, True, self.reason

    def compute_stat(self, observation) -> float:

        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)
