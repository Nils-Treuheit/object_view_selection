import numpy as np

from .vincent_utils import robust_center_scale
from .vincents_base import VincentSoftFilter


class VincentsAreaFilter(VincentSoftFilter):
    """Soft mask-area pre-filter, ported from score_mask_area in
    nit_view_selection/select_best_views.py.

    Mask area tends to be a continuous spectrum rather than a tight cluster
    with rare outliers, so AREA_SCORE_SOFTNESS is small to discriminate at
    all. Small masks are penalized (direction "low_bad").

    Besides deriving the (0, 1] selection weight this filter is also a
    working pre-filter: ``evaluate`` reports a quality-scaled stat score and
    implements both rejection criteria from ``BaseFilter`` — an absolute
    threshold-based garbage floor (``hard_min_area_fraction`` on the raw stat)
    and a population-based extreme-bad-outlier removal (``outlier_z``, fit once
    over the population via robust median/MAD).
    """

    AREA_SCORE_SOFTNESS = 0.3
    MAX_FRACTION = 0.20

    stat_attr = "vincent_area_fraction"
    weight_attr = "vincents_area"
    direction = "low_bad"
    softness = AREA_SCORE_SOFTNESS

    # rejection reason base; threshold/outlier modes append _threshold/_outlier
    reason = "vincents_area"

    def __init__(
        self,
        softness: float = AREA_SCORE_SOFTNESS,
        hard_min_area_fraction: float = 0.0,
        max_fraction: float = MAX_FRACTION,
        threshold_min: float | None = None,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness
        self.hard_min_area_fraction = hard_min_area_fraction
        self.max_fraction = max_fraction
        self.threshold_min = threshold_min
        self.outlier_z = outlier_z
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of the raw area fraction stat.

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
        """Compute the raw area fraction stat and apply both rejection criteria.

        The score is the stat scaled against a fixed global anchor
        (``max_fraction``, default 0.20) so it is comparable across datasets
        and doubles as the pre-filter goodness score.
        """
        # return default invalid value, pass=true and no reason when disabled
        if not self.enabled:
            return -1.0, True, ""

        # calculate raw stat and publish it
        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.stat_attr, stat)

        # quality scaled stat: area fraction score in (0, 1] anchored at max_fraction
        score = min(stat / self.max_fraction, 1.0)

        # Threshold-based Filter
        # absolute garbage floor: a mask occupying below hard_min_area_fraction
        # of the canvas is unusable regardless of the population
        if self.hard_min_area_fraction > 0.0 and stat < self.hard_min_area_fraction:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based Filter
        # robust median/MAD z-score of the raw stat: the "low_bad" (tiny mask)
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

        mask = observation.mask > 0

        pixel_count = float(np.sum(mask))

        canvas_area = float(mask.shape[0] * mask.shape[1])

        if canvas_area <= 0:
            return 0.0

        return pixel_count / canvas_area
