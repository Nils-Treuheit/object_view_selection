import numpy as np

from .base import BaseFilter
from .vincent_utils import robust_center_scale


class AreaFilter(BaseFilter):
    """Area pre-filter with both absolute garbage rejection and population-based outlier removal.

    Implements the two required rejection criteria from ``BaseFilter``:
    - **Threshold-based — Absolute Garbage Rejection**: rejects masks below
      ``hard_min_area_fraction`` regardless of population (truncated/empty frames).
    - **Population-based — Severe Outlier Rejection**: robust median/MAD z-score
      removes noticeably bad small-mask outliers in the "low_bad" tail.
    """

    DEFAULT_MAX_FRACTION = 0.20
    reason = "small_object"

    def __init__(
        self,
        hard_min_area_fraction: float = 0.0,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        outlier_z: float | None = None,
        enabled: bool = True,
    ):

        super().__init__(enabled)

        self.hard_min_area_fraction = hard_min_area_fraction
        self.max_fraction = max_fraction
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
            stat = getattr(obs.metrics, "area_fraction", 0.0)
            values.append(float(stat))
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
        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0
        pixel_count = float(np.sum(mask))
        canvas_area = float(mask.shape[0] * mask.shape[1])

        if canvas_area <= 0:
            stat = 0.0
        else:
            stat = pixel_count / canvas_area

        observation.metrics.area_fraction = stat

        # quality scaled stat: area fraction score in (0, 1] anchored at max_fraction
        score = min(stat / self.max_fraction, 1.0) if self.max_fraction > 0 else (1.0 if stat > 0 else 0.0)

        # Threshold-based Filter — absolute garbage floor
        # A mask occupying below hard_min_area_fraction of the canvas is unusable
        # regardless of the population (e.g., empty frames, truncated objects).
        if self.hard_min_area_fraction > 0.0 and stat < self.hard_min_area_fraction:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based Filter — severe outlier rejection
        # Robust median/MAD z-score of the raw area fraction: the "low_bad" (tiny mask)
        # tail at z <= -outlier_z is rejected as a severely bad outlier.
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        # Pass — report the quality scaled stat score
        return score, True, ""
