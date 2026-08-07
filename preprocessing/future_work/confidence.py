import numpy as np

from ..base import BaseFilter
from ..vincent_utils import robust_center_scale


class ConfidenceFilter(BaseFilter):
    """Confidence pre-filter that combines a threshold-based garbage floor and
    a population-based severe-outlier rejection layer.

    Unlike the soft-filters this does **not** derive (0, 1] selection weights;
    it is a hard filter. The stat is the per-observation confidence value from
    ``observation.metrics.confidence``, reported as-is and also scaled against
    ``reference_confidence`` so downstream diagnostics have a normalised score.

    Besides the inherited `evaluate` signature this filter implements both of
    the two ``BaseFilter`` rejection criteria:

    1. **Threshold-based — Absolute garbage floor**: if ``hard_min_confidence > 0``
       and the confidence is below it, the observation is rejected outright.
       This catches frames whose object-confidence is unusably low (e.g. false
       negatives, empty detections).

    2. **Population-based — Severe-outlier removal**: robust median/MAD z-score
       on the raw confidence values; observations in the ``low_bad`` tail at
       ``z <= -outlier_z`` are rejected as severely bad outliers. The population
       pass is driven by ``fit()`` when ``outlier_z is not None``.
    """

    stat_attr = "confidence"
    reason = "confidence"
    DEFAULT_REFERENCE_CONFIDENCE = 0.5
    DEFAULT_HARD_MIN_CONFIDENCE = 0.3

    def __init__(
        self,
        hard_min_confidence: float = DEFAULT_HARD_MIN_CONFIDENCE,
        reference_confidence: float = DEFAULT_REFERENCE_CONFIDENCE,
        outlier_z: float | None = None,
        enabled: bool = True,
    ):

        super().__init__(enabled)

        self.hard_min_confidence = hard_min_confidence
        self.reference_confidence = reference_confidence
        self.outlier_z = outlier_z
        # (median, robust_scale) of confidence, fit over the population
        self._robust: tuple[float, float] | None = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of confidence values.

        Skipped unless ``outlier_z`` is set; mirrors ``FilterVariant.fit`` so
        ``evaluate`` can compare each confidence against the distribution.
        """
        if self.outlier_z is None:
            return
        values = []
        for obs in observations:
            if not self.enabled:
                continue
            conf = float(getattr(obs.metrics, self.stat_attr, 0.0))
            values.append(conf)
        if values:
            median, robust_scale = robust_center_scale(np.array(values, dtype=float))
            if robust_scale <= 0:
                robust_scale = 1.0
            self._robust = (median, robust_scale)

    def evaluate(self, observation):
        """Compute the raw confidence stat and apply both rejection criteria.

        The score is ``confidence / reference_confidence`` (clipped to [0, 1])
        so it serves as a normalised goodness indicator for diagnostics while
        the two thresholds handle hard rejection.
        """
        # return default valid when disabled
        if not self.enabled:
            return 1.0, True, ""

        confidence = float(getattr(observation.metrics, self.stat_attr, 0.0))
        observation.metrics.confidence = confidence

        # quality-scaled score vs reference anchor
        score = min(confidence / self.reference_confidence, 1.0) if self.reference_confidence > 0 else 1.0

        # Threshold-based Filter — absolute garbage floor
        # A confidence below hard_min_confidence is unusable regardless of the population
        if self.hard_min_confidence > 0.0 and confidence < self.hard_min_confidence:
            return score, False, f"{self.reason}_threshold"

        # Population-based Filter — severe-outlier rejection
        # robust median/MAD z-score on raw confidence; the "low_bad" tail at
        # z <= -outlier_z is rejected as a severely bad outlier
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (confidence - median) / robust_scale
            if z <= -self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        # Pass
        return score, True, self.reason
