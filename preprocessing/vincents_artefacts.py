import numpy as np

from .base import BaseFilter
from .vincent_utils import compute_artifact_mask, mask_to_foreground, robust_center_scale

MASK_ARTIFACT_KERNEL_SIZE = 3


class VincentsArtifactsFilter(BaseFilter):
    """Mask-artifact pre-filter, ported from score_mask_artifacts in
    nit_view_selection/select_best_views.py.

    Artifact fraction = (open(mask) XOR close(mask)) / mask_pixels. Noisy
    speckle/hole/ragged-edge masks get a high artifact fraction and a low
    goodness score.

    Implements both rejection criteria from ``BaseFilter`` — an absolute
    threshold-based garbage ceiling (``hard_max_fraction`` on the raw stat)
    and a population-based extreme-bad-outlier removal (``outlier_z``, fit once
    over the population via robust median/MAD).
    """

    STAT_ATTR = "vincent_artifact_fraction"
    weight_attr = "vincents_artefacts"
    direction = "high_bad"
    softness = 0.3

    reason = "vincents_artefacts"

    def __init__(
        self,
        kernel_size: int = MASK_ARTIFACT_KERNEL_SIZE,
        max_fraction: float = 0.05,
        hard_max_fraction: float = 0.15,
        threshold_min: float | None = None,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(enabled)

        self.kernel_size = kernel_size
        self.max_fraction = max_fraction
        self.hard_max_fraction = hard_max_fraction
        self.threshold_min = threshold_min
        self.outlier_z = outlier_z
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust median/MAD of the raw artifact fraction.

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

    def compute_stat(self, observation) -> float:
        """Compute the raw artifact fraction for an observation."""
        foreground = mask_to_foreground(observation.mask)
        pixel_count = float(np.sum(foreground))
        if pixel_count <= 0:
            return 0.0
        artifact_pixel_count = float(
            np.sum(compute_artifact_mask(foreground, self.kernel_size))
        )
        return artifact_pixel_count / pixel_count

    def evaluate(self, observation):
        """Compute the raw artifact fraction and apply both rejection criteria.

        The score is ``1 - fraction / max_fraction``, scaled to [0, 1].
        """
        if not self.enabled:
            return 1.0, True, ""

        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.STAT_ATTR, stat)

        # Score — penalised by artifact fraction (higher = worse)
        score = float(np.clip(1.0 - stat / self.max_fraction, 0.0, 1.0))

        # Threshold-based Filter
        # absolute garbage ceiling: a fraction above hard_max_fraction is unusable
        if self.hard_max_fraction > 0.0 and stat >= self.hard_max_fraction:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based Filter
        # robust median/MAD z-score of the raw stat: the "high_bad" (artifacts)
        # tail at z >= outlier_z is rejected as a noticeably bad outlier
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (stat - median) / robust_scale
            if z >= self.outlier_z:
                return score, False, f"{self.reason}_outlier"

        # Pass Filter Criteria
        return score, True, self.reason
