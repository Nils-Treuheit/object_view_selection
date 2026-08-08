import numpy as np

from .base import ScoreFilter
from .vincent_utils import compute_artifact_mask, mask_to_foreground

# --------------------------------------------------------------------------- #
# True module globals: the default values live here and are passed through the
# filter constructor (they are deliberately not defined as class attributes).
# --------------------------------------------------------------------------- #

MASK_ARTIFACT_KERNEL_SIZE = 3
DEFAULT_MAX_FRACTION = 0.05
DEFAULT_HARD_MAX_FRACTION = 0.15


class VincentsArtifactsFilter(ScoreFilter):
    """Mask-artifact pre-filter, ported from score_mask_artifacts in
    nit_view_selection/select_best_views.py.

    Artifact fraction = (open(mask) XOR close(mask)) / mask_pixels. Noisy
    speckle/hole/ragged-edge masks get a high artifact fraction and a low
    goodness score.

    Implements both rejection criteria from ``BaseFilter`` via ``ScoreFilter``
    — an absolute threshold-based garbage ceiling (``hard_max_fraction`` on the
    raw stat) and a population-based extreme-bad-outlier removal
    (``outlier_z``, fit once over the population via robust median/MAD).
    """

    def __init__(
        self,
        kernel_size: int = MASK_ARTIFACT_KERNEL_SIZE,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        hard_max_fraction: float = DEFAULT_HARD_MAX_FRACTION,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(
            enabled=enabled,
            hard_max=hard_max_fraction,
            outlier_z=outlier_z,
            stat_attr="vincent_artifact_fraction",
            reason="vincents_artefacts",
            direction="high_bad",
        )

        self.kernel_size = kernel_size
        self.max_fraction = max_fraction

    def compute_stat(self, observation) -> float:
        """Raw artifact fraction for an observation."""
        foreground = mask_to_foreground(observation.mask)
        pixel_count = float(np.sum(foreground))
        if pixel_count <= 0:
            return 0.0
        artifact_pixel_count = float(
            np.sum(compute_artifact_mask(foreground, self.kernel_size))
        )
        return artifact_pixel_count / pixel_count

    def compute_score(self, stat: float) -> float:
        """``1 - fraction / max_fraction``, scaled to [0, 1]."""
        return float(np.clip(1.0 - stat / self.max_fraction, 0.0, 1.0))
