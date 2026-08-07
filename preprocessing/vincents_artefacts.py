import numpy as np

from .base import BaseFilter
from .vincent_utils import compute_artifact_mask, mask_to_foreground

MASK_ARTIFACT_KERNEL_SIZE = 3


class VincentsArtifactsFilter(BaseFilter):
    """Hard mask-artifact pre-filter, ported from score_mask_artifacts in
    nit_view_selection/select_best_views.py.

    Artifact fraction = (open(mask) XOR close(mask)) / mask_pixels. Noisy
    speckle/hole/ragged-edge masks get a high artifact fraction and a low
    goodness score.

    The filter itself never hard-rejects: it always passes and returns a
    (0, 1] goodness score (1.0 at zero artifacts, 0.0 at ``max_fraction``).
    Rejection is layered on top by ``FilterVariant`` with a very relaxed
    absolute floor (``threshold_min``) and a population-relative
    extreme-bad-outlier pass (``outlier_z``).
    """

    def __init__(
        self,
        kernel_size: int = MASK_ARTIFACT_KERNEL_SIZE,
        max_fraction: float = 0.05,
        enabled=True,
    ):

        super().__init__(enabled)

        self.kernel_size = kernel_size
        self.max_fraction = max_fraction

    def evaluate(self, observation):
        if not self.enabled:
            return 1.0, True, ""

        foreground = mask_to_foreground(observation.mask)

        pixel_count = float(np.sum(foreground))

        if pixel_count <= 0:
            fraction = 0.0
        else:
            artifact_pixel_count = float(
                np.sum(compute_artifact_mask(foreground, self.kernel_size))
            )
            fraction = artifact_pixel_count / pixel_count

        observation.metrics.vincent_artifact_fraction = fraction

        score = float(np.clip(1.0 - fraction / self.max_fraction, 0.0, 1.0))
        return score, True, "vincents_artefacts"
