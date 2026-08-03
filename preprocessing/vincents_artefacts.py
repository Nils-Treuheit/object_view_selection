import numpy as np

from .vincent_utils import compute_artifact_mask, mask_to_foreground
from .vincents_base import VincentSoftFilter


class VincentsArtifactsFilter(VincentSoftFilter):
    """Soft mask-artifact pre-filter, ported from score_mask_artifacts in
    nit_view_selection/select_best_views.py.

    Artifact fraction = (open(mask) XOR close(mask)) / mask_pixels. Noisy
    speckle/hole/ragged-edge masks are penalized (direction "high_bad").
    Artifact fraction clusters tightly with rare outliers, so the softness is
    deliberately large (3.0 robust-MADs).
    """

    ARTIFACT_SCORE_SOFTNESS = 3.0
    MASK_ARTIFACT_KERNEL_SIZE = 3

    stat_attr = "vincent_artifact_fraction"
    weight_attr = "vincents_artefacts"
    direction = "high_bad"
    softness = ARTIFACT_SCORE_SOFTNESS

    def __init__(
        self,
        softness: float = ARTIFACT_SCORE_SOFTNESS,
        kernel_size: int = MASK_ARTIFACT_KERNEL_SIZE,
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness
        self.kernel_size = kernel_size

    def compute_stat(self, observation) -> float:

        foreground = mask_to_foreground(observation.mask)

        pixel_count = float(np.sum(foreground))

        if pixel_count <= 0:
            return 0.0

        artifact_pixel_count = float(
            np.sum(compute_artifact_mask(foreground, self.kernel_size))
        )

        return artifact_pixel_count / pixel_count
