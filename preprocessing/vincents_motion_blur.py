import cv2

from .vincent_utils import (
    compute_boundary_blur_variance,
    mask_to_foreground,
)
from .vincents_base import VincentSoftFilter

# --------------------------------------------------------------------------- #
# True module globals: the default values live here and are passed through the
# filter constructor (they are deliberately not defined as class attributes).
# --------------------------------------------------------------------------- #

BLUR_SCORE_SOFTNESS = 0.3
BLUR_STROKE_WIDTH = 9
BLUR_MAX_VARIANCE = 10000.0


class VincentsMotionBlurFilter(VincentSoftFilter):
    """Soft motion-blur pre-filter, ported from score_motion_blur in
    nit_view_selection/select_best_views.py.

    Stat = variance of the Laplacian restricted to the boundary band
    (dilate XOR erode of the mask), stored on
    ``metrics.vincent_boundary_blur_variance``. Blurred boundaries are
    penalized (direction "low_bad"). Boundary sharpness is a continuous
    spectrum, so the softness is deliberately small (0.3 robust-MADs) to
    discriminate.

    Besides deriving the ``(0, 1]`` selection weight
    (``metrics.vincents_motion_blur``) this filter is also a working
    pre-filter: ``evaluate`` reports a quality-scaled stat score and
    implements both rejection criteria from ``BaseFilter`` — an absolute
    threshold-based garbage floor (``hard_min_variance`` on the raw stat) and
    a population-based extreme-bad-outlier removal (``outlier_z``, fit once
    over the population via robust median/MAD).
    """

    def __init__(
        self,
        softness: float = BLUR_SCORE_SOFTNESS,
        stroke_width: int = BLUR_STROKE_WIDTH,
        hard_min_variance: float = 0.0,
        max_variance: float = BLUR_MAX_VARIANCE,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(
            enabled=enabled,
            hard_min=hard_min_variance if hard_min_variance > 0.0 else None,
            outlier_z=outlier_z,
            stat_attr="vincent_boundary_blur_variance",
            reason="vincents_motion_blur",
            direction="low_bad",
            weight_attr="vincents_motion_blur",
            softness=softness,
        )

        self.stroke_width = stroke_width
        self.max_variance = max_variance

    def compute_stat(self, observation) -> float:
        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)

    def compute_score(self, stat: float) -> float:
        """Sharpness score in (0, 1] anchored at ``max_variance``."""
        return min(stat / self.max_variance, 1.0)
