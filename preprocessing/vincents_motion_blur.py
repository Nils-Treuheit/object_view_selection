import cv2
import numpy as np

from .vincent_utils import compute_boundary_blur_variance, mask_to_foreground
from .vincents_base import VincentSoftFilter


class VincentsMotionBlurFilter(VincentSoftFilter):
    """Soft motion-blur pre-filter, ported from score_motion_blur in
    nit_view_selection/select_best_views.py.

    Stat = variance of the Laplacian restricted to the boundary band
    (dilate XOR erode of the mask). Blurred boundaries are penalized
    (direction "low_bad"). Boundary sharpness is a continuous spectrum, so
    the softness is deliberately small (0.3 robust-MADs) to discriminate.
    """

    BLUR_SCORE_SOFTNESS = 0.3
    BLUR_STROKE_WIDTH = 9

    stat_attr = "vincent_boundary_blur_variance"
    weight_attr = "vincents_motion_blur"
    direction = "low_bad"
    softness = BLUR_SCORE_SOFTNESS

    def __init__(
        self,
        softness: float = BLUR_SCORE_SOFTNESS,
        stroke_width: int = BLUR_STROKE_WIDTH,
        hard_min_variance: float = 0.0,
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness
        self.stroke_width = stroke_width
        self.hard_min_variance = hard_min_variance

    def evaluate(self, observation):
        """Compute the raw boundary-blur stat; hard-reject the blurred tail.

        The soft filter normally never rejects, but a configurable absolute
        floor on the boundary-band variance catches frames whose object
        boundary is smeared by motion blur. The raw stat is always recorded so
        downstream diagnostics and the population weight pass see it.
        """
        if not self.enabled:
            return 1.0, True, ""

        stat = float(self.compute_stat(observation))
        setattr(observation.metrics, self.stat_attr, stat)

        if self.hard_min_variance > 0.0 and stat < self.hard_min_variance:
            return 0.0, False, "motion_blur"

        return 1.0, True, ""

    def compute_stat(self, observation) -> float:

        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)
