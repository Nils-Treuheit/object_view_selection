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
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness
        self.stroke_width = stroke_width

    def compute_stat(self, observation) -> float:

        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)
