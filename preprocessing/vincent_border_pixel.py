import numpy as np

from .base import BaseFilter
from .vincent_utils import touches_border_pixels


class VincentBorderPixelFilter(BaseFilter):

    def __init__(
        self,
        enabled=True,
    ):

        super().__init__(enabled)

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0

        touches = touches_border_pixels(mask)

        observation.metrics.vincent_touches_border = float(touches)

        if touches:
            return 0.0, False, "vincent_border_pixel"

        return 1.0, True, ""
