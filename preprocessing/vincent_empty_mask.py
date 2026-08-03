import numpy as np

from .base import BaseFilter


class VincentEmptyMaskFilter(BaseFilter):

    def __init__(
        self,
        enabled=True,
    ):

        super().__init__(enabled)

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0

        pixel_count = int(np.sum(mask))

        observation.metrics.vincent_pixel_count = float(pixel_count)

        if pixel_count <= 0:
            return 0.0, False, "vincent_empty_mask"

        return 1.0, True, ""
