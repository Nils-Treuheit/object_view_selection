import numpy as np

from .base import BaseFilter


class BorderFilter(BaseFilter):

    def __init__(
        self,
        maximum_ratio=0.01,
        enabled=True,
    ):

        super().__init__(enabled)

        self.maximum_ratio = maximum_ratio

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0

        border = np.zeros_like(mask)

        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        border_pixels = np.sum(mask & border)

        total = np.sum(mask)

        if total == 0:
            return 0, False, "empty_mask"

        ratio = border_pixels / total

        observation.metrics.border_ratio = ratio

        score = 1.0 - min(
            ratio / self.maximum_ratio,
            1.0,
        )

        passed = ratio <= self.maximum_ratio

        return score, passed, "border"
