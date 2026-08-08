import numpy as np

from ..base import BaseFilter


class AreaFilter(BaseFilter):

    def __init__(
        self,
        minimum_ratio=0.02,
        enabled=True,
    ):

        super().__init__(enabled)

        self.minimum_ratio = minimum_ratio

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0

        area = np.sum(mask)

        total = mask.shape[0] * mask.shape[1]

        ratio = area / total

        observation.metrics.area_ratio = ratio

        score = min(
            ratio / self.minimum_ratio,
            1.0,
        )

        passed = ratio >= self.minimum_ratio

        return score, passed, "small_object"
