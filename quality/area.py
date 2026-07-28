import numpy as np

from .base import QualityMetric


class AreaQuality(QualityMetric):

    name = "area"

    def compute(self, observation):

        mask = observation.mask > 0

        ratio = np.mean(mask)

        return min(
            ratio / 0.20,
            1.0,
        )
