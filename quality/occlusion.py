import numpy as np

from .base import QualityMetric


class OcclusionQuality(QualityMetric):

    name = "occlusion"

    def compute(self, observation):

        if observation.object_hand is None:

            return 1.0

        mask = observation.mask > 0

        hand = observation.object_hand > 0

        overlap = np.sum(mask & hand)

        total = np.sum(mask)

        if total == 0:

            return 0.0

        ratio = overlap / total

        return 1.0 - ratio
