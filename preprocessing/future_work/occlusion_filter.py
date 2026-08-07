import numpy as np

from ..base import BaseFilter


class OcclusionFilter(BaseFilter):

    def __init__(
        self,
        maximum_overlap=0.15,
        enabled=True,
    ):

        super().__init__(enabled)

        self.maximum_overlap = maximum_overlap

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        if observation.object_hand is None:

            return 1.0, True, ""

        mask = observation.mask > 0

        hand = observation.object_hand > 0

        overlap = np.sum(mask & hand)

        total = np.sum(mask)

        if total == 0:
            return 0, False, "empty_mask"

        ratio = overlap / total

        observation.metrics.hand_overlap = ratio

        score = 1.0 - min(
            ratio / self.maximum_overlap,
            1.0,
        )

        passed = ratio <= self.maximum_overlap

        return score, passed, "occlusion"
