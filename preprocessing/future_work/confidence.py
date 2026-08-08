from ..base import BaseFilter


class ConfidenceFilter(BaseFilter):

    def __init__(
        self,
        minimum_confidence=0.5,
        enabled=True,
    ):
        super().__init__(enabled)
        self.minimum_confidence = minimum_confidence

    def evaluate(self, observation):
        if not self.enabled:
            return 1.0, True, ""

        confidence = getattr(observation, "confidence", None)
        if confidence is None:
            return 1.0, True, ""

        observation.metrics.confidence = confidence
        score = min(confidence / self.minimum_confidence, 1.0)
        passed = confidence >= self.minimum_confidence

        return score, passed, "low_confidence"
