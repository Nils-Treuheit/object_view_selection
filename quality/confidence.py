from .base import QualityMetric


class ConfidenceQuality(QualityMetric):

    name = "confidence"

    def compute(self, observation):
        return getattr(observation.metrics, "confidence", 1.0)