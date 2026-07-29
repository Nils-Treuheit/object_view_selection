from .base import QualityMetric


class CompletenessQuality(QualityMetric):

    name = "completeness"

    def compute(self, observation):
        return getattr(observation.metrics, "completeness", 0.0)