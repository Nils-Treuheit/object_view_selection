from .base import QualityMetric


class CompletenessQuality(QualityMetric):

    name = "completeness"

    def compute(self, observation):

        return observation.metrics.get(
            "completeness",
            0.0,
        )
