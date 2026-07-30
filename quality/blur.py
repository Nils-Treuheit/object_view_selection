from preprocessing.blur_filter import BlurFilter

from .base import QualityMetric


class BlurQuality(QualityMetric):

    name = "blur"

    def __init__(self, max_lap=300.0):

        self.blur = BlurFilter(enabled=False)
        self.max_lap = max_lap

    def compute(self, observation):

        lap = self.blur.variance_of_laplacian(
            observation.image
        )

        score = min(
            lap / self.max_lap,
            1.0,
        )

        return score
