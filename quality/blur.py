from preprocessing.blur_filter import BlurFilter

from .base import QualityMetric


class BlurQuality(QualityMetric):

    name = "blur"

    def __init__(self):

        self.blur = BlurFilter(enabled=False)

    def compute(self, observation):

        lap = self.blur.variance_of_laplacian(
            observation.image
        )

        score = min(
            lap / 300.0,
            1.0,
        )

        return score
