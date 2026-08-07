import numpy as np

from .base import QualityMetric


class VincentsAreaQuality(QualityMetric):
    """Global mask-area quality from the raw area fraction.

    Anchored at a fixed global max fraction (0.20) so the score is
    comparable across datasets; identical scale to AreaQuality.
    """

    name = "vincents_area"

    def __init__(self, max_fraction: float = 0.20):
        self.max_fraction = max_fraction

    def compute(self, observation):
        fraction = getattr(observation.metrics, "vincent_area_fraction", 0.0)
        return min(fraction / self.max_fraction, 1.0)


class VincentsArtifactsQuality(QualityMetric):
    """Global mask-artifact quality from the raw artifact fraction.

    Artifact fraction = (open(mask) XOR close(mask)) / mask_pixels, computed
    by the ``VincentsArtifactsFilter`` pre-filter. Anchored at a fixed global
    max fraction; a mask whose artifact fraction reaches the anchor scores 0.
    """

    name = "vincents_artefacts"

    def __init__(self, max_fraction: float = 0.05):
        self.max_fraction = max_fraction

    def compute(self, observation):
        fraction = getattr(observation.metrics, "vincent_artifact_fraction", 0.0)
        return float(np.clip(1.0 - fraction / self.max_fraction, 0.0, 1.0))


class VincentsMotionBlurQuality(QualityMetric):
    """Global boundary-sharpness quality from the raw boundary-blur variance.

    Anchored at a fixed global max variance; a boundary whose Laplacian
    variance reaches the anchor scores 1.0.
    """

    name = "vincents_motion_blur"

    def __init__(self, max_variance: float = 10000.0):
        self.max_variance = max_variance

    def compute(self, observation):
        variance = getattr(observation.metrics, "vincent_boundary_blur_variance", 0.0)
        return min(variance / self.max_variance, 1.0)
