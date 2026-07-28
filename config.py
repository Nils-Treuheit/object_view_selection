from dataclasses import dataclass


@dataclass
class BlurConfig:

    enabled: bool = True

    threshold: float = 120.0


@dataclass
class AreaConfig:

    enabled: bool = True

    minimum_ratio: float = 0.02


@dataclass
class BorderConfig:

    enabled: bool = True

    maximum_ratio: float = 0.01


@dataclass
class OcclusionConfig:

    enabled: bool = True

    maximum_overlap: float = 0.15


@dataclass
class PipelineConfig:

    blur = BlurConfig()

    area = AreaConfig()

    border = BorderConfig()

    occlusion = OcclusionConfig()

    embedding = "dinov2"

    selector = "quality_diversity"

    num_views = 10
