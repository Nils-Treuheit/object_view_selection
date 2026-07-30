from dataclasses import dataclass, field


@dataclass
class BlurConfig:
    enabled: bool = True
    threshold: float = 120.0
    tenengrad_threshold: float = 35.0


@dataclass
class AreaConfig:
    enabled: bool = True
    minimum_ratio: float = 0.01


@dataclass
class BorderConfig:
    enabled: bool = True
    maximum_ratio: float = 0.05


@dataclass
class OcclusionConfig:
    enabled: bool = True
    maximum_overlap: float = 0.15


@dataclass
class ConfidenceConfig:
    enabled: bool = False
    minimum_confidence: float = 0.5


@dataclass
class CompletenessConfig:
    enabled: bool = True
    minimum_score: float = 0.65


@dataclass
class FilterConfig:
    blur: BlurConfig = field(default_factory=BlurConfig)
    area: AreaConfig = field(default_factory=AreaConfig)
    border: BorderConfig = field(default_factory=BorderConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    filter_order: list = field(default_factory=lambda: [
        "border", "area", "confidence", "blur", "occlusion", "completeness"
    ])


@dataclass
class QualityWeights:
    blur: float = 0.20
    area: float = 0.15
    occlusion: float = 0.20
    confidence: float = 0.10
    completeness: float = 0.35


@dataclass
class PipelineConfig:
    data_root: str = ""
    output_dir: str = "outputs"
    num_views: int = 10

    filters: FilterConfig = field(default_factory=FilterConfig)

    embedding: str = "auto"
    embedding_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    use_shape_descriptors: bool = False
    shape_descriptor: str = "hu"
    auto_thresholds: bool = True

    selector: str = "quality_diversity"
    selector_alpha: float = 0.4
    selector_beta: float = 0.6
    dpp_sigma: float = 0.5

    quality_weights: QualityWeights = field(default_factory=QualityWeights)

    save_visualization: bool = True
    save_rejected: bool = True
    save_embeddings: bool = True
    save_plots: bool = False
    debug: bool = False