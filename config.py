from dataclasses import dataclass, field


@dataclass
class LaplacianBlurConfig:
    """Boundary-band Laplacian-variance pre-filter (blur_laplacian).

    ``max_variance`` anchors the (0, 1] goodness score of the filter;
    ``threshold_min`` is the very relaxed absolute floor (below = awful
    quality) and ``outlier_z`` removes extreme bad outliers relative to the
    population (see preprocessing/variants.py).
    """
    enabled: bool = True
    stroke_width: int = 9
    max_variance: float = 20000.0
    hard_min_variance: float = 4000.0
    threshold_min: float = 0.01
    outlier_z: float = 3.0


@dataclass
class TenengradBlurConfig:
    """Boundary-band Tenengrad pre-filter (blur_tenengrad).

    Same rejection knobs as ``LaplacianBlurConfig``, on the mean Sobel
    magnitude restricted to the mask boundary band.
    """
    enabled: bool = True
    stroke_width: int = 9
    max_tenengrad: float = 150.0
    hard_min_tenengrad: float = 33.0
    threshold_min: float = 0.10
    outlier_z: float = 3.0


@dataclass
class AreaConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    minimum_ratio: float = 0.01
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class BorderConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    maximum_ratio: float = 0.05
    edge_maximum_ratio: float = 0.25
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class OcclusionConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    maximum_overlap: float = 0.15
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class ConfidenceConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = False
    minimum_confidence: float = 0.5
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class CompletenessConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    minimum_score: float = 0.65
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class VincentEmptyMaskConfig:
    enabled: bool = True
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class VincentBorderPixelConfig:
    enabled: bool = True
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class VincentsAreaConfig:
    enabled: bool = True
    softness: float = 0.3
    # absolute garbage floor on the raw area fraction (0 disables)
    hard_min_area_fraction: float = 0.0
    # threshold/outlier variants on the fit (0,1] weight (see preprocessing/variants.py)
    # also used as an extreme-bad-outlier cutoff on the raw stat via fit/evaluate
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class VincentsArtifactsConfig:
    enabled: bool = True
    kernel_size: int = 10
    # artifact fraction at which the filter's goodness score hits 0.0
    max_fraction: float = 0.05
    # very relaxed absolute floor and extreme-bad-outlier removal on the score
    threshold_min: float = 0.05
    outlier_z: float = 3.0


@dataclass
class VincentsMotionBlurConfig:
    enabled: bool = True
    softness: float = 0.3
    stroke_width: int = 9
    # Absolute hard-reject floor on the boundary-band Laplacian variance.
    # Frames whose object boundary is smeared/blurred (motion blur) score far
    # below sharp datasets (e.g. < ~150 on 480x640 triprong vs > 1400 min on
    # bottle), so this excludes the motion-blurred tail that the soft weight
    # would otherwise merely down-rank. 0 disables the hard reject.
    hard_min_variance: float = 120.0
    threshold_min: float | None = None
    outlier_z: float | None = None


@dataclass
class FilterConfig:
    blur_laplacian: LaplacianBlurConfig = field(default_factory=LaplacianBlurConfig)
    blur_tenengrad: TenengradBlurConfig = field(default_factory=TenengradBlurConfig)
    area: AreaConfig = field(default_factory=AreaConfig)
    border: BorderConfig = field(default_factory=BorderConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    vincent_empty_mask: VincentEmptyMaskConfig = field(default_factory=VincentEmptyMaskConfig)
    vincent_border_pixel: VincentBorderPixelConfig = field(default_factory=VincentBorderPixelConfig)
    vincents_area: VincentsAreaConfig = field(default_factory=VincentsAreaConfig)
    vincents_artefacts: VincentsArtifactsConfig = field(default_factory=VincentsArtifactsConfig)
    vincents_motion_blur: VincentsMotionBlurConfig = field(default_factory=VincentsMotionBlurConfig)
    filter_order: list = field(default_factory=lambda: [
        "vincent_empty_mask", "vincent_border_pixel",
        "blur_laplacian", "blur_tenengrad", "vincents_artefacts"
    ])


@dataclass
class QualityWeights:
    """Weights of the 4 quality components (weighted average)."""
    blur: float = 0.30
    area: float = 0.20
    vincents_artefacts: float = 0.20
    centerness: float = 0.30


@dataclass
class QualityAnchors:
    """Fixed, dataset-independent scales that map raw quality stats to [0, 1].

    Pre-filter criteria stay population-relative (median/MAD, percentiles);
    quality scoring uses these global anchors so scores are comparable
    across datasets.
    """

    blur_max_variance: float = 10000.0
    area_max_fraction: float = 0.20
    artifacts_max_fraction: float = 0.05


@dataclass
class QualityFloorConfig:
    """Adaptive minimum-quality floor applied before embedding selection.

    The floor drops the worst tail of the accepted pool so that low-quality
    samples are excluded from (or extremely unlikely to enter) the final
    selection set, while guaranteeing enough candidates remain for a diverse
    sample-set selection.
    """

    enabled: bool = True
    percentile: float = 0.10
    minimum_pool: int = 20
    absolute_min: float = 0.66


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
    selector_alpha: float = 0.60
    selector_beta: float = 0.40
    dpp_sigma: float = 0.5
    # Top kMeans Embedding Selection in xNN quality Neighborhood
    kmeans_init: str = "farthest"        # "farthest" | "best_quality"
    kmeans_xnn_k: int = 3                # xNN radius: 3 | 5 | 10

    quality_weights: QualityWeights = field(default_factory=QualityWeights)
    quality_anchors: QualityAnchors = field(default_factory=QualityAnchors)
    quality_floor: QualityFloorConfig = field(default_factory=QualityFloorConfig)

    save_visualization: bool = True
    save_rejected: bool = True
    save_embeddings: bool = True
    save_plots: bool = False
    debug: bool = False

    # stop after the pre-filter stage (no quality scoring / embedding /
    # selection / plots); still dumps accepted_samples/ and
    # rejected_samples/<reason>/
    only_pre_filter: bool = False