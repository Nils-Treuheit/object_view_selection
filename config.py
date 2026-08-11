from dataclasses import dataclass, field


@dataclass
class LaplacianBlurConfig:
    """Boundary-band Laplacian-variance pre-filter (blur_laplacian).

    ``max_variance`` anchors the (0, 1] goodness score of the filter;
    ``hard_min_variance`` is the absolute garbage floor on the raw stat and
    ``outlier_z`` removes extreme bad outliers relative to the population
    (both criteria implemented by the filter itself via ``ScoreFilter``).
    """
    enabled: bool = True
    stroke_width: int = 9
    max_variance: float = 20000.0
    hard_min_variance: float = 4200.0
    outlier_z: float = 2.0


@dataclass
class TenengradBlurConfig:
    """Boundary-band Tenengrad pre-filter (blur_tenengrad).

    Same rejection knobs as ``LaplacianBlurConfig``, on the mean Sobel
    magnitude restricted to the mask boundary band.
    """
    enabled: bool = True
    stroke_width: int = 9
    max_tenengrad: float = 150.0
    hard_min_tenengrad: float = 72.5
    outlier_z: float = 2.0


@dataclass
class AreaConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    minimum_ratio: float = 0.01
    outlier_z: float | None = None


@dataclass
class BorderConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    maximum_ratio: float = 0.05
    edge_maximum_ratio: float = 0.25
    outlier_z: float | None = None


@dataclass
class OcclusionConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    maximum_overlap: float = 0.15
    outlier_z: float | None = None


@dataclass
class ConfidenceConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = False
    minimum_confidence: float = 0.5
    outlier_z: float | None = None


@dataclass
class CompletenessConfig:
    # NOT part of the default pre-filter set and not tested / likely not
    # working as a proper pre-filter. Kept only for custom --filter_order.
    enabled: bool = True
    minimum_score: float = 0.65
    outlier_z: float | None = None


@dataclass
class VincentEmptyMaskConfig:
    enabled: bool = True
    outlier_z: float | None = None


@dataclass
class VincentBorderPixelConfig:
    enabled: bool = True
    outlier_z: float | None = None


@dataclass
class VincentsAreaConfig:
    enabled: bool = True
    softness: float = 0.3
    # absolute garbage floor on the raw area fraction (0 disables)
    hard_min_area_fraction: float = 0.0
    # extreme-bad-outlier cutoff on the raw stat via fit/evaluate
    outlier_z: float | None = None


@dataclass
class VincentsArtifactsConfig:
    enabled: bool = True
    kernel_size: int = 10
    # artifact fraction at which the filter's goodness score hits 0.0
    max_fraction: float = 0.05
    # absolute garbage ceiling and extreme-bad-outlier removal on the raw stat
    hard_max_fraction: float = 0.15
    outlier_z: float = 2.0


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
    # True when ``--filter_order`` was passed explicitly: only the named
    # pre-filters execute — including the soft filters (``vincents_area``,
    # ``vincents_motion_blur``), which otherwise always run as diagnostics.
    explicit_filter_order: bool = False


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

    OPT-IN: disabled by default. When enabled, the floor drops the worst tail
    of the accepted pool so that low-quality samples are excluded from (or
    extremely unlikely to enter) the final selection set, while guaranteeing
    enough candidates remain for a diverse sample-set selection.
    """

    enabled: bool = False
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
    auto_thresholds: bool = False

    selector: str = "quality_diversity"
    selector_alpha: float = 0.60
    selector_beta: float = 0.40
    # How the GQD selector aggregates the distance from a candidate to the
    # already-selected set: "min" (nearest sample), "max" (farthest sample),
    # "prototype" (average sample of the set).
    selector_diversity_mode: str = "min"
    # Include a descriptor-based divergence score in the GQD diversity term.
    selector_use_descriptors: bool = False
    selector_descriptor_weight: float = 0.5
    selector_descriptor: str = "silhouette"      # silhouette | hu | zernike | fourier | shape_context
    dpp_sigma: float = 0.5
    # Top kMeans Embedding Selection in xNN quality Neighborhood
    kmeans_init: str = "best_quality"        # "farthest" | "best_quality"
    # k-means clusters for the top_kmeans_xnn selector; None => one cluster per
    # requested view (k = num_views). A smaller explicit k selects k views via
    # kMeans-xNN and fills the rest from the clusters by average quality.
    kmeans_k: int | None = None
    kmeans_xnn_k: int = 10                    # xNN radius: nearest neighbours around each centroid

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