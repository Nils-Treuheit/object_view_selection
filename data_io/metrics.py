from dataclasses import dataclass


@dataclass
class ObservationMetrics:

    # preprocessing
    laplacian: float = 0.0
    tenengrad: float = 0.0
    area_ratio: float = 0.0
    border_ratio: float = 0.0
    edge_top_ratio: float = 0.0
    edge_bottom_ratio: float = 0.0
    edge_left_ratio: float = 0.0
    edge_right_ratio: float = 0.0
    edge_ratio: float = 0.0
    hand_overlap: float = 0.0

    # vincent hard pre-filters
    vincent_pixel_count: float = 0.0
    vincent_touches_border: float = 0.0

    # vincent soft pre-filters: raw stats
    vincent_area_fraction: float = 0.0
    vincent_artifact_fraction: float = 0.0
    vincent_boundary_blur_variance: float = 0.0

    # vincent soft pre-filters: population-adapted weights (0, 1]
    vincents_area: float = 0.0
    vincents_artefacts: float = 0.0
    vincents_motion_blur: float = 0.0

    # shape
    solidity: float = 0.0
    extent: float = 0.0
    convexity: float = 0.0
    completeness: float = 0.0

    # quality metrics
    blur: float = 0.0
    area: float = 0.0
    occlusion: float = 0.0
    confidence: float = 0.0

    # final score
    quality: float = 0.0
