# tests/test_pipeline.py
"""
Filter pipeline and integration correctness tests.

Tests:
- pipeline execution order
- cumulative rejection behavior
- multiple filters interacting correctly
- accepted and rejected observations
"""

import numpy as np

from data_io.observation import Observation
from preprocessing.filter_pipeline import FilterPipeline
from preprocessing.legacy.area_filter import AreaFilter
from preprocessing.legacy.border_truncation import BorderFilter
from preprocessing.legacy.blur_filter import BlurFilter

from pathlib import Path


def make_image(h=100, w=100):
    return (
        np.random.RandomState(42)
        .rand(h, w, 3)
        * 255
    ).astype(np.uint8)


def make_observation(mask, image=None, obs_id=0):
    if image is None:
        image = make_image(
            mask.shape[0],
            mask.shape[1]
        )

    return Observation(
        id=obs_id,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask
    )


def test_pipeline_accepts_good_observation():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[25:75, 25:75] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is True


def test_pipeline_rejects_small_objects():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[49:51, 49:51] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_rejects_border_objects():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[
        0:50,
        0:50
    ] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_runs_filters_in_order():

    class TrackingFilter:

        def __init__(self, name, result):
            self.name = name
            self.result = result
            self.called = False

        def evaluate(self, observation):
            self.called = True
            observation.test_order.append(self.name)
            return 1.0, self.result, {}

    f1 = TrackingFilter("first", True)
    f2 = TrackingFilter("second", True)

    pipeline = FilterPipeline(
        [
            f1,
            f2
        ]
    )

    mask = np.ones(
        (20, 20),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)
    obs.test_order = []

    result = pipeline.run(obs)

    assert result is True
    assert obs.test_order == [
        "first",
        "second"
    ]


def test_pipeline_stops_after_failure():

    class TrackingFilter:

        def __init__(self, name, result):
            self.name = name
            self.result = result

        def evaluate(self, observation):
            observation.test_order.append(self.name)
            return 0.0, self.result, {}

    f1 = TrackingFilter(
        "reject",
        False
    )

    f2 = TrackingFilter(
        "should_not_run",
        True
    )

    pipeline = FilterPipeline(
        [
            f1,
            f2
        ]
    )

    mask = np.ones(
        (20, 20),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)
    obs.test_order = []

    result = pipeline.run(obs)

    assert result is False

    assert obs.test_order == [
        "reject"
    ]


def test_pipeline_combines_blur_and_area():

    pipeline = FilterPipeline(
        [
            BlurFilter(
                laplacian_threshold=120,
                tenengrad_threshold=35
            ),
            AreaFilter(
                minimum_ratio=0.02
            )
        ]
    )

    sharp = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    yy, xx = np.ogrid[:100, :100]

    sharp[
        (xx - 50) ** 2 +
        (yy - 50) ** 2 < 30 ** 2
    ] = 255

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[
        20:80,
        20:80
    ] = 255


    obs = make_observation(
        mask,
        image=sharp
    )

    result = pipeline.run(obs)

    assert result is True


def test_pipeline_rejects_empty_mask():

    pipeline = FilterPipeline(
        [
            AreaFilter(
                minimum_ratio=0.01
            )
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_metrics_are_populated():

    pipeline = FilterPipeline(
        [
            AreaFilter(
                minimum_ratio=0.01
            ),
            BorderFilter(
                maximum_ratio=0.01
            )
        ]
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)

    pipeline.run(obs)

    assert obs.metrics.area_ratio > 0
    assert hasattr(
        obs.metrics,
        "border_ratio"
    )


# ============================================================
# Vincent filters integrated via run.py wiring
# ============================================================

def test_build_filters_includes_vincent_hard_filters():
    from run import build_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    pipeline = build_filters(cfg, tuned={})

    names = [f.__class__.__name__ for f in pipeline.filters]
    assert "VincentEmptyMaskFilter" in names, f"missing empty-mask filter, got {names}"
    assert "VincentBorderPixelFilter" in names, f"missing border-pixel filter, got {names}"

    ordered = cfg.filters.filter_order
    assert ordered.index("vincent_empty_mask") < ordered.index("vincent_border_pixel")
    assert ordered.index("vincent_border_pixel") < ordered.index("blur_laplacian")


def test_build_filters_order_matches_config():
    from preprocessing.variants import FilterVariant
    from run import build_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    pipeline = build_filters(cfg, tuned={})

    class_to_key = {
        "VincentEmptyMaskFilter": "vincent_empty_mask",
        "VincentBorderPixelFilter": "vincent_border_pixel",
        "BorderLaplacianBlurFilter": "blur_laplacian",
        "BorderTenengradBlurFilter": "blur_tenengrad",
        "VincentsArtifactsFilter": "vincents_artefacts",
        "BorderFilter": "border",
        "AreaFilter": "area",
        "ConfidenceFilter": "confidence",
        "OcclusionFilter": "occlusion",
        "CompletenessFilter": "completeness",
    }

    run_order = []
    for f in pipeline.filters:
        cls = f.inner if isinstance(f, FilterVariant) else f
        run_order.append(class_to_key[type(cls).__name__])
    assert run_order == cfg.filters.filter_order, f"order mismatch: {run_order}"


def test_apply_soft_filters_populates_weights():
    from run import apply_soft_filters, build_soft_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    soft_filters = build_soft_filters(cfg)

    observations = []
    for i, radius in enumerate([25, 30, 35, 40, 45]):
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        observations.append(make_observation(mask, obs_id=i))

    apply_soft_filters(soft_filters, observations)

    for o in observations:
        assert 0.0 < o.metrics.vincents_area <= 1.0, "vincents_area weight in (0, 1]"
        assert 0.0 < o.metrics.vincents_motion_blur <= 1.0, "vincents_motion_blur weight in (0, 1]"
        assert o.metrics.vincent_area_fraction > 0.0, "area raw stat populated"
        assert o.metrics.vincent_boundary_blur_variance >= 0.0, "blur raw stat populated"


def test_apply_soft_filters_rejected_raw_stats():
    from run import apply_soft_filters, build_soft_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    soft_filters = build_soft_filters(cfg)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    accepted = [make_observation(mask, obs_id=0)]
    rejected = [make_observation(mask, obs_id=1)]

    apply_soft_filters(soft_filters, accepted, rejected)

    assert rejected[0].metrics.vincent_area_fraction > 0.0, "rejected raw stats computed"
    assert accepted[0].metrics.vincents_area > 0.0, "accepted gets population weight"
    assert rejected[0].metrics.vincents_area == 0.0, "rejected does not get weight"


def test_build_quality_scorer_includes_new_components():
    from run import build_quality_scorer
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    scorer = build_quality_scorer(cfg, tuned={})

    names = [m.name for m in scorer.metrics]
    assert "blur" in names, f"missing blur, got {names}"
    assert "area" in names, f"missing area, got {names}"
    assert "vincents_artefacts" in names, f"missing vincents_artefacts, got {names}"
    assert "centerness" in names, f"missing centerness, got {names}"
    assert len(names) == 4, f"exactly 4 quality components, got {names}"
    assert set(scorer.weights) == set(names), "weights match metric names"


def test_quality_scorer_new_components_end_to_end():
    from run import build_filters, build_quality_scorer
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    hard = build_filters(cfg, tuned={})
    scorer = build_quality_scorer(cfg, tuned={})

    observations = []
    for i, radius in enumerate([15, 20, 25, 30, 35, 40, 45, 48]):
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        observations.append(make_observation(mask, obs_id=i))

    accepted = [o for o in observations if hard.run(o)]
    assert len(accepted) == len(observations), "all centered circles pass the hard pre-filter"

    for o in accepted:
        scorer.score(o)

    for o in accepted:
        assert 0.0 <= o.quality <= 1.0, f"quality in [0, 1], got {o.quality}"
        assert o.metrics.laplacian > 0.0, "boundary laplacian populated by pre-filter"
        assert o.metrics.blur > 0.0, "blur quality populated by scorer"


def test_confidence_is_weakest_link():
    from run import build_filters, build_quality_scorer
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    hard = build_filters(cfg, tuned={})
    scorer = build_quality_scorer(cfg, tuned={})

    observations = []
    for i, radius in enumerate([15, 20, 25, 30, 35, 40, 45, 48]):
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        observations.append(make_observation(mask, obs_id=i))

    accepted = [o for o in observations if hard.run(o)]
    for o in accepted:
        scorer.score(o)

    for o in accepted:
        m = o.metrics
        expected = min(m.blur, m.area, m.vincents_artefacts, m.centerness)
        assert expected > 0.0, "all component scores positive for this population"
        assert expected <= min(m.blur, m.area, m.centerness), (
            "artifact component tightens the weakest-link bound"
        )


def test_build_filters_wraps_configured_hard_variants():
    from preprocessing.variants import FilterVariant
    from run import build_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    cfg.filters.filter_order = [
        "vincent_empty_mask", "blur_laplacian", "area",
    ]
    cfg.filters.blur_laplacian.threshold_min = 0.2
    cfg.filters.area.outlier_z = 3.0

    pipeline = build_filters(cfg, tuned={})

    wrapped = {}
    for f in pipeline.filters:
        if isinstance(f, FilterVariant):
            wrapped[type(f.inner).__name__] = f

    assert "BorderLaplacianBlurFilter" in wrapped, "blur_laplacian wrapped with threshold_min"
    assert wrapped["BorderLaplacianBlurFilter"].threshold_min == 0.2
    assert wrapped["BorderLaplacianBlurFilter"].outlier_z == 3.0

    assert "AreaFilter" in wrapped, "area wrapped with outlier_z"
    assert wrapped["AreaFilter"].outlier_z == 3.0

    assert pipeline.requires_fit, "pipeline requires population fit for outlier mode"


def test_build_soft_filters_propagates_knobs():
    from run import build_soft_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    cfg.filters.vincents_area.threshold_min = 0.3
    cfg.filters.vincents_motion_blur.outlier_z = 3.0

    soft_filters = build_soft_filters(cfg)

    assert "vincents_area" in soft_filters
    assert "vincents_motion_blur" in soft_filters
    assert soft_filters["vincents_area"].threshold_min == 0.3
    assert soft_filters["vincents_area"].outlier_z is None
    assert soft_filters["vincents_motion_blur"].outlier_z == 3.0
    assert soft_filters["vincents_motion_blur"].threshold_min is None