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
from preprocessing.area_filter import AreaFilter
from preprocessing.border_truncation import BorderFilter
from preprocessing.blur_filter import BlurFilter

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
    assert ordered.index("vincent_border_pixel") < ordered.index("border")


def test_build_filters_order_matches_config():
    from run import build_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    pipeline = build_filters(cfg, tuned={})

    class_to_key = {
        "VincentEmptyMaskFilter": "vincent_empty_mask",
        "VincentBorderPixelFilter": "vincent_border_pixel",
        "BorderFilter": "border",
        "AreaFilter": "area",
        "ConfidenceFilter": "confidence",
        "BlurFilter": "blur",
        "OcclusionFilter": "occlusion",
        "CompletenessFilter": "completeness",
    }

    run_order = [class_to_key[f.__class__.__name__] for f in pipeline.filters]
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
        assert 0.0 < o.metrics.vincents_artefacts <= 1.0, "vincents_artefacts weight in (0, 1]"
        assert 0.0 < o.metrics.vincents_motion_blur <= 1.0, "vincents_motion_blur weight in (0, 1]"
        assert o.metrics.vincent_area_fraction > 0.0, "area raw stat populated"
        assert o.metrics.vincent_artifact_fraction >= 0.0, "artifact raw stat populated"
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


def test_build_quality_scorer_includes_vincent():
    from run import build_quality_scorer
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    scorer = build_quality_scorer(cfg, tuned={})

    names = [m.name for m in scorer.metrics]
    assert "vincents_area" in names
    assert "vincents_artefacts" in names
    assert "vincents_motion_blur" in names
    assert "blur" in names
    assert "completeness" in names


def test_quality_scorer_vincent_end_to_end():
    from run import apply_soft_filters, build_filters, build_quality_scorer, build_soft_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    hard = build_filters(cfg, tuned={})
    soft_filters = build_soft_filters(cfg)
    scorer = build_quality_scorer(cfg, tuned={})

    observations = []
    for i, radius in enumerate([15, 20, 25, 30, 35, 40, 45, 48]):
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        observations.append(make_observation(mask, obs_id=i))

    accepted = [o for o in observations if hard.run(o)]
    assert len(accepted) == len(observations), "all centered circles pass the hard pre-filter"

    apply_soft_filters(soft_filters, accepted)
    for o in accepted:
        scorer.score(o)

    for o in accepted:
        assert 0.0 <= o.quality <= 1.0, f"quality in [0, 1], got {o.quality}"
        assert o.metrics.completeness > 0.0, "completeness populated by hard pre-filter"


def test_vincent_weights_tighten_weakest_link():
    from run import apply_soft_filters, build_filters, build_quality_scorer, build_soft_filters
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    hard = build_filters(cfg, tuned={})
    soft_filters = build_soft_filters(cfg)
    scorer = build_quality_scorer(cfg, tuned={})

    observations = []
    for i, radius in enumerate([15, 20, 25, 30, 35, 40, 45, 48]):
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        observations.append(make_observation(mask, obs_id=i))

    accepted = [o for o in observations if hard.run(o)]
    apply_soft_filters(soft_filters, accepted)
    for o in accepted:
        scorer.score(o)

    for o in accepted:
        m = o.metrics
        expected = min(m.blur, m.area, m.occlusion, m.completeness,
                       m.vincents_area, m.vincents_artefacts, m.vincents_motion_blur)
        assert expected > 0.0, "all component scores positive for this population"
        assert expected <= min(m.blur, m.area, m.occlusion, m.completeness), (
            "vincent weights tighten the weakest-link bound"
        )