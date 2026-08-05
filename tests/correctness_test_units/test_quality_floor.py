# tests/test_quality_floor.py
"""
Quality floor + global quality anchor correctness tests.

Tests:
- adaptive floor drops the worst tail
- floor guarantees a minimum candidate pool
- floor never drops below num_views candidates
- absolute minimum quality respected
- small pools / num_views >= pool size
- global (dataset-independent) anchors used by the quality scorer
"""

from __future__ import annotations

import numpy as np

from tests.test_utils import check


def _qualities(n=100, seed=0, low=0.4, high=1.0):
    rng = np.random.RandomState(seed)
    return rng.uniform(low, high, n)


def _cfg(**overrides):
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    for key, value in overrides.items():
        setattr(cfg.quality_floor, key, value)
    return cfg


def test_floor_drops_bottom_percentile():
    from run import compute_quality_floor

    q = _qualities(100)
    cfg = _cfg(percentile=0.10, absolute_min=0.0)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        np.isclose(floor, np.percentile(q, 10), atol=0.01),
        f"floor sits at the configured percentile ({floor:.4f})",
    )
    check(
        85 <= len(kept) <= 91,
        f"floor drops ~bottom 10% (kept {len(kept)} of 100)",
    )


def test_floor_minimum_pool_guarantee():
    from run import compute_quality_floor

    q = _qualities(100)
    cfg = _cfg(percentile=0.5, minimum_pool=30)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        len(kept) >= 30,
        f"floor keeps at least minimum_pool ({len(kept)} >= 30)",
    )


def test_floor_never_drops_below_num_views():
    from run import compute_quality_floor

    q = _qualities(100)
    cfg = _cfg(percentile=0.9, absolute_min=0.99, minimum_pool=5)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        len(kept) >= 10,
        f"floor never drops below num_views (kept {len(kept)} >= 10)",
    )


def test_floor_absolute_min_excludes_bad_samples():
    from run import compute_quality_floor

    q = _qualities(100, low=0.3, high=0.9)
    cfg = _cfg(absolute_min=0.6)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    check(
        floor >= 0.6,
        f"floor respects absolute minimum ({floor:.4f} >= 0.6)",
    )
    kept = q[q >= floor]
    check(
        np.all(kept >= 0.6),
        "all samples above the floor meet the absolute minimum",
    )


def test_floor_small_pool_keeps_everything():
    from run import compute_quality_floor

    q = _qualities(15)
    cfg = _cfg()
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        len(kept) >= 10,
        f"small pool keeps at least num_views ({len(kept)} >= 10)",
    )


def test_floor_num_views_exceeds_pool():
    from run import compute_quality_floor

    q = _qualities(8)
    cfg = _cfg(absolute_min=0.9)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        len(kept) == 8,
        "num_views >= pool size keeps the whole pool",
    )


def test_floor_zero_keeps_everything():
    from run import compute_quality_floor

    q = _qualities(100)
    cfg = _cfg(percentile=0.0, absolute_min=0.0)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    kept = q[q >= floor]
    check(
        len(kept) == 100,
        "zero floor keeps the whole pool",
    )


def test_quality_scorer_uses_global_anchors():
    from run import build_quality_scorer
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    scorer = build_quality_scorer(cfg, tuned={})

    blur_metric = next(m for m in scorer.metrics if m.name == "blur")
    check(
        blur_metric.max_lap == cfg.quality_anchors.blur_max_lap,
        f"BlurQuality anchored at fixed max_lap={cfg.quality_anchors.blur_max_lap}",
    )
    area_metric = next(m for m in scorer.metrics if m.name == "vincents_area")
    check(
        area_metric.max_fraction == cfg.quality_anchors.vincents_area_max_fraction,
        "VincentsAreaQuality anchored at fixed max fraction",
    )
    artifact_metric = next(m for m in scorer.metrics if m.name == "vincents_artefacts")
    check(
        artifact_metric.max_fraction == cfg.quality_anchors.vincents_artifacts_max_fraction,
        "VincentsArtifactsQuality anchored at fixed max fraction",
    )
    blur2_metric = next(m for m in scorer.metrics if m.name == "vincents_motion_blur")
    check(
        blur2_metric.max_variance == cfg.quality_anchors.vincents_motion_blur_max_variance,
        "VincentsMotionBlurQuality anchored at fixed max variance",
    )


def test_global_vincent_quality_anchors():
    from quality.vincent import (
        VincentsAreaQuality,
        VincentsArtifactsQuality,
        VincentsMotionBlurQuality,
    )
    from data_io.metrics import ObservationMetrics
    from data_io.observation import Observation
    from pathlib import Path

    def make_obs(area_frac=0.1, artifact_frac=0.0, blur_var=5000.0):
        obs = Observation(id=0, image_path=Path(""), mask_path=Path(""),
                          object_hand_path=None)
        obs.metrics = ObservationMetrics(
            vincent_area_fraction=area_frac,
            vincent_artifact_fraction=artifact_frac,
            vincent_boundary_blur_variance=blur_var,
        )
        return obs

    area = VincentsAreaQuality()
    check(
        np.isclose(area.compute(make_obs(area_frac=0.10)), 0.5),
        "vincents_area maps fraction/0.2",
    )
    check(
        np.isclose(area.compute(make_obs(area_frac=0.30)), 1.0),
        "vincents_area saturates at anchor",
    )

    artifacts = VincentsArtifactsQuality()
    check(
        np.isclose(artifacts.compute(make_obs(artifact_frac=0.0)), 1.0),
        "clean mask scores 1.0",
    )
    check(
        np.isclose(artifacts.compute(make_obs(artifact_frac=0.05)), 0.0),
        "artifact fraction at anchor scores 0.0",
    )
    check(
        np.isclose(artifacts.compute(make_obs(artifact_frac=0.025)), 0.5),
        "artifact fraction halfway scores 0.5",
    )

    motion = VincentsMotionBlurQuality()
    check(
        np.isclose(motion.compute(make_obs(blur_var=10000.0)), 1.0),
        "boundary blur variance at anchor scores 1.0",
    )
    check(
        np.isclose(motion.compute(make_obs(blur_var=5000.0)), 0.5),
        "boundary blur variance halfway scores 0.5",
    )
    check(
        np.isclose(motion.compute(make_obs(blur_var=0.0)), 0.0),
        "no boundary sharpness scores 0.0",
    )


def test_quality_floor_selects_only_high_quality_pool():
    from run import compute_quality_floor

    rng = np.random.RandomState(1)
    q = np.concatenate([
        rng.uniform(0.2, 0.45, 30),
        rng.uniform(0.6, 1.0, 70),
    ])
    cfg = _cfg(percentile=0.10, absolute_min=0.5)
    floor = compute_quality_floor(q, num_views=10, cfg=cfg)
    pool = q[q >= floor]
    check(
        np.all(pool >= 0.5),
        "selection pool contains only samples above the absolute minimum",
    )
    check(
        len(pool) >= 10,
        f"selection pool has enough candidates ({len(pool)} >= 10)",
    )
    check(
        pool.mean() > q.mean(),
        "pool average quality exceeds accepted average quality",
    )


def test_greedy_selector_default_balance():
    from selection.greedy_quality_diversity import GreedyQualityDiversity
    from config import PipelineConfig

    cfg = PipelineConfig(auto_thresholds=False)
    s = GreedyQualityDiversity(alpha=cfg.selector_alpha, beta=cfg.selector_beta)
    check(
        np.isclose(cfg.selector_alpha, 0.60) and np.isclose(cfg.selector_beta, 0.40),
        f"default balance is 0.60/0.40 (got {cfg.selector_alpha}/{cfg.selector_beta})",
    )
    check(np.isclose(s.alpha, cfg.selector_alpha), "selector uses config alpha")


QUALITY_FLOOR_TESTS = [
    ("Floor drops bottom percentile", test_floor_drops_bottom_percentile),
    ("Floor minimum pool guarantee", test_floor_minimum_pool_guarantee),
    ("Floor never drops below num_views", test_floor_never_drops_below_num_views),
    ("Floor absolute min", test_floor_absolute_min_excludes_bad_samples),
    ("Floor small pool", test_floor_small_pool_keeps_everything),
    ("Floor num_views exceeds pool", test_floor_num_views_exceeds_pool),
    ("Floor zero keeps everything", test_floor_zero_keeps_everything),
    ("Quality scorer global anchors", test_quality_scorer_uses_global_anchors),
    ("Global Vincent quality anchors", test_global_vincent_quality_anchors),
    ("Floor selects high-quality pool", test_quality_floor_selects_only_high_quality_pool),
    ("Greedy default balance", test_greedy_selector_default_balance),
]
