#!/usr/bin/env python3
"""
Correctness tests for preprocessing filters.

Tests

- BlurFilter
- AreaFilter
- BorderFilter
- OcclusionFilter
- CompletenessFilter
"""

from pathlib import Path

import numpy as np

from data_io.observation import Observation

from preprocessing.base import BaseFilter
from preprocessing.legacy.blur_filter import BlurFilter
from preprocessing.legacy.area_filter import AreaFilter
from preprocessing.legacy.border_truncation import BorderFilter
from preprocessing.future_work.occlusion_filter import OcclusionFilter
from preprocessing.future_work.completeness_filter import CompletenessFilter

from tests.test_utils import (
    check,
    make_image,
    full_mask,
    gaussian_blur,
    make_circle_mask,
    make_flower_mask,
)

# ------------------------------------------------------------
# helper
# ------------------------------------------------------------

def observation(image,
                mask,
                hand=None,
                idx=0):

    return Observation(
        id=idx,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask,
        object_hand=hand,
    )


# ============================================================
# BLUR
# ============================================================

def test_blur():

    h = 100
    w = 100

    sharp = np.zeros((h, w, 3), dtype=np.uint8)

    ys, xs = np.ogrid[:h, :w]

    sharp[
        (xs - w // 2) ** 2 +
        (ys - h // 2) ** 2
        <= 35 ** 2
    ] = 255

    blurry = gaussian_blur(sharp, sigma=10)

    bf = BlurFilter(
        laplacian_threshold=120,
        tenengrad_threshold=35,
    )

    obs_sharp = observation(
        sharp,
        np.ones((h, w), np.uint8) * 255,
        idx=0,
    )

    obs_blur = observation(
        blurry,
        np.ones((h, w), np.uint8) * 255,
        idx=1,
    )

    _, pass_sharp, _ = bf.evaluate(obs_sharp)
    _, pass_blur, _ = bf.evaluate(obs_blur)

    lap_sharp = obs_sharp.metrics.laplacian
    lap_blur = obs_blur.metrics.laplacian

    ten_sharp = obs_sharp.metrics.tenengrad
    ten_blur = obs_blur.metrics.tenengrad

    check(
        lap_sharp > lap_blur * 10,
        f"Laplacian drops after blur ({lap_sharp:.1f} -> {lap_blur:.1f})",
    )

    check(
        ten_sharp > ten_blur,
        f"Tenengrad decreases ({ten_sharp:.2f} -> {ten_blur:.2f})",
    )

    check(
        pass_sharp,
        "Sharp image passes blur filter",
    )

    check(
        not pass_blur,
        "Blurred image rejected",
    )


def test_blur_monotonic():

    h = 150
    w = 150

    img = np.zeros((h, w, 3), dtype=np.uint8)

    ys, xs = np.ogrid[:h, :w]

    img[
        (xs - w // 2) ** 2 +
        (ys - h // 2) ** 2
        <= 45 ** 2
    ] = 255

    bf = BlurFilter()

    sigmas = [0, 2, 4, 8, 16]

    laplacians = []

    for sigma in sigmas:

        if sigma == 0:
            blurred = img
        else:
            blurred = gaussian_blur(img, sigma)

        obs = observation(
            blurred,
            np.ones((h, w), np.uint8) * 255,
        )

        bf.evaluate(obs)

        laplacians.append(obs.metrics.laplacian)

    monotonic = all(
        laplacians[i] >= laplacians[i + 1]
        for i in range(len(laplacians) - 1)
    )

    check(
        monotonic,
        f"Laplacian decreases monotonically {laplacians}",
    )


# ============================================================
# AREA
# ============================================================

def test_area():

    h = 100
    w = 100

    img = make_image(h, w)

    large = np.zeros((h, w), np.uint8)
    large[25:75, 25:75] = 255

    tiny = np.zeros((h, w), np.uint8)
    tiny[49:51, 49:51] = 255

    af = AreaFilter(
        minimum_ratio=0.02,
    )

    obs_large = observation(img, large)
    obs_small = observation(img, tiny)

    _, pass_large, _ = af.evaluate(obs_large)
    _, pass_small, _ = af.evaluate(obs_small)

    check(
        pass_large,
        "25% mask passes",
    )

    check(
        not pass_small,
        "0.04% mask rejected",
    )

    check(
        abs(obs_large.metrics.area_ratio - 0.25) < 1e-2,
        f"Area ratio={obs_large.metrics.area_ratio:.4f}",
    )

    check(
        abs(obs_small.metrics.area_ratio - 0.0004) < 1e-4,
        f"Small area={obs_small.metrics.area_ratio:.6f}",
    )


# ============================================================
# BORDER
# ============================================================

def test_border():

    h = 100
    w = 100

    img = make_image(h, w)

    center = np.zeros((h, w), np.uint8)
    center[25:75, 25:75] = 255

    border = np.zeros((h, w), np.uint8)
    border[0:75, 0:50] = 255

    bf = BorderFilter(
        maximum_ratio=0.01,
    )

    obs_center = observation(img, center)
    obs_border = observation(img, border)

    _, pass_center, _ = bf.evaluate(obs_center)
    _, pass_border, _ = bf.evaluate(obs_border)

    expected = (75 + 50 - 1) / (75 * 50)

    check(
        pass_center,
        "Centered object accepted",
    )

    check(
        not pass_border,
        "Border object rejected",
    )

    check(
        abs(obs_border.metrics.border_ratio - expected) < 0.001,
        f"Border ratio={obs_border.metrics.border_ratio:.4f}",
    )


def test_border_corner_case():

    img = make_image(100, 100)

    mask = np.zeros((100, 100), np.uint8)

    mask[0, 0] = 255

    bf = BorderFilter(
        maximum_ratio=0.01,
    )

    obs = observation(img, mask)

    _, passed, _ = bf.evaluate(obs)

    check(
        not passed,
        "Single border pixel rejected",
    )

    check(
        abs(obs.metrics.border_ratio - 1.0) < 0.01,
        f"Border ratio={obs.metrics.border_ratio:.3f}",
    )


def test_border_cutoff_each_edge():

    img = make_image(100, 100)

    bf = BorderFilter(
        maximum_ratio=0.05,
        edge_maximum_ratio=0.25,
    )

    cases = {
        "top": (slice(0, 30), slice(None)),
        "bottom": (slice(70, 100), slice(None)),
        "left": (slice(None), slice(0, 30)),
        "right": (slice(None), slice(70, 100)),
    }

    for edge, (rs, cs) in cases.items():
        mask = np.zeros((100, 100), np.uint8)
        mask[rs, cs] = 255

        obs = observation(img, mask)
        _, passed, _ = bf.evaluate(obs)

        check(
            not passed,
            f"Object cut off at {edge} edge rejected",
        )

        # The ring/area ratio alone is 100/(100*30) ~ 0.033 < 0.05,
        # so it is the per-edge contact ratio that triggers rejection.
        expected_edge = 100 / 100
        attr = f"edge_{edge}_ratio"
        check(
            abs(getattr(obs.metrics, attr) - expected_edge) < 1e-4,
            f"  {attr}={getattr(obs.metrics, attr):.3f}",
        )


def test_border_fully_visible_tangent_accepted():

    img = make_image(100, 100)

    # Circle fully in frame, tangent to the bottom edge.
    mask = make_circle_mask(
        100,
        100,
        radius=30,
        center=(50, 69),
    )

    bf = BorderFilter(
        maximum_ratio=0.05,
        edge_maximum_ratio=0.25,
    )

    obs = observation(img, mask)
    _, passed, _ = bf.evaluate(obs)

    check(
        passed,
        "Fully visible tangent circle accepted",
    )

    check(
        obs.metrics.edge_bottom_ratio < 0.05,
        f"Tangent contact ratio={obs.metrics.edge_bottom_ratio:.3f}",
    )


def test_border_centered_accepted():

    img = make_image(100, 100)

    mask = np.zeros((100, 100), np.uint8)
    mask[25:75, 25:75] = 255

    bf = BorderFilter(
        maximum_ratio=0.05,
        edge_maximum_ratio=0.25,
    )

    obs = observation(img, mask)
    score, passed, _ = bf.evaluate(obs)

    check(passed, "Centered object accepted")
    check(abs(score - 1.0) < 1e-4, f"Centered score={score:.3f}")

    check(
        obs.metrics.edge_ratio == 0.0,
        "Centered object has zero edge contact",
    )


# ============================================================
# OCCLUSION
# ============================================================

def test_occlusion():

    h = 100
    w = 100

    img = make_image(h, w)

    object_mask = np.zeros((h, w), np.uint8)
    object_mask[25:75, 25:75] = 255

    light = np.zeros((h, w), np.uint8)
    light[25:35, 25:35] = 255

    heavy = np.zeros((h, w), np.uint8)
    heavy[25:75, 25:50] = 255

    filt = OcclusionFilter(
        maximum_overlap=0.15,
    )

    none = observation(img, object_mask)
    low = observation(img, object_mask, light)
    high = observation(img, object_mask, heavy)

    _, pass_none, _ = filt.evaluate(none)
    _, pass_low, _ = filt.evaluate(low)
    _, pass_high, _ = filt.evaluate(high)

    check(pass_none, "No hand accepted")

    check(pass_low, "4% overlap accepted")

    check(not pass_high, "50% overlap rejected")

    check(
        abs(high.metrics.hand_overlap - 0.5) < 0.01,
        f"Overlap={high.metrics.hand_overlap:.3f}",
    )


def test_occlusion_monotonic():

    h = 100
    w = 100

    img = make_image(h, w)

    object_mask = np.zeros((h, w), np.uint8)
    object_mask[20:80, 20:80] = 255

    overlaps = []

    filt = OcclusionFilter()

    for width in [0, 10, 20, 30, 40]:

        hand = np.zeros((h, w), np.uint8)

        if width > 0:
            hand[20:80, 20:20 + width] = 255

        obs = observation(
            img,
            object_mask,
            hand if width > 0 else None,
        )

        filt.evaluate(obs)

        overlaps.append(obs.metrics.hand_overlap)

    monotonic = all(
        overlaps[i] <= overlaps[i + 1]
        for i in range(len(overlaps) - 1)
    )

    check(
        monotonic,
        f"Overlap monotonic {overlaps}",
    )


# ============================================================
# COMPLETENESS
# ============================================================

def test_completeness():

    img = make_image()

    circle = make_circle_mask(radius=50)

    flower = make_flower_mask()

    cf = CompletenessFilter(
        minimum_score=0.65,
    )

    obs_circle = observation(img, circle)
    obs_flower = observation(img, flower)

    _, pass_circle, _ = cf.evaluate(obs_circle)
    _, pass_flower, _ = cf.evaluate(obs_flower)

    check(
        pass_circle,
        "Circle passes",
    )

    check(
        not pass_flower,
        "Flower rejected",
    )

    check(
        obs_circle.metrics.solidity > 0.95,
        f"Circle solidity={obs_circle.metrics.solidity:.3f}",
    )

    check(
        obs_circle.metrics.completeness >
        obs_flower.metrics.completeness,
        (
            f"{obs_circle.metrics.completeness:.3f}"
            " > "
            f"{obs_flower.metrics.completeness:.3f}"
        ),
    )


# ============================================================
# OUTLIER / SCORE REJECTION
# ============================================================

class _FakeScoreFilter(BaseFilter):

    def __init__(self, scores, reason="x", passed_by_index=None):
        super().__init__(enabled=True)
        self.scores = scores
        self.reason = reason
        self.passed_by_index = passed_by_index or {}

    def evaluate(self, observation):
        score = self.scores[observation.id]
        passed = self.passed_by_index.get(observation.id, True)
        return score, passed, self.reason


def test_outlier_filter_rejects_extreme_bad():

    from preprocessing.variants import OutlierFilter

    rng = np.random.RandomState(0)
    scores = list(rng.normal(0.5, 0.1, 50))
    scores[49] = -1.0

    inner = _FakeScoreFilter(scores)
    var = OutlierFilter(inner, outlier_z=3.0)

    check(var.need_fitting(), "Outlier filter requires a fit pass")

    pop = [
        observation(make_image(), full_mask(), idx=i)
        for i in range(50)
    ]
    var.fit(pop)

    _, p_bad, r_bad = var.evaluate(pop[49])
    _, p_good, r_good = var.evaluate(pop[0])

    check(
        (not p_bad) and r_bad == "x_outlier",
        f"Extreme bad z rejected, reason={r_bad}",
    )
    check(
        p_good and r_good == "x",
        f"Healthy sample passes, reason={r_good}",
    )


def test_outlier_filter_no_knob_is_noop():

    from preprocessing.variants import OutlierFilter

    inner = _FakeScoreFilter({0: 0.9, 1: 0.2})
    var = OutlierFilter(inner)

    check(not var.need_fitting(), "No fit pass without outlier_z")

    good = observation(make_image(), full_mask(), idx=0)
    low = observation(make_image(), full_mask(), idx=1)

    _, p_good, r_good = var.evaluate(good)
    _, p_low, r_low = var.evaluate(low)

    check(p_good and r_good == "x", f"Healthy passes, reason={r_good}")
    check(p_low and r_low == "x", "No rejection without outlier_z")


def test_outlier_filter_keeps_inner_reason():

    from preprocessing.variants import OutlierFilter

    inner = _FakeScoreFilter({0: 0.9, 1: 0.4}, passed_by_index={1: False})
    var = OutlierFilter(inner, outlier_z=3.0)

    bad = observation(make_image(), full_mask(), idx=1)

    _, p, r = var.evaluate(bad)

    check(
        (not p) and r == "x",
        f"Inner reject reason kept verbatim, reason={r}",
    )


def test_soft_filter_threshold_rejects_below_floor():

    from preprocessing.vincents_area_filter import VincentsAreaFilter

    filt = VincentsAreaFilter(hard_min_area_fraction=0.02)

    big = observation(make_image(), full_mask(), idx=0)
    tiny = observation(make_image(), np.zeros((100, 100), np.uint8), idx=1)

    _, p_big, _ = filt.evaluate(big)
    _, p_tiny, r_tiny = filt.evaluate(tiny)

    check(p_big, "Full mask passes the area floor")
    check(
        (not p_tiny) and r_tiny == "vincents_area_threshold",
        f"Empty mask rejected below floor, reason={r_tiny}",
    )


def test_soft_filter_outlier_rejects_extreme_bad():

    from preprocessing.vincents_area_filter import VincentsAreaFilter

    filt = VincentsAreaFilter(outlier_z=3.0)

    check(filt.need_fitting(), "Soft filter requires a fit pass for outlier mode")

    pop = []
    for i, radius in enumerate([60, 50, 55, 58, 52, 1]):
        mask = np.zeros((100, 100), np.uint8)
        yy, xx = np.ogrid[:100, :100]
        mask[(xx - 50) ** 2 + (yy - 50) ** 2 < radius ** 2] = 255
        pop.append(observation(make_image(), mask, idx=i))

    filt.fit(pop)

    _, p_bad, r_bad = filt.evaluate(pop[5])
    _, p_good, _ = filt.evaluate(pop[0])

    check(
        (not p_bad) and r_bad == "vincents_area_outlier",
        f"Extreme bad z rejected, reason={r_bad}",
    )
    check(p_good, "Typical mask passes the outlier check")


def test_soft_filter_no_knobs_is_noop():

    from preprocessing.vincents_area_filter import VincentsAreaFilter

    filt = VincentsAreaFilter()

    check(not filt.need_fitting(), "No fit pass without outlier_z")

    for i in range(4):
        obs = observation(make_image(), full_mask(), idx=i)
        _, passed, reason = filt.evaluate(obs)
        check(passed, f"Nothing rejected without knobs (obs {i})")
        check(reason == "vincents_area", f"Pass reason={reason}")


def test_pipeline_outlier_integration():

    from preprocessing.variants import OutlierFilter
    from preprocessing.filter_pipeline import FilterPipeline

    inner = _FakeScoreFilter({0: 0.5, 1: 0.55, 2: 0.6, 3: -1.0})
    var = OutlierFilter(inner, outlier_z=3.0)
    pipeline = FilterPipeline([var])

    check(pipeline.need_fitting, "Pipeline flags the outlier fit")

    pop = [
        observation(make_image(), full_mask(), idx=i)
        for i in range(4)
    ]
    pipeline.fit_observations(pop)

    check(pipeline.run(pop[0]), "Healthy passes through pipeline")
    check(not pipeline.run(pop[3]), "Extreme bad score rejected by pipeline")
    check(
        pop[3].rejection_reason == "x_outlier",
        f"Pipeline reason={pop[3].rejection_reason}",
    )


# ============================================================
# exported tests
# ============================================================

FILTER_TESTS = [
    ("Blur filter", test_blur),
    ("Blur monotonicity", test_blur_monotonic),
    ("Area filter", test_area),
    ("Border filter", test_border),
    ("Border corner case", test_border_corner_case),
    ("Occlusion filter", test_occlusion),
    ("Occlusion monotonicity", test_occlusion_monotonic),
    ("Completeness filter", test_completeness),
    ("Outlier filter", test_outlier_filter_rejects_extreme_bad),
    ("Outlier no-op", test_outlier_filter_no_knob_is_noop),
    ("Outlier inner reason", test_outlier_filter_keeps_inner_reason),
    ("Soft filter threshold", test_soft_filter_threshold_rejects_below_floor),
    ("Soft filter outlier", test_soft_filter_outlier_rejects_extreme_bad),
    ("Soft filter no-op", test_soft_filter_no_knobs_is_noop),
    ("Pipeline outlier integration", test_pipeline_outlier_integration),
]