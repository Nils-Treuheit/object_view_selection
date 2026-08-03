"""
Correctness tests for the Vincent pre-filters (ported from
nit_view_selection/select_best_views.py).

Tests
- VincentEmptyMaskFilter (hard)
- VincentBorderPixelFilter (hard)
- VincentsAreaFilter (soft)
- VincentsArtifactsFilter (soft)
- VincentsMotionBlurFilter (soft)
- robust population scoring helpers
"""

from pathlib import Path

import numpy as np

from data_io.observation import Observation

from preprocessing.vincent_border_pixel import VincentBorderPixelFilter
from preprocessing.vincent_empty_mask import VincentEmptyMaskFilter
from preprocessing.vincent_utils import (
    compute_artifact_mask,
    compute_boundary_blur_variance,
    fit_robust_scores,
    mask_to_foreground,
    one_sided_weight,
    robust_center_scale,
    touches_border_pixels,
)
from preprocessing.vincents_area_filter import VincentsAreaFilter
from preprocessing.vincents_artefacts import VincentsArtifactsFilter
from preprocessing.vincents_motion_blur import VincentsMotionBlurFilter

from tests.test_utils import (
    check,
    make_image,
    gaussian_blur,
    make_circle_mask,
    make_flower_mask,
    make_rectangle_mask,
    full_mask,
    blank_mask,
    edge_image,
)


def observation(image, mask, idx=0):
    return Observation(
        id=idx,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask,
        object_hand=None,
    )


# ============================================================
# HARD FILTERS
# ============================================================

def test_empty_mask_hard():
    f = VincentEmptyMaskFilter()

    empty = observation(make_image(), blank_mask())
    score, passed, reason = f.evaluate(empty)
    check(passed is False, "empty mask is rejected")
    check(reason == "vincent_empty_mask", "empty mask reason is vincent_empty_mask")
    check(score == 0.0, "empty mask score is 0")
    check(empty.metrics.vincent_pixel_count == 0.0, "pixel count recorded for empty mask")

    full = observation(make_image(), full_mask())
    score, passed, reason = f.evaluate(full)
    check(passed is True, "full mask is accepted")
    check(reason == "", "full mask has empty reason")
    check(score == 1.0, "full mask score is 1")
    expected = float(full_mask().shape[0] * full_mask().shape[1])
    check(full.metrics.vincent_pixel_count == expected, "pixel count recorded for full mask")


def test_empty_mask_disabled():
    f = VincentEmptyMaskFilter(enabled=False)
    score, passed, reason = f.evaluate(observation(make_image(), blank_mask()))
    check(passed is True, "disabled empty-mask filter passes everything")
    check(score == 1.0, "disabled empty-mask filter score is 1")


def test_border_pixel_hard():
    f = VincentBorderPixelFilter()

    centered = observation(make_image(), make_circle_mask(200, 200, radius=60))
    score, passed, reason = f.evaluate(centered)
    check(passed is True, "centered mask is accepted")
    check(centered.metrics.vincent_touches_border == 0.0, "centered mask does not touch border")

    touching = observation(
        make_image(),
        make_circle_mask(200, 200, radius=60, center=(0, 100)),
    )
    score, passed, reason = f.evaluate(touching)
    check(passed is False, "mask touching left border is rejected")
    check(reason == "vincent_border_pixel", "border-pixel reason is vincent_border_pixel")
    check(touching.metrics.vincent_touches_border == 1.0, "touches-border flag recorded")


def test_touches_border_pixels():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 5] = 255
    check(touches_border_pixels(mask > 0), "top-row pixel counts as touching")

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255
    check(not touches_border_pixels(mask > 0), "interior pixel does not touch border")


# ============================================================
# SOFT FILTERS: raw stats
# ============================================================

def test_area_filter_raw_stat():
    f = VincentsAreaFilter()

    small = observation(make_image(), make_circle_mask(200, 200, radius=30))
    f.evaluate(small)
    area = 200 * 200
    circle_pixels = np.sum(make_circle_mask(200, 200, radius=30) > 0)
    check(abs(small.metrics.vincent_area_fraction - circle_pixels / area) < 1e-9,
          "area fraction equals pixel_count / canvas_area")

    full = observation(make_image(), full_mask())
    f.evaluate(full)
    check(full.metrics.vincent_area_fraction == 1.0, "full mask area fraction is 1")


def test_artifact_filter_raw_stat():
    f = VincentsArtifactsFilter()

    clean = observation(make_image(), make_circle_mask(200, 200, radius=60))
    f.evaluate(clean)
    check(clean.metrics.vincent_artifact_fraction == 0.0,
          "clean circle has zero artifact fraction")

    noisy = make_circle_mask(200, 200, radius=60).astype(np.uint8)
    rng = np.random.RandomState(0)
    noise = (rng.rand(200, 200) < 0.02).astype(np.uint8) * 255
    noisy[noise > 0] = 255
    noisy_obs = observation(make_image(), noisy)
    f.evaluate(noisy_obs)
    check(noisy_obs.metrics.vincent_artifact_fraction > 0.0,
          "speckled mask has positive artifact fraction")


def test_compute_artifact_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 255
    artifacts = compute_artifact_mask(mask)
    check(bool(artifacts.any()), "isolated speckle counts as artifact")

    clean = make_circle_mask(200, 200, radius=60)
    rng = np.random.RandomState(0)
    noisy = clean.copy()
    noise = (rng.rand(200, 200) < 0.02).astype(np.uint8) * 255
    noisy[noise > 0] = 255

    frac_clean = np.sum(compute_artifact_mask(clean)) / np.sum(clean > 0)
    frac_noisy = np.sum(compute_artifact_mask(noisy)) / np.sum(noisy > 0)
    check(frac_noisy > frac_clean, "noisy mask has higher artifact fraction than clean mask")


def test_motion_blur_raw_stat():
    f = VincentsMotionBlurFilter()

    sharp_img = edge_image(200, 200)
    blurred_img = gaussian_blur(edge_image(200, 200), sigma=8, kernel=41)
    mask = make_rectangle_mask(200, 200, rect_width=180, rect_height=180)

    sharp = observation(sharp_img, mask)
    f.evaluate(sharp)
    blurred = observation(blurred_img, mask)
    f.evaluate(blurred)

    check(sharp.metrics.vincent_boundary_blur_variance > blurred.metrics.vincent_boundary_blur_variance,
          "sharp boundary has higher boundary-blur variance than blurred")


def test_mask_to_foreground():
    binary = mask_to_foreground(make_circle_mask(100, 100))
    check(binary.dtype == np.uint8, "foreground mask is uint8")
    check(binary.max() == 1, "foreground mask values are 0/1")

    rgb_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_mask[10:20, 10:20, :] = 255
    fg = mask_to_foreground(rgb_mask)
    check(fg[15, 15] == 1, "multichannel mask collapses to single channel")


def test_boundary_blur_variance():
    gray_sharp = np.zeros((100, 100), dtype=np.uint8)
    gray_sharp[:, 50:] = 255
    gray_blur = cv2_gaussian(gray_sharp, 6, 25)
    fg = np.zeros((100, 100), dtype=np.uint8)
    fg[10:90, 10:90] = 255

    v_sharp = compute_boundary_blur_variance(gray_sharp, fg, stroke_width=9)
    v_blur = compute_boundary_blur_variance(gray_blur, fg, stroke_width=9)
    check(v_sharp > v_blur, "boundary-blur variance separates sharp from blurred edge")


def cv2_gaussian(gray, sigma, kernel):
    import cv2
    return cv2.GaussianBlur(gray, (kernel, kernel), sigma)


# ============================================================
# ROBUST POPULATION SCORING
# ============================================================

def _populated_observations(area_fractions, image):
    f = VincentsAreaFilter()
    obs_list = []
    for i, frac in enumerate(area_fractions):
        o = observation(image, full_mask(), idx=i)
        o.metrics.vincent_area_fraction = frac
        obs_list.append(o)
    return obs_list


def test_robust_center_scale():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    median, scale = robust_center_scale(values)
    check(median == 2.5, "median is robust center")
    check(scale > 0.0, "robust scale is positive")


def test_one_sided_weight_high_bad():
    values = np.array([0.0, 0.1, 0.5, 1.0])
    median, scale = robust_center_scale(values)
    w = one_sided_weight(values, median, scale, "high_bad", softness=3.0)
    check(w[0] == 1.0, "lowest value (good side) gets full weight")
    check(w[-1] < w[0], "highest value (bad side) gets penalized")
    check(np.all(w <= 1.0) and np.all(w >= 0.0), "weights stay in [0, 1]")


def test_one_sided_weight_low_bad():
    values = np.array([0.0, 0.1, 0.5, 1.0])
    median, scale = robust_center_scale(values)
    w = one_sided_weight(values, median, scale, "low_bad", softness=0.3)
    check(w[-1] == 1.0, "highest value (good side) gets full weight")
    check(w[0] < w[-1], "lowest value (bad side) gets penalized")
    check(np.all(w <= 1.0) and np.all(w >= 0.0), "weights stay in [0, 1]")


def test_fit_robust_scores_small_mask_penalized():
    # area: low_bad; tiny mask should be penalized below full-area mask
    fracs = [0.01, 0.02, 0.5, 0.6, 0.55, 0.58, 0.5, 0.62]
    obs_list = _populated_observations(fracs, make_image())
    fit_robust_scores(obs_list, "vincent_area_fraction", "vincents_area", "low_bad", 0.3)

    tiny = min(obs_list, key=lambda o: o.metrics.vincent_area_fraction)
    large = max(obs_list, key=lambda o: o.metrics.vincent_area_fraction)
    check(tiny.metrics.vincents_area < large.metrics.vincents_area,
          "tiny area gets lower population weight than large area")
    for o in obs_list:
        check(0.0 < o.metrics.vincents_area <= 1.0, "population weights are in (0, 1]")


def test_fit_robust_scores_high_artifact_penalized():
    fracs = [0.01, 0.02, 0.03, 0.02, 0.03, 0.9]
    obs_list = _populated_observations(fracs, make_image())
    fit_robust_scores(obs_list, "vincent_area_fraction", "vincents_area", "high_bad", 3.0)

    bad = max(obs_list, key=lambda o: o.metrics.vincent_area_fraction)
    good = min(obs_list, key=lambda o: o.metrics.vincent_area_fraction)
    check(bad.metrics.vincents_area < good.metrics.vincents_area,
          "high-artifact observation gets lower weight")


def test_vincents_area_end_to_end():
    f = VincentsAreaFilter()
    obs_list = [
        observation(make_image(), make_circle_mask(200, 200, radius=20), idx=i)
        for i in range(5)
    ]
    obs_list += [
        observation(make_image(), make_circle_mask(200, 200, radius=80), idx=i)
        for i in range(5, 12)
    ]
    for o in obs_list:
        f.evaluate(o)
    f.fit_weights(obs_list)

    small = [o.metrics.vincents_area for o in obs_list if o.metrics.vincent_area_fraction < 0.1]
    large = [o.metrics.vincents_area for o in obs_list if o.metrics.vincent_area_fraction > 0.2]
    check(np.mean(small) < np.mean(large), "population weights rank small vs large masks")


def test_vincents_motion_blur_end_to_end():
    f = VincentsMotionBlurFilter()
    sharp_img = edge_image(200, 200)
    blurred_img = gaussian_blur(edge_image(200, 200), sigma=8, kernel=41)
    mask = make_rectangle_mask(200, 200, rect_width=180, rect_height=180)

    obs_list = []
    for i in range(6):
        obs_list.append(observation(sharp_img, mask, idx=i))
    for i in range(6, 12):
        obs_list.append(observation(blurred_img, mask, idx=i))

    for o in obs_list:
        f.evaluate(o)
    f.fit_weights(obs_list)

    sharp_w = [o.metrics.vincents_motion_blur for o in obs_list[:6]]
    blur_w = [o.metrics.vincents_motion_blur for o in obs_list[6:]]
    check(np.mean(sharp_w) > np.mean(blur_w),
          "sharp-boundary observations rank above blurred ones")


def test_soft_filter_disabled():
    f = VincentsAreaFilter(enabled=False)
    o = observation(make_image(), full_mask())
    score, passed, reason = f.evaluate(o)
    check(passed is True, "disabled soft filter passes")
    check(score == 1.0, "disabled soft filter score is 1")
