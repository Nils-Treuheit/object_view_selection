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

from preprocessing.blur_filter import BlurFilter
from preprocessing.area_filter import AreaFilter
from preprocessing.border_truncation import BorderFilter
from preprocessing.occlusion_filter import OcclusionFilter
from preprocessing.completeness_filter import CompletenessFilter

from tests.test_utils import (
    check,
    make_image,
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
]