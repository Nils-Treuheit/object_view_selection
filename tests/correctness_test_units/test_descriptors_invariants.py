#!/usr/bin/env python3
"""
tests/test_descriptors_invariants.py

Correctness tests for invariant descriptors:
    - Hu Moments
    - Zernike Moments

These tests verify mathematical invariance properties rather than exact
numerical values.

Exports:
    DESCRIPTOR_TESTS
"""

import numpy as np

from tests.test_utils import check


# ---------------------------------------------------------------------
# Synthetic masks
# ---------------------------------------------------------------------

def make_circle_mask(h=100, w=100, radius=30, center=None):
    ys, xs = np.ogrid[:h, :w]

    if center is None:
        cx = w // 2
        cy = h // 2
    else:
        cx, cy = center

    return (
        ((xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2)
        .astype(np.uint8)
        * 255
    )


def make_rectangle_mask(h=100, w=100):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


# ---------------------------------------------------------------------
# HU MOMENTS
# ---------------------------------------------------------------------

def test_hu_translation():
    from descriptors.hu import hu_moments

    base = make_circle_mask(radius=30)
    shifted = make_circle_mask(radius=30, center=(30, 40))

    hu1 = hu_moments(base)
    hu2 = hu_moments(shifted)

    diff = np.max(np.abs(hu1 - hu2))

    check(
        np.allclose(hu1, hu2, atol=1e-3),
        f"Hu translation invariant (max diff={diff:.6f})",
    )


def test_hu_rotation():
    from descriptors.hu import hu_moments

    base = make_circle_mask(radius=30)
    rotated = np.rot90(base)

    hu1 = hu_moments(base)
    hu2 = hu_moments(rotated)

    diff = np.max(np.abs(hu1 - hu2))

    check(
        np.allclose(hu1, hu2, atol=0.1),
        f"Hu rotation invariant (max diff={diff:.6f})",
    )


def test_hu_scale():
    from descriptors.hu import hu_moments

    big = make_circle_mask(radius=30)
    small = make_circle_mask(radius=15)

    hu_big = hu_moments(big)
    hu_small = hu_moments(small)

    diff = np.max(np.abs(hu_big - hu_small))

    check(
        np.allclose(hu_big, hu_small, atol=0.01),
        f"Hu scale invariant (max diff={diff:.6f})",
    )


def test_hu_shape_difference():
    from descriptors.hu import hu_moments

    circle = make_circle_mask(radius=30)
    rectangle = make_rectangle_mask()

    hu_circle = hu_moments(circle)
    hu_rect = hu_moments(rectangle)

    diff = np.max(np.abs(hu_circle - hu_rect))

    check(
        not np.allclose(hu_circle, hu_rect, atol=0.04),
        f"Circle and rectangle differ (max diff={diff:.6f})",
    )


def test_hu_properties():
    from descriptors.hu import hu_moments

    hu = hu_moments(make_circle_mask())

    check(
        len(hu) == 7,
        "Hu returns 7 moments",
    )

    check(
        np.all(np.isfinite(hu)),
        "Hu moments are finite",
    )


# ---------------------------------------------------------------------
# ZERNIKE MOMENTS
# ---------------------------------------------------------------------

def test_zernike_dimension():
    from descriptors.zernike import zernike_moments

    z = zernike_moments(make_circle_mask(), degree=6)

    check(
        len(z) > 0,
        f"Zernike descriptor length = {len(z)}",
    )


def test_zernike_finite():
    from descriptors.zernike import zernike_moments

    z = zernike_moments(make_circle_mask(), degree=6)

    check(
        np.all(np.isfinite(z)),
        "Zernike values are finite",
    )

    check(
        np.all(z >= 0),
        "Zernike magnitudes are non-negative",
    )


def test_zernike_deterministic():
    from descriptors.zernike import zernike_moments

    mask = make_circle_mask()

    z1 = zernike_moments(mask, degree=6)
    z2 = zernike_moments(mask, degree=6)

    check(
        np.allclose(z1, z2, atol=1e-6),
        "Zernike deterministic",
    )


def test_zernike_rotation():
    from descriptors.zernike import zernike_moments

    base = make_circle_mask(radius=30)
    rotated = np.rot90(base)

    z1 = zernike_moments(base, degree=6)
    z2 = zernike_moments(rotated, degree=6)

    diff = np.max(np.abs(z1 - z2))

    check(
        np.allclose(z1, z2, atol=0.01),
        f"Zernike rotation invariant (max diff={diff:.6f})",
    )


def test_zernike_scale():
    from descriptors.zernike import zernike_moments

    big = make_circle_mask(radius=30)
    small = make_circle_mask(radius=15)

    z_big = zernike_moments(big, degree=6)
    z_small = zernike_moments(small, degree=6)

    diff = np.max(np.abs(z_big - z_small))

    check(
        diff > 0.1,
        f"Zernike differs at different scales (max diff={diff:.6f})",
    )


def test_zernike_shape_difference():
    from descriptors.zernike import zernike_moments

    circle = make_circle_mask(radius=30)
    rectangle = make_rectangle_mask()

    z_circle = zernike_moments(circle, degree=6)
    z_rect = zernike_moments(rectangle, degree=6)

    diff = np.max(np.abs(z_circle - z_rect))

    check(
        not np.allclose(z_circle, z_rect, atol=0.1),
        f"Circle and rectangle produce different Zernike (max diff={diff:.6f})",
    )


# ---------------------------------------------------------------------
# Export list
# ---------------------------------------------------------------------

DESCRIPTOR_TESTS = [
    ("Hu translation", test_hu_translation),
    ("Hu rotation", test_hu_rotation),
    ("Hu scale", test_hu_scale),
    ("Hu shape discrimination", test_hu_shape_difference),
    ("Hu properties", test_hu_properties),
    ("Zernike dimension", test_zernike_dimension),
    ("Zernike finiteness", test_zernike_finite),
    ("Zernike deterministic", test_zernike_deterministic),
    ("Zernike rotation", test_zernike_rotation),
    ("Zernike scale", test_zernike_scale),
    ("Zernike shape discrimination", test_zernike_shape_difference),
]