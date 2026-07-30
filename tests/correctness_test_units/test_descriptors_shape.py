#!/usr/bin/env python3
"""
tests/test_descriptors_shape.py

Correctness tests for contour-based descriptors:
    - Fourier Descriptors
    - Shape Context

These tests verify mathematical properties such as determinism,
normalization, invariance, and shape discrimination.

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
# FOURIER DESCRIPTORS
# ---------------------------------------------------------------------

def test_fourier_dimension():
    from descriptors.fourier import fourier_descriptors

    fd = fourier_descriptors(
        make_circle_mask(),
        num_descriptors=32,
    )

    check(
        len(fd) == 32,
        f"Fourier descriptor length = {len(fd)}",
    )


def test_fourier_finite():
    from descriptors.fourier import fourier_descriptors

    fd = fourier_descriptors(
        make_circle_mask(),
        num_descriptors=32,
    )

    check(
        np.all(np.isfinite(fd)),
        "Fourier descriptors are finite",
    )

    check(
        np.all(fd >= 0),
        "Fourier descriptors are non-negative",
    )


def test_fourier_normalization():
    from descriptors.fourier import fourier_descriptors

    fd = fourier_descriptors(
        make_circle_mask(),
        num_descriptors=32,
    )

    check(
        abs(fd.sum() - 1.0) < 1e-2,
        f"L1 normalization (sum={fd.sum():.6f})",
    )


def test_fourier_deterministic():
    from descriptors.fourier import fourier_descriptors

    mask = make_circle_mask()

    fd1 = fourier_descriptors(mask, num_descriptors=32)
    fd2 = fourier_descriptors(mask, num_descriptors=32)

    check(
        np.allclose(fd1, fd2),
        "Fourier deterministic",
    )


def test_fourier_translation():
    from descriptors.fourier import fourier_descriptors

    base = make_circle_mask(radius=30)
    shifted = make_circle_mask(radius=30, center=(30, 40))

    fd1 = fourier_descriptors(base, num_descriptors=32)
    fd2 = fourier_descriptors(shifted, num_descriptors=32)

    diff = np.max(np.abs(fd1 - fd2))

    check(
        np.allclose(fd1, fd2, atol=1e-2),
        f"Translation invariant (max diff={diff:.6f})",
    )


def test_fourier_rotation():
    from descriptors.fourier import fourier_descriptors

    base = make_circle_mask(radius=30)
    rotated = np.rot90(base)

    fd1 = fourier_descriptors(base, num_descriptors=32)
    fd2 = fourier_descriptors(rotated, num_descriptors=32)

    diff = np.max(np.abs(fd1 - fd2))

    check(
        np.allclose(fd1, fd2, atol=5e-2),
        f"Rotation invariant (max diff={diff:.6f})",
    )


def test_fourier_scale():
    from descriptors.fourier import fourier_descriptors

    r40 = make_circle_mask(radius=40)
    r50 = make_circle_mask(radius=50)

    fd1 = fourier_descriptors(r40, num_descriptors=32)
    fd2 = fourier_descriptors(r50, num_descriptors=32)

    diff = np.max(np.abs(fd1 - fd2))

    check(
        np.allclose(fd1, fd2, atol=5e-2),
        f"Scale invariant (max diff={diff:.6f})",
    )


def test_fourier_shape_difference():
    from descriptors.fourier import fourier_descriptors

    circle = make_circle_mask(radius=30)
    rectangle = make_rectangle_mask()

    fd_circle = fourier_descriptors(circle, num_descriptors=32)
    fd_rect = fourier_descriptors(rectangle, num_descriptors=32)

    diff = np.max(np.abs(fd_circle - fd_rect))

    check(
        not np.allclose(fd_circle, fd_rect, atol=5e-2),
        f"Circle and rectangle differ (max diff={diff:.6f})",
    )


# ---------------------------------------------------------------------
# SHAPE CONTEXT
# ---------------------------------------------------------------------

def test_shape_context_dimension():
    from descriptors.shape_context import shape_context_descriptor

    sc = shape_context_descriptor(make_circle_mask())

    check(
        len(sc) > 0,
        f"Shape Context dimension = {len(sc)}",
    )


def test_shape_context_normalization():
    from descriptors.shape_context import shape_context_descriptor

    sc = shape_context_descriptor(make_circle_mask())

    check(
        abs(sc.sum() - 1.0) < 1e-2,
        f"Histogram normalized (sum={sc.sum():.6f})",
    )


def test_shape_context_nonnegative():
    from descriptors.shape_context import shape_context_descriptor

    sc = shape_context_descriptor(make_circle_mask())

    check(
        np.all(sc >= 0),
        "Shape Context histogram non-negative",
    )


def test_shape_context_finite():
    from descriptors.shape_context import shape_context_descriptor

    sc = shape_context_descriptor(make_circle_mask())

    check(
        np.all(np.isfinite(sc)),
        "Shape Context values are finite",
    )


def test_shape_context_deterministic():
    from descriptors.shape_context import shape_context_descriptor

    mask = make_circle_mask()

    sc1 = shape_context_descriptor(mask)
    sc2 = shape_context_descriptor(mask)

    check(
        np.allclose(sc1, sc2),
        "Shape Context deterministic",
    )


def test_shape_context_translation():
    from descriptors.shape_context import shape_context_descriptor

    base = make_circle_mask(radius=30)
    shifted = make_circle_mask(radius=30, center=(30, 40))

    sc1 = shape_context_descriptor(base)
    sc2 = shape_context_descriptor(shifted)

    diff = np.max(np.abs(sc1 - sc2))

    check(
        np.allclose(sc1, sc2, atol=5e-2),
        f"Translation invariant (max diff={diff:.6f})",
    )


def test_shape_context_rotation():
    from descriptors.shape_context import shape_context_descriptor

    base = make_circle_mask(radius=30)
    rotated = np.rot90(base)

    sc1 = shape_context_descriptor(base)
    sc2 = shape_context_descriptor(rotated)

    diff = np.max(np.abs(sc1 - sc2))

    check(
        np.allclose(sc1, sc2, atol=0.1),
        f"Approximately rotation invariant (max diff={diff:.6f})",
    )


def test_shape_context_shape_difference():
    from descriptors.shape_context import shape_context_descriptor

    circle = make_circle_mask(radius=30)
    rectangle = make_rectangle_mask()

    sc_circle = shape_context_descriptor(circle)
    sc_rect = shape_context_descriptor(rectangle)

    diff = np.max(np.abs(sc_circle - sc_rect))

    check(
        not np.allclose(sc_circle, sc_rect, atol=0.08),
        f"Circle and rectangle differ (max diff={diff:.6f})",
    )


# ---------------------------------------------------------------------
# Export list
# ---------------------------------------------------------------------

DESCRIPTOR_TESTS = [
    ("Fourier dimension", test_fourier_dimension),
    ("Fourier finite", test_fourier_finite),
    ("Fourier normalization", test_fourier_normalization),
    ("Fourier deterministic", test_fourier_deterministic),
    ("Fourier translation", test_fourier_translation),
    ("Fourier rotation", test_fourier_rotation),
    ("Fourier scale", test_fourier_scale),
    ("Fourier shape discrimination", test_fourier_shape_difference),
    ("Shape Context dimension", test_shape_context_dimension),
    ("Shape Context normalization", test_shape_context_normalization),
    ("Shape Context non-negative", test_shape_context_nonnegative),
    ("Shape Context finite", test_shape_context_finite),
    ("Shape Context deterministic", test_shape_context_deterministic),
    ("Shape Context translation", test_shape_context_translation),
    ("Shape Context rotation", test_shape_context_rotation),
    ("Shape Context shape discrimination", test_shape_context_shape_difference),
]