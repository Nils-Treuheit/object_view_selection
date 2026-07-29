# tests/test_utils.py
"""
Shared utilities for correctness tests.

Provides:
- PASS/FAIL accounting
- deterministic synthetic images
- binary masks
- geometric shapes
- transformations
- image degradations
- synthetic embeddings
- numerical helpers
"""

from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# TEST RESULT TRACKING
# ============================================================

PASS = 0
FAIL = 0


def check(condition: bool, message: str):
    """
    Record a pass/fail result without raising exceptions.
    """

    global PASS, FAIL

    if condition:
        PASS += 1
        print(f"  [PASS] {message}")
    else:
        FAIL += 1
        print(f"  [FAIL] {message}")


def reset_results():
    """
    Reset global counters.
    """

    global PASS, FAIL

    PASS = 0
    FAIL = 0


def get_results():
    """
    Return current pass/fail counts.
    """

    return PASS, FAIL


# ============================================================
# SYNTHETIC IMAGES
# ============================================================

def make_image(
    height=200,
    width=200,
    seed=0,
):
    """
    Deterministic RGB image.
    """

    rng = np.random.RandomState(seed)

    return (
        rng.rand(height, width, 3) * 255
    ).astype(np.uint8)


def edge_image(
    height=100,
    width=100,
):
    """
    High-frequency stripe image.
    Useful for blur tests.
    """

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[:, ::2] = 255

    return image


def gaussian_blur(
    image,
    sigma=10,
    kernel=31,
):
    """
    Apply Gaussian blur.
    """

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        sigma,
    )


def add_noise(
    image,
    sigma=10,
    seed=0,
):
    """
    Add deterministic Gaussian noise.
    """

    rng = np.random.RandomState(seed)

    noise = rng.normal(
        0,
        sigma,
        image.shape,
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def to_grayscale(
    rgb,
):
    """
    Convert RGB image to 3-channel grayscale.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return np.stack(
        [gray, gray, gray],
        axis=-1,
    )


# ============================================================
# BASIC MASKS
# ============================================================

def blank_mask(
    height=100,
    width=100,
):
    """
    Empty binary mask.
    """

    return np.zeros(
        (height, width),
        dtype=np.uint8,
    )


def full_mask(
    height=100,
    width=100,
):
    """
    Full binary mask.
    """

    return np.ones(
        (height, width),
        dtype=np.uint8,
    ) * 255


# ============================================================
# GEOMETRIC SHAPES
# ============================================================

def make_circle_mask(
    height=200,
    width=200,
    radius=60,
    center=None,
):
    """
    Filled circle.
    """

    if center is None:
        cx = width // 2
        cy = height // 2
    else:
        cx, cy = center

    ys, xs = np.ogrid[:height, :width]

    mask = (
        (xs - cx) ** 2 +
        (ys - cy) ** 2
        <= radius ** 2
    )

    return (
        mask.astype(np.uint8)
        * 255
    )


def circle_mask(
    h=100,
    w=100,
    radius=30,
    center=None,
):
    """
    Backwards-compatible alias.
    """

    return make_circle_mask(
        h,
        w,
        radius,
        center,
    )


def make_rectangle_mask(
    height=200,
    width=200,
    rect_width=100,
    rect_height=80,
):
    """
    Filled rectangle.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    x0 = width // 2 - rect_width // 2
    y0 = height // 2 - rect_height // 2

    mask[
        y0:y0 + rect_height,
        x0:x0 + rect_width,
    ] = 255

    return mask


def rectangle_mask(
    h=100,
    w=100,
    x1=25,
    y1=25,
    x2=75,
    y2=75,
):
    """
    Explicit coordinate rectangle.
    """

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    mask[
        y1:y2,
        x1:x2,
    ] = 255

    return mask


def make_square_mask(
    height=200,
    width=200,
    size=80,
):
    """
    Filled square.
    """

    return make_rectangle_mask(
        height,
        width,
        size,
        size,
    )


def make_triangle_mask(
    height=200,
    width=200,
):
    """
    Filled triangle.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    pts = np.array(
        [
            [width // 2, 30],
            [30, height - 30],
            [width - 30, height - 30],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [pts],
        255,
    )

    return mask


def make_ring_mask(
    height=200,
    width=200,
    outer_radius=70,
    inner_radius=30,
):
    """
    Circle with hole.
    """

    mask = make_circle_mask(
        height,
        width,
        outer_radius,
    )

    hole = make_circle_mask(
        height,
        width,
        inner_radius,
    )

    mask[hole > 0] = 0

    return mask


def make_flower_mask(
    height=200,
    width=200,
    petals=5,
):
    """
    Star/flower shaped mask.
    """

    ys, xs = np.ogrid[:height, :width]

    cx = width // 2
    cy = height // 2

    radius = np.sqrt(
        (xs - cx) ** 2 +
        (ys - cy) ** 2
    )

    angle = np.arctan2(
        ys - cy,
        xs - cx,
    )

    boundary = (
        50 +
        20 *
        np.sin(
            petals * angle
        )
    )

    return (
        radius <= boundary
    ).astype(np.uint8) * 255


def make_crescent_mask(
    height=200,
    width=200,
):
    """
    Crescent shape.
    """

    outer = make_circle_mask(
        height,
        width,
        60,
    )

    inner = np.zeros_like(
        outer
    )

    ys, xs = np.ogrid[
        :height,
        :width,
    ]

    cx = width // 2 + 20
    cy = height // 2

    inner[
        (xs - cx) ** 2 +
        (ys - cy) ** 2
        <= 55 ** 2
    ] = 255

    outer[inner > 0] = 0

    return outer


# ============================================================
# MASK TRANSFORMATIONS
# ============================================================

def translate_mask(
    mask,
    dx,
    dy,
):
    """
    Translate mask.
    """

    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    return cv2.warpAffine(
        mask,
        matrix,
        (
            mask.shape[1],
            mask.shape[0],
        ),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def shifted_mask(
    mask,
    dx,
    dy,
):
    """
    Alias for translation.
    """

    return translate_mask(
        mask,
        dx,
        dy,
    )


def rotate_mask(
    mask,
    angle,
):
    """
    Rotate binary mask.
    """

    h, w = mask.shape

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def scale_mask(
    mask,
    scale,
):
    """
    Scale mask around center.
    """

    h, w = mask.shape

    resized = cv2.resize(
        mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    result = np.zeros_like(mask)

    rh, rw = resized.shape

    y0 = max(
        0,
        (h - rh) // 2,
    )

    x0 = max(
        0,
        (w - rw) // 2,
    )

    y1 = min(
        h,
        y0 + rh,
    )

    x1 = min(
        w,
        x0 + rw,
    )

    result[
        y0:y1,
        x0:x1,
    ] = resized[
        :y1-y0,
        :x1-x0,
    ]

    return result


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def max_difference(
    a,
    b,
):
    return float(
        np.max(
            np.abs(a - b)
        )
    )


def l2_distance(
    a,
    b,
):
    return float(
        np.linalg.norm(
            a - b
        )
    )


def cosine_similarity(
    a,
    b,
):
    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


# ============================================================
# EMBEDDING GENERATORS
# ============================================================

def make_embeddings(
    n=30,
    dim=16,
    seed=0,
):
    """
    Random embedding matrix + quality scores.
    """

    rng = np.random.RandomState(seed)

    embeddings = rng.randn(
        n,
        dim,
    ).astype(np.float32)

    quality = rng.rand(
        n,
    ).astype(np.float32)

    return embeddings, quality


def random_embeddings(
    n=20,
    dim=8,
    seed=42,
):
    """
    Backwards-compatible embedding helper.
    """

    rng = np.random.RandomState(seed)

    return rng.randn(
        n,
        dim,
    ).astype(np.float32)


def random_quality(
    n=20,
    seed=42,
):
    """
    Backwards-compatible quality helper.
    """

    rng = np.random.RandomState(seed)

    return rng.rand(
        n,
    ).astype(np.float32)


def make_cluster_embeddings():
    """
    Three-cluster synthetic embedding space.
    """

    embeddings = np.array(
        [
            [0, 0],
            [0.1, 0],
            [-0.1, 0],
            [10, 0],
            [10.2, 0],
            [-10, 0],
            [-10.1, 0],
        ],
        dtype=np.float32,
    )

    quality = np.array(
        [
            10,
            9.5,
            9.0,
            1,
            0.5,
            1,
            0.5,
        ],
        dtype=np.float32,
    )

    return embeddings, quality