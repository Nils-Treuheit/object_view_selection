#!/usr/bin/env python3
"""
Shared utilities for correctness tests.

This module provides

- PASS/FAIL accounting
- synthetic image generation
- synthetic masks
- geometric transformations
- common helper assertions
"""

from __future__ import annotations

import cv2
import numpy as np

PASS = 0
FAIL = 0


# ------------------------------------------------------------
# PASS / FAIL
# ------------------------------------------------------------

def check(condition: bool, message: str):
    """Record a pass/fail without raising."""
    global PASS, FAIL

    if condition:
        PASS += 1
        print(f"  [PASS] {message}")
    else:
        FAIL += 1
        print(f"  [FAIL] {message}")


def reset_results():
    global PASS, FAIL
    PASS = 0
    FAIL = 0


def get_results():
    return PASS, FAIL


# ------------------------------------------------------------
# RANDOM IMAGE
# ------------------------------------------------------------

def make_image(height=200, width=200, seed=0):
    """
    Deterministic RGB image.
    """

    rng = np.random.RandomState(seed)

    return (rng.rand(height, width, 3) * 255).astype(np.uint8)


# ------------------------------------------------------------
# BASIC SHAPES
# ------------------------------------------------------------

def make_circle_mask(height=200, width=200, radius=60):
    ys, xs = np.ogrid[:height, :width]

    cx = width // 2
    cy = height // 2

    mask = ((xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2)

    return mask.astype(np.uint8) * 255


def make_rectangle_mask(height=200,
                        width=200,
                        rect_width=100,
                        rect_height=80):
    mask = np.zeros((height, width), np.uint8)

    x0 = width // 2 - rect_width // 2
    y0 = height // 2 - rect_height // 2

    mask[y0:y0 + rect_height,
         x0:x0 + rect_width] = 255

    return mask


def make_square_mask(height=200,
                     width=200,
                     size=80):

    return make_rectangle_mask(
        height,
        width,
        rect_width=size,
        rect_height=size,
    )


def make_triangle_mask(height=200,
                       width=200):

    mask = np.zeros((height, width), np.uint8)

    pts = np.array([
        [width // 2, 30],
        [30, height - 30],
        [width - 30, height - 30]
    ], np.int32)

    cv2.fillPoly(mask, [pts], 255)

    return mask


def make_ring_mask(height=200,
                   width=200,
                   outer_radius=70,
                   inner_radius=30):

    mask = make_circle_mask(height, width, outer_radius)

    hole = make_circle_mask(height, width, inner_radius)

    mask[hole > 0] = 0

    return mask


def make_flower_mask(height=200,
                     width=200,
                     petals=5):

    ys, xs = np.ogrid[:height, :width]

    cx = width // 2
    cy = height // 2

    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    angle = np.arctan2(ys - cy, xs - cx)

    radius = 50 + 20 * np.sin(petals * angle)

    mask = (r <= radius)

    return mask.astype(np.uint8) * 255


def make_crescent_mask(height=200,
                       width=200):

    outer = make_circle_mask(height, width, 60)

    inner = np.zeros_like(outer)

    ys, xs = np.ogrid[:height, :width]

    cx = width // 2 + 20
    cy = height // 2

    inner[((xs - cx) ** 2 + (ys - cy) ** 2 <= 55 ** 2)] = 255

    outer[inner > 0] = 0

    return outer


# ------------------------------------------------------------
# GEOMETRIC TRANSFORMS
# ------------------------------------------------------------

def translate_mask(mask,
                   dx,
                   dy):

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    return cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def rotate_mask(mask,
                angle):

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


def scale_mask(mask,
               scale):

    h, w = mask.shape

    scaled = cv2.resize(
        mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    out = np.zeros_like(mask)

    sh, sw = scaled.shape

    y = (h - sh) // 2
    x = (w - sw) // 2

    if sh <= h and sw <= w:
        out[y:y + sh,
            x:x + sw] = scaled
    else:
        sy = (sh - h) // 2
        sx = (sw - w) // 2

        out = scaled[
            sy:sy + h,
            sx:sx + w,
        ]

    return out


# ------------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------------

def gaussian_blur(image,
                  sigma=10):

    return cv2.GaussianBlur(
        image,
        (31, 31),
        sigma,
    )


def to_grayscale(rgb):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return np.stack(
        [gray, gray, gray],
        axis=-1,
    )


# ------------------------------------------------------------
# NUMERICAL HELPERS
# ------------------------------------------------------------

def max_difference(a,
                   b):

    return float(np.max(np.abs(a - b)))


def l2_distance(a,
                b):

    return float(np.linalg.norm(a - b))


def cosine_similarity(a,
                      b):

    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.dot(a, b) /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


# ------------------------------------------------------------
# RANDOM EMBEDDINGS
# ------------------------------------------------------------

def make_embeddings(
    n=30,
    dim=16,
    seed=0,
):

    rng = np.random.RandomState(seed)

    emb = rng.randn(n, dim).astype(np.float32)

    quality = rng.rand(n).astype(np.float32)

    return emb, quality


# ------------------------------------------------------------
# CLUSTERED EMBEDDINGS
# ------------------------------------------------------------

def make_cluster_embeddings():

    embeddings = np.array([
        [0, 0],
        [0.1, 0],
        [-0.1, 0],
        [10, 0],
        [10.2, 0],
        [-10, 0],
        [-10.1, 0],
    ], dtype=np.float32)

    quality = np.array([
        10,
        9.5,
        9.0,
        1,
        0.5,
        1,
        0.5,
    ], dtype=np.float32)

    return embeddings, quality