# tests/test_utils.py
"""
Synthetic data and test utility functions.

Shared helpers for correctness tests:
- synthetic images
- geometric masks
- transformations
- deterministic random generators
"""

import numpy as np
import cv2


def random_image(
    h=100,
    w=100,
    seed=0
):
    """
    Deterministic RGB image.
    """

    rng = np.random.RandomState(seed)

    return (
        rng.rand(h, w, 3) * 255
    ).astype(np.uint8)


def blank_mask(
    h=100,
    w=100
):
    """
    Empty binary mask.
    """

    return np.zeros(
        (h, w),
        dtype=np.uint8
    )


def full_mask(
    h=100,
    w=100
):
    """
    Full binary mask.
    """

    return np.ones(
        (h, w),
        dtype=np.uint8
    ) * 255


def circle_mask(
    h=100,
    w=100,
    radius=30,
    center=None
):
    """
    Filled circle mask.
    """

    if center is None:
        cx = w // 2
        cy = h // 2
    else:
        cx, cy = center

    yy, xx = np.ogrid[
        :h,
        :w
    ]

    mask = (
        (xx - cx) ** 2 +
        (yy - cy) ** 2
        <= radius ** 2
    )

    return (
        mask.astype(np.uint8)
        * 255
    )


def rectangle_mask(
    h=100,
    w=100,
    x1=25,
    y1=25,
    x2=75,
    y2=75
):
    """
    Filled rectangle mask.
    """

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    mask[
        y1:y2,
        x1:x2
    ] = 255

    return mask


def shifted_mask(
    mask,
    dx,
    dy
):
    """
    Translate mask without changing size.
    """

    h, w = mask.shape

    shifted = np.zeros_like(mask)

    src_x1 = max(0, -dx)
    src_x2 = min(
        w,
        w - dx
    )

    src_y1 = max(0, -dy)
    src_y2 = min(
        h,
        h - dy
    )

    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (
        src_x2 - src_x1
    )

    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (
        src_y2 - src_y1
    )

    shifted[
        dst_y1:dst_y2,
        dst_x1:dst_x2
    ] = mask[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    return shifted


def rotate_mask(
    mask,
    angle
):
    """
    Rotate binary mask around center.
    """

    h, w = mask.shape

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST
    )

    return rotated


def scale_mask(
    mask,
    scale
):
    """
    Resize mask around center.
    """

    h, w = mask.shape

    resized = cv2.resize(
        mask,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    result = np.zeros_like(mask)

    rh, rw = resized.shape

    y0 = max(
        0,
        (h - rh) // 2
    )

    x0 = max(
        0,
        (w - rw) // 2
    )

    y1 = min(
        h,
        y0 + rh
    )

    x1 = min(
        w,
        x0 + rw
    )

    result[
        y0:y1,
        x0:x1
    ] = resized[
        :y1-y0,
        :x1-x0
    ]

    return result


def add_noise(
    image,
    sigma=10,
    seed=0
):
    """
    Add Gaussian noise.
    """

    rng = np.random.RandomState(seed)

    noise = rng.normal(
        0,
        sigma,
        image.shape
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


def blur_image(
    image,
    kernel=31,
    sigma=10
):
    """
    Gaussian blur helper.
    """

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        sigma
    )


def edge_image(
    h=100,
    w=100
):
    """
    Synthetic high-frequency image.
    """

    image = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    image[:, ::2] = 255

    return image


def random_embeddings(
    n=20,
    dim=8,
    seed=42
):
    """
    Deterministic embedding matrix.
    """

    rng = np.random.RandomState(seed)

    return rng.randn(
        n,
        dim
    ).astype(np.float32)


def random_quality(
    n=20,
    seed=42
):
    """
    Deterministic quality scores.
    """

    rng = np.random.RandomState(seed)

    return rng.rand(
        n
    ).astype(np.float32)