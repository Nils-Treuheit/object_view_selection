"""
Crop function correctness tests.

Tests:
- bbox_crop
- masked_crop
- padded_square_crop
"""

import numpy as np

from tests.test_utils import check, make_image


def test_bbox_crop():
    from embeddings.crop import bbox_crop

    h, w = 100, 100
    img = make_image(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:70, 20:80] = 255

    crop = bbox_crop(img, mask)

    check(crop.shape == (40, 60, 3), f"Bbox crop (40,60) = {crop.shape}")


def test_masked_crop():
    from embeddings.crop import masked_crop

    h, w = 100, 100
    img = make_image(h, w)

    donut_mask = np.zeros((h, w), dtype=np.uint8)
    donut_mask[25:75, 25:75] = 255
    donut_mask[40:60, 40:60] = 0

    crop = masked_crop(img, donut_mask)

    check(crop.shape == (50, 50, 3), f"Masked crop (50,50) = {crop.shape}")
    check(np.sum(crop[15:35, 15:35]) == 0,
          f"Center hole is black: sum={np.sum(crop[15:35, 15:35])}")


def test_masked_crop_offcenter():
    from embeddings.crop import masked_crop

    h, w = 100, 100
    img = make_image(h, w)

    off_donut = np.zeros((h, w), dtype=np.uint8)
    off_donut[20:80, 30:90] = 255
    off_donut[45:65, 55:75] = 0
    crop_off = masked_crop(img, off_donut)

    check(crop_off.shape == (60, 60, 3), f"Off-center masked crop (60,60) = {crop_off.shape}")
    check(np.sum(crop_off[25:45, 25:45]) == 0,
          f"Off-center hole region is black: sum={np.sum(crop_off[25:45, 25:45])}")


def test_padded_square_crop():
    from embeddings.crop import padded_square_crop

    h, w = 100, 100
    img = make_image(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:70, 20:80] = 255

    crop = padded_square_crop(img, mask)

    check(crop.shape == (224, 224, 3), f"Square crop 224x224 = {crop.shape}")


CROP_TESTS = [
    ("Bbox crop", test_bbox_crop),
    ("Masked crop", test_masked_crop),
    ("Masked crop off-center", test_masked_crop_offcenter),
    ("Padded square crop", test_padded_square_crop),
]
