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


# ============================================================
# contrast input: static maximum-contrast background
# ============================================================

def _obs(image, mask):
    import types
    return types.SimpleNamespace(image=image, mask=mask)


def _square_object(brightness):
    h = w = 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    img[30:70, 30:70] = brightness
    return img, mask


def test_contrast_background_bright_border_is_black():
    from embeddings.crop import compute_contrast_background

    img, mask = _square_object(240)
    bg = compute_contrast_background([_obs(img, mask)])
    check(bg == 0, f"Bright border -> black bg, got {bg}")


def test_contrast_background_dark_border_is_white():
    from embeddings.crop import compute_contrast_background

    img, mask = _square_object(30)
    bg = compute_contrast_background([_obs(img, mask)])
    check(bg == 255, f"Dark border -> white bg, got {bg}")


def test_contrast_background_aggregates_over_set():
    from embeddings.crop import compute_contrast_background

    bright_img, bright_mask = _square_object(250)
    dark_img, dark_mask = _square_object(20)

    bg = compute_contrast_background([
        _obs(dark_img, dark_mask),
        _obs(bright_img, bright_mask),
    ])
    check(bg == 0, f"Mean border brightness >= 128 -> black, got {bg}")

    bg = compute_contrast_background([
        _obs(dark_img, dark_mask),
        _obs(dark_img, dark_mask),
        _obs(bright_img, bright_mask),
    ])
    check(bg == 255, f"Mean border brightness < 128 -> white, got {bg}")


def test_contrast_input_black_background():
    from embeddings.crop import contrast_input

    img, mask = _square_object(240)
    out = contrast_input(img, mask, background=0)

    check(out.shape == (224, 224, 3), f"Contrast input 224x224 = {out.shape}")
    check(out[0, 0].tolist() == [0, 0, 0], "Background outside grown mask is black")
    check(out[112, 112].mean() > 200, f"Object preserved at centre, mean={out[112, 112].mean():.1f}")


def test_contrast_input_white_background():
    from embeddings.crop import contrast_input

    img, mask = _square_object(30)
    out = contrast_input(img, mask, background=255)

    check(out[0, 0].tolist() == [255, 255, 255], "Background outside grown mask is white")
    check(out[112, 112].mean() < 80, f"Object preserved at centre, mean={out[112, 112].mean():.1f}")


def test_contrast_input_grows_mask():
    from embeddings.crop import contrast_input

    img, mask = _square_object(240)
    g0 = contrast_input(img, mask, background=0, grow=0)
    g5 = contrast_input(img, mask, background=0, grow=5)

    check(g0[0, :].mean() > 200, "grow=0: object touches top edge (no margin)")
    check(g5[0, :].mean() < 20, f"grow=5: background margin at top edge, mean={g5[0, :].mean():.1f}")
    check(g5[112, 112].mean() > 200, "Object still present at centre")


def test_contrast_input_none_falls_back_to_legacy():
    from embeddings.crop import contrast_input, padded_square_crop

    img, mask = _square_object(240)
    out = contrast_input(img, mask, background=None)
    legacy = padded_square_crop(img, mask)
    check(np.array_equal(out, legacy), "background=None -> legacy padded_square_crop")


CROP_TESTS = [
    ("Bbox crop", test_bbox_crop),
    ("Masked crop", test_masked_crop),
    ("Masked crop off-center", test_masked_crop_offcenter),
    ("Padded square crop", test_padded_square_crop),
    ("Contrast bg bright -> black", test_contrast_background_bright_border_is_black),
    ("Contrast bg dark -> white", test_contrast_background_dark_border_is_white),
    ("Contrast bg aggregates set", test_contrast_background_aggregates_over_set),
    ("Contrast input black bg", test_contrast_input_black_background),
    ("Contrast input white bg", test_contrast_input_white_background),
    ("Contrast input grows mask", test_contrast_input_grows_mask),
    ("Contrast input legacy fallback", test_contrast_input_none_falls_back_to_legacy),
]
