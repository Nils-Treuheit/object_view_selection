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


def _wide_object(obj_color, local_bg):
    """Wide-centred object; grown bbox is 30x70 so padding is vertical."""
    h = w = 100
    img = np.full((h, w, 3), local_bg, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[40:60, 20:80] = 255
    img[40:60, 20:80] = obj_color
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

    img, mask = _wide_object(obj_color=240, local_bg=128)
    out = contrast_input(img, mask, background=0)

    check(out.shape == (224, 224, 3), f"Contrast input 224x224 = {out.shape}")
    check(out[0, 0].tolist() == [0, 0, 0], f"Padding is black bg, got {out[0, 0]}")
    check(out[112, 112].mean() > 200, f"Object preserved at centre, mean={out[112, 112].mean():.1f}")
    check(out[70, 112].mean() > 100, f"Grown margin keeps image content, mean={out[70, 112].mean():.1f}")


def test_contrast_input_white_background():
    from embeddings.crop import contrast_input

    img, mask = _wide_object(obj_color=30, local_bg=200)
    out = contrast_input(img, mask, background=255)

    check(out[0, 0].tolist() == [255, 255, 255], f"Padding is white bg, got {out[0, 0]}")
    check(out[112, 112].mean() < 80, f"Object preserved at centre, mean={out[112, 112].mean():.1f}")
    check(out[70, 112].mean() > 150, f"Grown margin keeps image content, mean={out[70, 112].mean():.1f}")


def test_contrast_input_grows_mask():
    from embeddings.crop import contrast_input

    img, mask = _square_object(240)
    g0 = contrast_input(img, mask, background=255, grow=0)
    g5 = contrast_input(img, mask, background=255, grow=5)

    check(g0[0, :].mean() > 200, "grow=0: object fills crop, no margin")
    check(g5[0, :].mean() < 20, f"grow=5: margin is original image content, not bg, mean={g5[0, :].mean():.1f}")
    check(g5[112, 112].mean() > 200, "Object still present at centre")


def test_contrast_mask_aligns_original_mask():
    from embeddings.crop import contrast_input, contrast_mask

    img, mask = _wide_object(obj_color=240, local_bg=0)
    out = contrast_input(img, mask, background=255, grow=5, size=224)
    cm = contrast_mask(mask, grow=5, size=224)

    check(cm.shape == (224, 224), f"Contrast mask 224x224 = {cm.shape}")
    check(cm[0, 0] == 0, "Top padding is outside the original mask")
    check(cm[112, 112] == 255, "Object centre is inside the original mask")
    frac = (cm == 255).mean()
    check(0.18 < frac < 0.32, f"Original-mask fraction ~0.24, got {frac:.3f}")
    under = out[cm == 255]
    check(under.mean() > 200, f"Input under the original mask is object content, mean={under.mean():.1f}")


def test_contrast_input_none_falls_back_to_legacy():
    from embeddings.crop import contrast_input, padded_square_crop

    img, mask = _square_object(240)
    out = contrast_input(img, mask, background=None)
    legacy = padded_square_crop(img, mask)
    check(np.array_equal(out, legacy), "background=None -> legacy padded_square_crop")


def test_contrast_input_rgba_alpha_values():
    from embeddings.crop import contrast_input

    img, mask = _wide_object(obj_color=240, local_bg=128)
    rgb = contrast_input(img, mask, background=0, rgba=False)
    out = contrast_input(img, mask, background=0, rgba=True)

    check(out.shape == (224, 224, 4), f"RGBA input 224x224x4 = {out.shape}")
    check(np.array_equal(out[..., :3], rgb), "RGBA RGB channels match the RGB output")
    check(abs(int(out[112, 112, 3]) - 255) <= 3, f"Mask alpha ~1.0, got {out[112, 112, 3]}")
    check(abs(int(out[70, 112, 3]) - round(0.8 * 255)) <= 3, f"Margin alpha ~0.8, got {out[70, 112, 3]}")
    check(abs(int(out[0, 0, 3]) - round(0.66 * 255)) <= 3, f"Background alpha ~0.66, got {out[0, 0, 3]}")


def test_contrast_input_rgba_white_background():
    from embeddings.crop import contrast_input

    img, mask = _wide_object(obj_color=30, local_bg=200)
    out = contrast_input(img, mask, background=255, rgba=True)

    check(out.shape == (224, 224, 4), f"RGBA shape = {out.shape}")
    check(abs(int(out[112, 112, 3]) - 255) <= 3, f"Mask alpha ~1.0, got {out[112, 112, 3]}")
    check(abs(int(out[0, 0, 3]) - round(0.66 * 255)) <= 3, f"Background alpha ~0.66, got {out[0, 0, 3]}")
    check(out[0, 0].tolist()[:3] == [255, 255, 255], "RGB unchanged (white bg)")


def test_contrast_input_rgba_empty_mask():
    from embeddings.crop import contrast_input

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    out = contrast_input(img, mask, background=255, rgba=True)

    check(out.shape == (224, 224, 4), f"RGBA empty-mask shape = {out.shape}")
    check(out[..., 3].mean() == round(0.66 * 255), "All-background alpha when mask empty")


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
    ("Contrast mask aligns original mask", test_contrast_mask_aligns_original_mask),
    ("Contrast input legacy fallback", test_contrast_input_none_falls_back_to_legacy),
    ("Contrast input RGBA alpha values", test_contrast_input_rgba_alpha_values),
    ("Contrast input RGBA white bg", test_contrast_input_rgba_white_background),
    ("Contrast input RGBA empty mask", test_contrast_input_rgba_empty_mask),
]
