import cv2
import numpy as np


def bbox_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        y1, y2, x1, x2 = 0, image.shape[0], 0, image.shape[1]
    else:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
    return image[y1:y2, x1:x2]


def masked_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    crop = bbox_crop(image, mask)
    mask_crop = bbox_crop(mask, mask)
    if crop.ndim == 3:
        return crop * (mask_crop[..., None] > 0)
    return crop * (mask_crop > 0)


def padded_square_crop(image: np.ndarray, mask: np.ndarray, size: int = 224) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    crop = image[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    s = max(h, w)

    square = np.zeros((s, s, 3), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop

    from PIL import Image
    pil = Image.fromarray(square)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def grow_mask(mask: np.ndarray, grow: int = 5) -> np.ndarray:
    """Dilate a binary mask by ``grow`` pixels on every side."""
    if grow <= 0:
        return np.asarray(mask).astype(np.uint8)
    kernel = np.ones((2 * grow + 1, 2 * grow + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask).astype(np.uint8), kernel)


def compute_contrast_background(observations, ring: int = 2, threshold: float = 128) -> int:
    """Pick the maximum-contrast static background colour over the whole set.

    Samples the object's outer rim (mask pixels within ``ring`` px of the
    edge) from each frame and averages the brightness over every observation.
    A mostly bright border set maps to 0 (black background, super-dark
    contrast); a mostly dark border set maps to 255 (white background,
    super-bright contrast). Falls back to 0 when no border pixels are found.
    """
    values = []
    kernel = np.ones((2 * ring + 1, 2 * ring + 1), dtype=np.uint8)
    for obs in observations:
        image = getattr(obs, "image", None)
        mask = getattr(obs, "mask", None)
        if image is None or mask is None:
            continue
        m = np.asarray(mask) > 0
        if not m.any():
            continue
        ring_mask = m & ~(cv2.erode(m.astype(np.uint8), kernel) > 0)
        if not ring_mask.any():
            continue
        values.append(np.asarray(image, dtype=float)[ring_mask].mean())
    if not values:
        return 0
    return 0 if np.mean(values) >= threshold else 255


def contrast_input(image: np.ndarray, mask: np.ndarray, background,
                   grow: int = 5, size: int = 224) -> np.ndarray:
    """Grown-mask crop of the object composited onto a solid background.

    The crop extent is the object's bounding box grown by ``grow`` px on every
    side; object pixels (original mask) keep their content, everything else in
    the crop is replaced by ``background`` (0 = black or 255 = white). The
    result is square-padded with the same background and resized to ``size`` x
    ``size``. With ``background=None`` this falls back to the legacy
    ``padded_square_crop`` (raw crop, zero padding) so models keep their
    previous behaviour until a background is set.
    """
    if background is None:
        return padded_square_crop(image, mask, size=size)

    grown = grow_mask(mask, grow=grow)
    ys, xs = np.where(grown > 0)
    if len(ys) == 0:
        return np.full((size, size, 3), background, dtype=np.uint8)

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    obj = np.asarray(image, dtype=float)[y1:y2, x1:x2]
    m = np.asarray(mask)[y1:y2, x1:x2] > 0
    if obj.ndim == 3:
        comp = np.where(m[..., None], obj, float(background))
    else:
        comp = np.where(m, obj, float(background))

    h, w = comp.shape[:2]
    s = max(h, w)
    square = np.full((s, s, 3), background, dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    if comp.ndim == 2:
        square[y_off:y_off + h, x_off:x_off + w] = comp[..., None]
    else:
        square[y_off:y_off + h, x_off:x_off + w] = comp

    from PIL import Image
    pil = Image.fromarray(square)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)