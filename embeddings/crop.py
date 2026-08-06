import cv2
import numpy as np

ALPHA_MASK = 1.0
ALPHA_MARGIN = 0.8
ALPHA_BACKGROUND = 0.66


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
                   grow: int = 5, size: int = 224, rgba: bool = False) -> np.ndarray:
    """Grown-mask cut-out of the object placed on a solid background.

    The crop extent is the object's bounding box grown by ``grow`` px on every
    side. The cut-out is the image masked by the *grown* mask, so the object
    plus a small local-context margin keeps its original pixels; everything
    outside the grown mask (square padding) is filled with ``background``
    (0 = black or 255 = white). The result is square-padded with the same
    background and resized to ``size`` x ``size``. With ``background=None``
    this falls back to the legacy ``padded_square_crop`` (raw crop, zero
    padding) so models keep their previous behaviour until a background is
    set.

    With ``rgba=True`` the output gains an alpha channel (0..255) encoding the
    region type: 255 over the original mask, ~204 (0.8) over the grown cut-out
    margin that keeps original image content, and ~168 (0.66) over the static
    background. RGB values are identical either way, so models that can ingest
    an extra channel may use ``rgba=True`` while RGB-only models keep
    ``rgba=False`` with no change.
    """
    if background is None:
        rgb = padded_square_crop(image, mask, size=size)
        if not rgba:
            return rgb
        a = _legacy_alpha(mask, size=size)
        return np.dstack([rgb, a])

    grown = grow_mask(mask, grow=grow)
    ys, xs = np.where(grown > 0)
    if len(ys) == 0:
        rgb = np.full((size, size, 3), background, dtype=np.uint8)
        if not rgba:
            return rgb
        a = np.full((size, size), 255 * ALPHA_BACKGROUND, dtype=np.uint8)
        return np.dstack([rgb, a])

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    obj = np.asarray(image, dtype=float)[y1:y2, x1:x2]
    g = grown[y1:y2, x1:x2] > 0
    m = np.asarray(mask)[y1:y2, x1:x2] > 0
    if obj.ndim == 3:
        rgb_crop = np.where(g[..., None], obj, float(background))
    else:
        rgb_crop = np.where(g, obj, float(background))
    alpha_crop = np.where(m, ALPHA_MASK, np.where(g, ALPHA_MARGIN, ALPHA_BACKGROUND))

    h, w = rgb_crop.shape[:2]
    s = max(h, w)
    y_off = (s - h) // 2
    x_off = (s - w) // 2

    rgb_square = np.full((s, s, 3), background, dtype=np.uint8)
    if rgb_crop.ndim == 2:
        rgb_square[y_off:y_off + h, x_off:x_off + w] = rgb_crop[..., None]
    else:
        rgb_square[y_off:y_off + h, x_off:x_off + w] = rgb_crop

    rgb = _resize_rgb(rgb_square, size)
    if not rgba:
        return rgb

    alpha_square = np.full((s, s), ALPHA_BACKGROUND, dtype=np.float32)
    alpha_square[y_off:y_off + h, x_off:x_off + w] = alpha_crop
    a = _resize_alpha(alpha_square, size)
    return np.dstack([rgb, a])


def _resize_rgb(square: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(square)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def _resize_alpha(square: np.ndarray, size: int) -> np.ndarray:
    """Resize a float32 alpha map, returning uint8 (0..255).

    BILINEAR is used (not LANCZOS) so the resampled values stay inside the
    source range (0.66..1.0) instead of ringing below it along the mask edge.
    """
    from PIL import Image
    pil = Image.fromarray(np.clip(square, 0.0, 1.0).astype(np.float32))
    pil = pil.resize((size, size), Image.BILINEAR)
    return (np.array(pil, dtype=np.float32) * 255).round().astype(np.uint8)


def _legacy_alpha(mask: np.ndarray, size: int = 224) -> np.ndarray:
    """Alpha for the legacy ``padded_square_crop`` fallback: mask 1.0, rest 0.66."""
    m = np.asarray(mask) > 0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return np.full((size, size), 255 * ALPHA_BACKGROUND, dtype=np.uint8)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = np.where(m[y1:y2, x1:x2], ALPHA_MASK, ALPHA_BACKGROUND).astype(np.float32)
    h, w = crop.shape[:2]
    s = max(h, w)
    square = np.full((s, s), ALPHA_BACKGROUND, dtype=np.float32)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop
    return _resize_alpha(square, size)


def contrast_mask(mask: np.ndarray, grow: int = 5, size: int = 224) -> np.ndarray:
    """Original (not grown) mask aligned to ``contrast_input``'s geometry.

    Returns a 0/255 mask in the same ``size`` x ``size`` frame as
    ``contrast_input`` (grown bbox crop, centred, squared, resized) so it can
    be overlaid directly on the embedding input. Only the original mask pixels
    are 255; the grown margin and padding are 0.
    """
    grown = grow_mask(mask, grow=grow)
    ys, xs = np.where(grown > 0)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    m = (np.asarray(mask)[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    s = max(h, w)
    square = np.zeros((s, s), dtype=np.uint8)
    y_off = (s - h) // 2
    x_off = (s - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = m

    from PIL import Image
    pil = Image.fromarray(square)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)