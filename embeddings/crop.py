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