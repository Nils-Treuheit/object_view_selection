import numpy as np
from skimage.transform import resize


def _zernike_radial(n, m, r):
    from math import factorial
    radial = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        coeff = ((-1) ** k
                 * factorial(n - k)
                 / (factorial(k)
                    * factorial((n + abs(m)) // 2 - k)
                    * factorial((n - abs(m)) // 2 - k)))
        radial += coeff * r ** (n - 2 * k)
    return radial


def zernike_moments(mask: np.ndarray, radius: int = 64, degree: int = 10) -> np.ndarray:
    from math import factorial
    binary = (mask > 0).astype(np.float64)
    h, w = binary.shape
    s = min(h, w)
    resized = resize(binary, (s, s), anti_aliasing=False, preserve_range=True)

    ny, nx = s, s
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    mask_circle = r <= 1
    r = r[mask_circle]
    theta = theta[mask_circle]
    pixel_vals = resized[mask_circle]

    moments = []
    for n in range(degree + 1):
        for m in range(-n, n + 1, 2):
            if (n - abs(m)) % 2 != 0:
                continue
            radial = _zernike_radial(n, m, r)
            if m >= 0:
                z = radial * np.cos(m * theta)
            else:
                z = radial * np.sin(-m * theta)
            Z = np.sum(z * pixel_vals) * (n + 1) / np.pi
            moments.append(abs(Z))

    return np.array(moments[1:])