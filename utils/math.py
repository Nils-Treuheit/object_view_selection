import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-10:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    ex = np.exp((x - x.max()) / temperature)
    return ex / ex.sum()