"""
Smoke tests for utility modules.
"""

import numpy as np

from tests.smoke_test_utils import check, first_usable_observation


def test_utils(ds):
    obs = first_usable_observation(ds)

    from utils.geometry import contour_area, bounding_box, mask_centroid
    check(contour_area(obs.mask) > 0, "Contour area")
    bb = bounding_box(obs.mask)
    check(len(bb) == 4, f"Bounding box={bb}")
    cy, cx = mask_centroid(obs.mask)
    check(0 <= cy < obs.mask.shape[0], f"Centroid y={cy:.1f}")

    from utils.math import normalize, softmax
    n = normalize(np.array([1., 2., 3., 4., 5.]))
    check(abs(n.min()) < 1e-6 and abs(n.max() - 1) < 1e-6, "Normalize")
    sm = softmax(np.array([1., 2., 3.]))
    check(abs(sm.sum() - 1) < 1e-6, "Softmax")

    from utils.visualization import create_overview_grid
    grid = create_overview_grid([obs.image, obs.image], [obs.mask, obs.mask],
                                titles=["A", "B"], cols=2)
    check(grid.ndim == 3 and grid.shape[2] == 3, f"Grid shape={grid.shape}")


UTILS_TESTS = [
    ("Utility module smoke tests", test_utils),
]
