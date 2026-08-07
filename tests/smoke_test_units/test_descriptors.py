"""
Smoke tests for shape descriptors.
"""

import time
import numpy as np

from tests.smoke_test_utils import check, first_usable_observation


def test_descriptors(ds):
    mask = first_usable_observation(ds).mask

    from descriptors.hu import hu_moments
    t0 = time.time()
    hu = hu_moments(mask)
    dt = time.time() - t0
    check(len(hu) == 7, f"Hu dim=7, got {len(hu)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(hu)), "Hu finite")

    from descriptors.zernike import zernike_moments
    t0 = time.time()
    z = zernike_moments(mask, degree=6)
    dt = time.time() - t0
    check(len(z) > 0, f"Zernike dim={len(z)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(z)), "Zernike finite")

    from descriptors.fourier import fourier_descriptors
    t0 = time.time()
    fd = fourier_descriptors(mask, num_descriptors=32)
    dt = time.time() - t0
    check(len(fd) == 32, f"Fourier dim=32, got {len(fd)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(fd)), "Fourier finite")

    from descriptors.shape_context import shape_context_descriptor
    t0 = time.time()
    sc = shape_context_descriptor(mask)
    dt = time.time() - t0
    check(len(sc) > 0, f"Shape context dim={len(sc)}, {dt*1000:.1f}ms")
    check(abs(sc.sum() - 1.0) < 0.01, f"Shape context sums to ~1: {sc.sum():.4f}")

    check(np.allclose(hu, hu_moments(mask)), "Hu deterministic")


DESCRIPTOR_TESTS = [
    ("Descriptor smoke tests", test_descriptors),
]
