"""
Smoke tests for subset selection algorithms.
"""

import numpy as np

from tests.smoke_test_utils import check


def test_selection():
    n_obs, n_sel = 50, 8
    rng = np.random.RandomState(42)
    emb = rng.randn(n_obs, 64).astype(np.float32)
    qual = rng.rand(n_obs).astype(np.float32)

    from selection.fps import FarthestPointSampling
    idx = FarthestPointSampling().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "FPS")

    from selection.greedy_quality_diversity import GreedyQualityDiversity
    idx = GreedyQualityDiversity(alpha=0.4, beta=0.6).select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "GQD")

    from selection.facility_location import FacilityLocation
    idx = FacilityLocation().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "FacilityLocation")

    from selection.dpp import DPPSelector
    idx = DPPSelector(sigma=0.5).select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "DPP")

    from selection.next_best_view import NextBestView
    idx = NextBestView().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "NBV")


def test_selection_edge_cases():
    rng = np.random.RandomState(42)
    emb = rng.randn(50, 64).astype(np.float32)
    qual = rng.rand(50).astype(np.float32)

    from selection.fps import FarthestPointSampling
    idx0 = FarthestPointSampling().select(emb[:0], qual[:0], n=5)
    check(len(idx0) == 0, "FPS empty")

    idx1 = FarthestPointSampling().select(emb[:3], qual[:3], n=10)
    check(len(idx1) == 3, f"FPS capped: {len(idx1)}/3")


SELECTION_TESTS = [
    ("Selection smoke tests", test_selection),
    ("Selection edge cases", test_selection_edge_cases),
]
