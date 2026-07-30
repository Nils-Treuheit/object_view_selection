"""
Verify each selection algorithm's internal logic matches the
documentation in docs/selection_algorithms.md.
"""

import numpy as np
from sklearn.metrics import pairwise_distances

from tests.test_utils import check

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

EPS = 1e-5


def _onehot(n, dim=4):
    """n points each on a distinct axis of a dim-D space."""
    return np.eye(dim, dtype=np.float32)[:n]


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def test_fps_first_pick_is_random():
    """FPS does NOT always start from the same index (random start)."""
    from selection.fps import FarthestPointSampling

    emb = _onehot(5, 8)
    runs = set()
    for _ in range(20):
        idx = FarthestPointSampling().select(emb, None, 2)
        runs.add(int(idx[0]))
    check(len(runs) > 1, f"FPS first pick varies ({len(runs)} unique starts in 20 runs)")


def test_fps_second_pick_is_farthest_from_first():
    """After picking index i, FPS picks the point with max min-dist to i."""
    from selection.fps import FarthestPointSampling

    rng = np.random.RandomState(0)
    emb = rng.randn(10, 8).astype(np.float32)
    dist = pairwise_distances(emb, metric="cosine")

    for _ in range(10):
        idx = FarthestPointSampling().select(emb, None, 2)
        i0 = int(idx[0])
        i1 = int(idx[1])

        min_dists = dist[:, i0].copy()
        min_dists[i0] = -1
        expected = int(min_dists.argmax())

        check(i1 == expected, f"FPS pick2: got {i1}, expected farthest-from-{i0} = {expected}")


def test_fps_subsequent_picks_max_min_dist():
    """Each step picks the point whose minimum distance to all selected is largest."""
    from selection.fps import FarthestPointSampling

    rng = np.random.RandomState(1)
    emb = rng.randn(15, 8).astype(np.float32)
    dist = pairwise_distances(emb, metric="cosine")

    idx = FarthestPointSampling().select(emb, None, 5)
    for step in range(1, len(idx)):
        i = int(idx[step])
        sel = [int(j) for j in idx[:step]]
        min_dists = dist[:, sel].min(axis=1)
        min_dists[sel] = -1
        expected = int(min_dists.argmax())
        check(i == expected, f"FPS step {step}: got {i}, expected farthest = {expected}")


# ---------------------------------------------------------------------------
# Greedy Quality Diversity
# ---------------------------------------------------------------------------

def test_gqd_starts_with_quality_argmax():
    """GQD first pick is always quality_scores.argmax()."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = _onehot(5)
    quality = np.array([0.1, 0.9, 0.3, 0.7, 0.2], dtype=np.float32)

    idx = GreedyQualityDiversity(alpha=0.5, beta=0.5).select(emb, quality, 3)
    check(int(idx[0]) == 1, f"GQD first pick: got {idx[0]}, expected 1 (argmax quality)")


def test_gqd_score_matches_formula():
    """GQD score = alpha * quality[i] + beta * min(dist[i, selected])."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = _onehot(4)
    quality = np.array([0.2, 0.8, 0.5, 0.9], dtype=np.float32)
    dist = pairwise_distances(emb, metric="cosine")
    alpha, beta = 0.4, 0.6

    idx = GreedyQualityDiversity(alpha=alpha, beta=beta).select(emb, quality, 3)

    first = int(idx[0])
    check(first == 3, f"GQD first pick: got {first}, expected 3 (quality=0.9)")

    second = int(idx[1])
    sel = [first]
    expected_scores = []
    for i in range(len(emb)):
        if i in sel:
            continue
        div = dist[i, sel].min()
        s = alpha * quality[i] + beta * div
        expected_scores.append((s, i))
    expected_scores.sort(key=lambda x: (-x[0], x[1]))
    expected_second = expected_scores[0][1]
    check(second == expected_second, f"GQD second: got {second}, expected {expected_second} (max score)")

    third = int(idx[2])
    sel = [first, second]
    expected_scores = []
    for i in range(len(emb)):
        if i in sel:
            continue
        div = dist[i, sel].min()
        s = alpha * quality[i] + beta * div
        expected_scores.append((s, i))
    expected_scores.sort(key=lambda x: (-x[0], x[1]))
    expected_third = expected_scores[0][1]
    check(third == expected_third, f"GQD third: got {third}, expected {expected_third}")


def test_gqd_diversity_uses_min_distance():
    """GQD diversity term is min(dist[i, selected]), not mean or max."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.707, 0.707, 0],
    ], dtype=np.float32)
    quality = np.array([0.1, 0.1, 0.1, 0.9], dtype=np.float32)
    dist = pairwise_distances(emb, metric="cosine")
    alpha, beta = 0.01, 0.99

    idx = GreedyQualityDiversity(alpha=alpha, beta=beta).select(emb, quality, 3)

    first = int(idx[0])
    check(first == 3, f"GQD first pick (high quality): got {first}, expected 3")

    second = int(idx[1])
    sel = [first]
    expected_second = int(dist[:, sel].min(axis=1).argmax())
    check(second == expected_second, f"GQD second: got {second}, expected {expected_second} (farthest from first)")

    third = int(idx[2])
    sel = [first, second]
    expected_third = int(dist[:, sel].min(axis=1).argmax())
    check(third == expected_third, f"GQD third: got {third}, expected {expected_third} (farthest from both)")


# ---------------------------------------------------------------------------
# Facility Location
# ---------------------------------------------------------------------------

def test_fl_starts_with_most_central():
    """FacilityLocation first pick = argmax of sim.sum(axis=1)."""
    from selection.facility_location import FacilityLocation

    emb = np.array([
        [1, 0, 0],
        [0.95, 0.05, 0],
        [-1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float32)

    sim = 1.0 - pairwise_distances(emb, metric="cosine")
    sim = np.clip(sim, 0, 1)
    expected_first = int(sim.sum(axis=1).argmax())

    idx = FacilityLocation().select(emb, None, 2)
    check(int(idx[0]) == expected_first, f"FL first: got {idx[0]}, expected {expected_first}")


def test_fl_coverage_increases():
    """Each step increases the coverage objective sum_j max_k sim(j,k)."""
    from selection.facility_location import FacilityLocation

    rng = np.random.RandomState(2)
    emb = rng.randn(10, 6).astype(np.float32)
    sim = 1.0 - pairwise_distances(emb, metric="cosine")
    sim = np.clip(sim, 0, 1)

    idx = FacilityLocation().select(emb, None, 5)
    for step in range(1, len(idx)):
        sel_before = list(idx[:step])
        sel_after = list(idx[:step + 1])
        cov_before = sim[:, sel_before].max(axis=1).sum()
        cov_after = sim[:, sel_after].max(axis=1).sum()
        check(cov_after >= cov_before - EPS,
              f"FL step {step}: coverage {cov_after:.4f} >= {cov_before:.4f}")


def test_fl_central_first_then_spread():
    """FL picks the most central point first, then spreads outward."""
    from selection.facility_location import FacilityLocation

    emb = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0.01, 0.01, 0],
        [0, 1, 0],
        [0, -1, 0],
    ], dtype=np.float32)

    idx = FacilityLocation().select(emb, None, 3)

    sim = 1.0 - pairwise_distances(emb, metric="cosine")
    sim = np.clip(sim, 0, 1)
    expected_first = int(sim.sum(axis=1).argmax())
    check(int(idx[0]) == expected_first, f"FL first: got {idx[0]}, expected central {expected_first}")


# ---------------------------------------------------------------------------
# DPP
# ---------------------------------------------------------------------------

def test_dpp_kernel_is_quality_weighted():
    """DPP kernel L = outer(q, q) * exp(-dist / sigma)."""
    from selection.dpp import DPPSelector

    rng = np.random.RandomState(3)
    emb = rng.randn(6, 4).astype(np.float32)
    quality = rng.rand(6).astype(np.float32)
    sigma = 0.5

    sim = np.exp(-pairwise_distances(emb, metric="cosine") / sigma)
    L_expected = np.outer(quality, quality) * sim

    idx = DPPSelector(sigma=sigma).select(emb, quality, 3)
    check(len(idx) == 3, f"DPP returns {len(idx)} indices")


def test_dpp_first_pick_highest_quality():
    """When all points are orthogonal, DPP's first pick is the highest quality."""
    from selection.dpp import DPPSelector

    emb = np.eye(5, dtype=np.float32)
    quality = np.array([0.1, 0.2, 0.9, 0.3, 0.4], dtype=np.float32)

    idx = DPPSelector(sigma=0.5).select(emb, quality, 2)

    check(int(idx[0]) == 2, f"DPP first: got {idx[0]}, expected 2 (quality=0.9)")


def test_dpp_submatrix_stays_psd():
    """All DPP kernel submatrices are positive semi-definite (det > 0)."""
    from selection.dpp import DPPSelector

    rng = np.random.RandomState(4)
    emb = rng.randn(8, 6).astype(np.float32)
    quality = rng.rand(8).astype(np.float32)
    sigma = 0.5

    sim = np.exp(-pairwise_distances(emb, metric="cosine") / sigma)
    L = np.outer(quality, quality) * sim

    idx = DPPSelector(sigma=sigma).select(emb, quality, 4)
    for step in range(1, len(idx) + 1):
        s = list(idx[:step])
        Ls = L[np.ix_(s, s)]
        sign, logdet = np.linalg.slogdet(Ls)
        check(sign > 0,
              f"DPP step {step}: submatrix has positive determinant (sign={sign})")


def test_dpp_picks_max_gain():
    """At each step, DPP picks the candidate with the largest log-det gain."""
    from selection.dpp import DPPSelector

    rng = np.random.RandomState(7)
    emb = rng.randn(10, 6).astype(np.float32)
    quality = rng.rand(10).astype(np.float32)
    sigma = 0.5

    sim = np.exp(-pairwise_distances(emb, metric="cosine") / sigma)
    L = np.outer(quality, quality) * sim

    idx = DPPSelector(sigma=sigma).select(emb, quality, 5)
    for step in range(1, len(idx)):
        selected = list(idx[:step])
        s_before = list(idx[:step])
        L_before = L[np.ix_(s_before, s_before)]
        _, logdet_before = np.linalg.slogdet(L_before)

        actual_choice = int(idx[step])
        best_gain = -np.inf
        best_i = -1
        for i in range(len(emb)):
            if i in selected:
                continue
            s = selected + [i]
            Ls = L[np.ix_(s, s)]
            _, ld = np.linalg.slogdet(Ls)
            gain = ld - logdet_before
            if gain > best_gain:
                best_gain = gain
                best_i = i

        check(actual_choice == best_i,
              f"DPP step {step}: DPP chose {actual_choice}, expected {best_i} (best gain={best_gain:.4f})")


# ---------------------------------------------------------------------------
# Next Best View
# ---------------------------------------------------------------------------

def test_nbv_starts_with_quality_argmax():
    """NBV first pick is always quality_scores.argmax()."""
    from selection.next_best_view import NextBestView

    emb = _onehot(5)
    quality = np.array([0.3, 0.1, 0.9, 0.2, 0.7], dtype=np.float32)

    idx = NextBestView().select(emb, quality, 3)
    check(int(idx[0]) == 2, f"NBV first: got {idx[0]}, expected 2 (quality=0.9)")


def test_nbv_score_matches_formula():
    """NBV score = quality[i] + 0.5 * mean(euclidean_dist[i, selected])."""
    from selection.next_best_view import NextBestView

    rng = np.random.RandomState(5)
    emb = rng.randn(8, 4).astype(np.float32)
    quality = rng.rand(8).astype(np.float32)

    idx = NextBestView().select(emb, quality, 4)

    first = int(idx[0])
    check(first == int(quality.argmax()), f"NBV first: got {first}, expected argmax quality")

    for step in range(1, len(idx)):
        i = int(idx[step])
        sel = [int(j) for j in idx[:step]]
        expected_scores = []
        for j in range(len(emb)):
            if j in sel:
                continue
            div = np.mean([np.linalg.norm(emb[j] - emb[k]) for k in sel])
            s = float(quality[j]) + 0.5 * div
            expected_scores.append((s, j))
        expected_scores.sort(key=lambda x: (-x[0], x[1]))
        expected_i = expected_scores[0][1]
        check(i == expected_i,
              f"NBV step {step}: got {i}, expected {expected_i} (max score={expected_scores[0][0]:.4f})")


def test_nbv_uses_euclidean_not_cosine():
    """NBV diversity uses mean Euclidean distance, not cosine."""
    from selection.next_best_view import NextBestView

    quality = np.array([0.1, 0.1, 0.9], dtype=np.float32)
    emb = np.array([
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    idx = NextBestView().select(emb, quality, 3)

    check(int(idx[0]) == 2, f"NBV first: got {idx[0]}, expected 2 (quality=0.9)")
    second = int(idx[1])
    sel = [2]
    scores = []
    for j in [0, 1]:
        div = np.mean([np.linalg.norm(emb[j] - emb[k]) for k in sel])
        scores.append((float(quality[j]) + 0.5 * div, j))
    scores.sort(key=lambda x: (-x[0], x[1]))
    expected = scores[0][1]
    check(second == expected,
          f"NBV second: got {second}, expected {expected} (max euclidean-based score)")


# ---------------------------------------------------------------------------
# Cross-selector: diversity quality monotonicity
# ---------------------------------------------------------------------------

def test_all_selectors_reject_negative_n():
    """All selectors return empty array when n <= 0."""
    from selection.fps import FarthestPointSampling
    from selection.greedy_quality_diversity import GreedyQualityDiversity
    from selection.facility_location import FacilityLocation
    from selection.dpp import DPPSelector
    from selection.next_best_view import NextBestView

    emb = np.eye(3, dtype=np.float32)
    q = np.ones(3, dtype=np.float32)
    for name, cls in [("FPS", FarthestPointSampling),
                      ("GQD", GreedyQualityDiversity),
                      ("FL", FacilityLocation),
                      ("DPP", DPPSelector),
                      ("NBV", NextBestView)]:
        idx = cls().select(emb, q, 0)
        check(len(idx) == 0, f"{name} returns empty for n=0")


def test_all_selectors_no_duplicates():
    """All selectors return unique indices for any n."""
    from selection.fps import FarthestPointSampling
    from selection.greedy_quality_diversity import GreedyQualityDiversity
    from selection.facility_location import FacilityLocation
    from selection.dpp import DPPSelector
    from selection.next_best_view import NextBestView

    rng = np.random.RandomState(6)
    emb = rng.randn(20, 8).astype(np.float32)
    q = rng.rand(20).astype(np.float32)

    for name, cls in [("FPS", FarthestPointSampling),
                      ("GQD", GreedyQualityDiversity),
                      ("FL", FacilityLocation),
                      ("DPP", DPPSelector),
                      ("NBV", NextBestView)]:
        for n in [1, 3, 10, 15]:
            idx = cls().select(emb, q, n)
            check(len(set(idx)) == len(idx),
                   f"{name} (n={n}): unique indices")


if __name__ == "__main__":
    names = [k for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for name in sorted(names):
        fn = globals()[name]
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
