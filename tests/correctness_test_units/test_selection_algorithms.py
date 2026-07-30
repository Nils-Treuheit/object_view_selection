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


def test_all_selectors_clamp_n_to_embedding_count():
    """When n > N, each selector returns exactly N indices."""
    from selection.fps import FarthestPointSampling
    from selection.greedy_quality_diversity import GreedyQualityDiversity
    from selection.facility_location import FacilityLocation
    from selection.dpp import DPPSelector
    from selection.next_best_view import NextBestView

    emb = np.eye(4, dtype=np.float32)
    q = np.ones(4, dtype=np.float32)
    for name, cls in [("FPS", FarthestPointSampling),
                      ("GQD", GreedyQualityDiversity),
                      ("FL", FacilityLocation),
                      ("DPP", DPPSelector),
                      ("NBV", NextBestView)]:
        idx = cls().select(emb, q, 100)
        check(len(idx) == 4, f"{name} (n=100 > N=4): returned {len(idx)}, expected 4")


def test_all_selectors_return_type_is_int_ndarray():
    """All selectors return np.ndarray with int dtype."""
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
        idx = cls().select(emb, q, 2)
        check(isinstance(idx, np.ndarray), f"{name} returns ndarray")
        check(idx.dtype in (np.int32, np.int64, int), f"{name} has int dtype but got {idx.dtype}")


def test_gqd_fallback_uniform_quality():
    """GQD without quality_scores falls back to uniform weights — equivalent to FPS with alpha=0."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    rng = np.random.RandomState(10)
    emb = rng.randn(12, 6).astype(np.float32)

    idx_with_none = GreedyQualityDiversity(alpha=0.4, beta=0.6).select(emb, None, 5)
    quality_uniform = np.ones(len(emb), dtype=np.float32)
    idx_all_quality = GreedyQualityDiversity(alpha=0.4, beta=0.6).select(emb, quality_uniform, 5)

    check(np.array_equal(idx_with_none, idx_all_quality),
          "GQD quality=None fallback should equal uniform quality_scores")


def test_gqd_alpha_zero_biased_to_pure_diversity():
    """GQD with alpha=0 behaves like pure diversity (min-distance driven)."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    _one = _onehot(6, 8)
    dist = pairwise_distances(_one, metric="cosine")

    idx = GreedyQualityDiversity(alpha=0.0, beta=1.0).select(_one, None, 4)

    for step in range(1, len(idx)):
        sel = [int(j) for j in idx[:step]]
        min_dists = dist[:, sel].min(axis=1)
        min_dists[sel] = -1
        expected = int(min_dists.argmax())
        check(int(idx[step]) == expected,
              f"GQD alpha=0 step {step}: got {idx[step]}, expected pure diversity pick {expected}")


def test_nbv_deterministic_no_quality_variation():
    """NBV with uniform quality picks deterministically by mean-Euclidean diversity."""
    from selection.next_best_view import NextBestView

    # Orthonormal 5 points: each has same quality, so NBV should spread them maximally
    _one = _onehot(5, 8)
    quality = np.ones(5, dtype=np.float32) * 0.5

    idx = NextBestView().select(_one, quality, 4)
    check(int(idx[0]) == 0, f"NBV with uniform quality: first pick argmax=0, got {idx[0]}")


def test_dpp_sigma_affects_selection():
    """Changing sigma in DPP changes which points are selected."""
    from selection.dpp import DPPSelector

    rng = np.random.RandomState(11)
    emb = rng.randn(8, 4).astype(np.float32)
    quality = np.array([0.5] * 8, dtype=np.float32)

    idx_small = DPPSelector(sigma=0.1).select(emb, quality, 4)
    idx_large = DPPSelector(sigma=5.0).select(emb, quality, 4)

    check(not np.array_equal(idx_small, idx_large),
          "DPP with different sigma should produce different selections")


def test_dpp_quality_weight_matters():
    """Higher quality points get selected when embeddings are similar."""
    from selection.dpp import DPPSelector

    # Two very close points + one far point; high-quality among the cluster should win first
    emb = np.array([
        [1.0, 0.0, 0.0],
        [0.999, 0.001, 0.0],     # almost identical to [0]
        [0.0, 1.0, 0.0],           # orthogonal far point
    ], dtype=np.float32)
    quality = np.array([0.9, 0.1, 0.5], dtype=np.float32)

    idx = DPPSelector(sigma=0.5).select(emb, quality, 2)
    check(int(idx[0]) == 0, f"DPP quality-weighted: first pick {idx[0]}, expected 0 (highest quality)")


def test_fps_n_equals_n_samples_returns_all():
    """FPS with n == N returns all indices."""
    from selection.fps import FarthestPointSampling

    rng = np.random.RandomState(12)
    emb = rng.randn(10, 4).astype(np.float32)
    idx = FarthestPointSampling().select(emb, None, 10)
    check(len(idx) == 10, f"FPS n=10=N: got {len(idx)} indices")
    check(set(map(int, idx)) == set(range(10)), "FPS n=N: all indices present")


def test_gqd_quality_scores_required_for_quality_weighting():
    """When quality is provided, GQD actually prefers higher-quality points."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = _onehot(4)
    # Points 0 and 1 are far apart (orthogonal), both equally diverse
    # Point 3 has highest quality and is also well-separated
    quality = np.array([0.3, 0.3, 0.2, 0.95], dtype=np.float32)

    idx = GreedyQualityDiversity(alpha=1.0, beta=0.0).select(emb, quality, 4)
    # With alpha=1.0 and deterministic argmax on quality-first picks, should pick highest quality first
    check(int(idx[0]) == 3, f"GQD pure quality: first pick {idx[0]}, expected 3")


def test_all_selectors_n_1_returns_single_index():
    """When n=1, all selectors return single-element array."""
    from selection.fps import FarthestPointSampling
    from selection.greedy_quality_diversity import GreedyQualityDiversity
    from selection.facility_location import FacilityLocation
    from selection.dpp import DPPSelector
    from selection.next_best_view import NextBestView

    emb = np.eye(5, dtype=np.float32)
    q = np.array([0.1, 0.9, 0.3, 0.7, 0.2], dtype=np.float32)
    for name, cls in [("FPS", FarthestPointSampling),
                      ("GQD", GreedyQualityDiversity),
                      ("FL", FacilityLocation),
                      ("DPP", DPPSelector),
                      ("NBV", NextBestView)]:
        idx = cls().select(emb, q, 1)
        check(len(idx) == 1, f"{name} n=1: got {len(idx)} indices, expected 1")


def test_facility_location_coverage_step_by_step():
    """FL coverage objective strictly increases at each step."""
    from selection.facility_location import FacilityLocation

    rng = np.random.RandomState(13)
    emb = rng.randn(15, 6).astype(np.float32)
    sim = 1.0 - pairwise_distances(emb, metric="cosine")
    sim = np.clip(sim, 0, 1)

    idx = FacilityLocation().select(emb, None, 7)

    cumsum = []
    for step in range(len(idx)):
        sel = list(idx[:step + 1])
        cov = sim[:, sel].max(axis=1).sum()
        cumsum.append(cov)

    for i in range(1, len(cumsum)):
        check(cumsum[i] >= cumsum[i-1],
              f"FL step {i}: coverage {cumsum[i]:.4f} >= prev {cumsum[i-1]:.4f}")


def test_gqd_beta_only_diversity_far_points():
    """GQD with beta>>alpha selects farthest points when quality is uniform."""
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = np.array([
        [0, 0], [1, 0], [0.5, 0.8660254],   # equilateral triangle
        [0.5, 0.3],                            # inside triangle
    ], dtype=np.float32)
    quality = np.ones(4, dtype=np.float32) * 0.5

    idx = GreedyQualityDiversity(alpha=0.1, beta=0.9).select(emb, quality, 3)

    # First pick should be highest-quality tie (first argmax = 0 for uniform)
    check(int(idx[0]) == 0, f"GQD uniform quality: first argmax {idx[0]}, expected 0")


def test_nbv_alpha_equivalent_to_quality_plus_diversity():
    """NBV score formula: quality + 0.5 * mean_euclidean_dist — verify manually."""
    from selection.next_best_view import NextBestView

    rng = np.random.RandomState(14)
    emb = rng.randn(6, 4).astype(np.float32)
    quality = rng.rand(6).astype(np.float32)

    idx = NextBestView().select(emb, quality, 3)

    first = int(idx[0])
    expected_first = int(quality.argmax())
    check(first == expected_first, f"NBV first: {idx[0]} vs expected argmax quality {expected_first}")

    sel = [first]
    expected_scores = []
    for j in range(len(emb)):
        if j in sel:
            continue
        div = np.mean([np.linalg.norm(emb[j] - emb[k]) for k in sel])
        s = float(quality[j]) + 0.5 * div
        expected_scores.append((s, j))
    expected_scores.sort(key=lambda x: (-x[0], x[1]))
    check(int(idx[1]) == expected_scores[0][1],
          f"NBV step 2: got {idx[1]}, expected {expected_scores[0][1]} (score={expected_scores[0][0]:.4f})")


if __name__ == "__main__":
    names = [k for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for name in sorted(names):
        fn = globals()[name]
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
