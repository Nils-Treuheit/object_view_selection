#!/usr/bin/env python3
"""
tests/test_selection.py

Correctness tests for subset selection algorithms:

    - Farthest Point Sampling (FPS)
    - Greedy Quality Diversity (GQD)
    - Determinantal Point Process (DPP)

Tests focus on:
    - deterministic behaviour where expected
    - diversity preservation
    - duplicate avoidance
    - quality sensitivity
    - edge cases

Exports:
    SELECTION_TESTS
"""

import numpy as np

from tests.test_utils import check


# ---------------------------------------------------------------------
# Synthetic embeddings
# ---------------------------------------------------------------------

def make_random_embeddings(
    n=20,
    dim=8,
    seed=42,
):
    rng = np.random.RandomState(seed)
    return rng.randn(n, dim).astype(np.float32)


def make_cluster_embeddings():
    """
    Three clearly separated clusters.
    """
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [-0.1, 0.0],

            [10.0, 0.0],
            [10.1, 0.0],
            [9.9, 0.0],

            [-10.0, 0.0],
            [-10.1, 0.0],
            [-9.9, 0.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# FPS
# ---------------------------------------------------------------------

def test_fps_returns_correct_number():
    from selection.fps import FarthestPointSampling

    emb = make_random_embeddings(
        n=20,
        dim=8,
    )

    selector = FarthestPointSampling()

    idx = selector.select(
        emb,
        None,
        n=5,
    )

    check(
        len(idx) == 5,
        f"FPS returns requested number of samples ({len(idx)})",
    )


def test_fps_unique_indices():
    from selection.fps import FarthestPointSampling

    emb = make_random_embeddings()

    idx = FarthestPointSampling().select(
        emb,
        None,
        n=10,
    )

    check(
        len(set(idx)) == 10,
        "FPS selects unique indices",
    )


def test_fps_selects_all_equidistant_points():
    from selection.fps import FarthestPointSampling

    # Orthogonal vectors:
    # all pairwise cosine distances are equal
    emb = np.eye(
        3,
        dtype=np.float32,
    )

    idx = FarthestPointSampling().select(
        emb,
        None,
        n=3,
    )

    check(
        len(set(idx)) == 3,
        f"FPS selects all equidistant points: {idx}",
    )


def test_fps_prefers_diverse_points():
    from selection.fps import FarthestPointSampling
    from sklearn.metrics import pairwise_distances

    emb = np.array(
        [
            [1, 0],
            [0.9, 0.1],
            [-1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )

    idx = FarthestPointSampling().select(
        emb,
        None,
        n=2,
    )

    selected = emb[idx]

    distance = pairwise_distances(
        selected,
        metric="cosine",
    )[0, 1]

    check(
        distance > 1.5,
        f"FPS selects distant points (cosine distance={distance:.3f})",
    )


def test_fps_handles_duplicates():
    from selection.fps import FarthestPointSampling

    emb = np.ones(
        (5, 4),
        dtype=np.float32,
    )

    idx = FarthestPointSampling().select(
        emb,
        None,
        n=3,
    )

    check(
        len(set(idx)) == 3,
        "FPS handles duplicate embeddings",
    )


def test_fps_caps_at_dataset_size():
    from selection.fps import FarthestPointSampling

    emb = make_random_embeddings(
        n=3,
        dim=4,
    )

    idx = FarthestPointSampling().select(
        emb,
        None,
        n=10,
    )

    check(
        len(idx) == 3,
        f"FPS caps selection at dataset size ({len(idx)})",
    )


# ---------------------------------------------------------------------
# GREEDY QUALITY DIVERSITY
# ---------------------------------------------------------------------

def test_gqd_returns_unique_points():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = make_random_embeddings()

    quality = np.random.RandomState(1).rand(
        len(emb)
    ).astype(np.float32)

    selector = GreedyQualityDiversity()

    idx = selector.select(
        emb,
        quality,
        n=5,
    )

    check(
        len(set(idx)) == 5,
        "GQD selects unique indices",
    )


def test_gqd_prefers_high_quality():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = make_random_embeddings(
        n=10,
        dim=4,
    )

    quality = np.ones(
        10,
        dtype=np.float32,
    )

    quality[7] = 100.0

    selector = GreedyQualityDiversity(
        alpha=0.9,
        beta=0.1,
    )

    idx = selector.select(
        emb,
        quality,
        n=3,
    )

    check(
        idx[0] == 7,
        f"GQD selects highest quality first ({idx})",
    )


def test_gqd_quality_mode_clusters():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = make_cluster_embeddings()

    quality = np.array(
        [
            10,
            9,
            8,

            1,
            1,
            1,

            1,
            1,
            1,
        ],
        dtype=np.float32,
    )

    selector = GreedyQualityDiversity(
        alpha=0.99,
        beta=0.01,
    )

    idx = selector.select(
        emb,
        quality,
        n=3,
    )

    check(
        0 in idx,
        f"Quality mode selects best item ({idx})",
    )


def test_gqd_diversity_mode_spreads_clusters():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    emb = make_cluster_embeddings()

    quality = np.ones(
        len(emb),
        dtype=np.float32,
    )

    selector = GreedyQualityDiversity(
        alpha=0.01,
        beta=0.99,
    )

    idx = selector.select(
        emb,
        quality,
        n=3,
    )

    cluster_positions = set(
        np.sign(
            emb[idx, 0]
        )
    )

    check(
        len(cluster_positions) >= 2,
        f"GQD diversity mode spreads clusters ({idx})",
    )


# ---------------------------------------------------------------------
# DPP
# ---------------------------------------------------------------------

def test_dpp_returns_unique_points():
    from selection.dpp import DPPSelector

    emb = make_random_embeddings(
        n=10,
        dim=8,
    )

    quality = np.random.RandomState(0).rand(
        10
    ).astype(np.float32)

    idx = DPPSelector().select(
        emb,
        quality,
        n=4,
    )

    check(
        len(set(idx)) == 4,
        "DPP selects unique indices",
    )


def test_dpp_avoids_duplicates():
    from selection.dpp import DPPSelector

    emb = np.array(
        [
            [1, 0],
            [1.001, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )

    quality = np.ones(
        5,
        dtype=np.float32,
    )

    idx = DPPSelector(
        sigma=0.5,
    ).select(
        emb,
        quality,
        n=2,
    )

    distance = np.linalg.norm(
        emb[idx[0]] - emb[idx[1]]
    )

    check(
        distance > 0.5,
        f"DPP avoids duplicate embeddings (distance={distance:.3f})",
    )


def test_dpp_prefers_quality():
    from selection.dpp import DPPSelector

    emb = np.eye(
        5,
        dtype=np.float32,
    )

    quality = np.array(
        [
            0.1,
            0.2,
            0.9,
            0.3,
            0.4,
        ],
        dtype=np.float32,
    )

    idx = DPPSelector(
        sigma=0.5,
    ).select(
        emb,
        quality,
        n=2,
    )

    check(
        idx[0] == 2,
        f"DPP selects highest quality item first ({idx})",
    )


def test_dpp_handles_small_dataset():
    from selection.dpp import DPPSelector

    emb = np.random.RandomState(0).randn(
        3,
        4,
    ).astype(np.float32)

    quality = np.ones(
        3,
        dtype=np.float32,
    )

    idx = DPPSelector().select(
        emb,
        quality,
        n=10,
    )

    check(
        len(idx) == 3,
        f"DPP caps at dataset size ({len(idx)})",
    )


# ---------------------------------------------------------------------
# Export list
# ---------------------------------------------------------------------

SELECTION_TESTS = [
    ("FPS count", test_fps_returns_correct_number),
    ("FPS uniqueness", test_fps_unique_indices),
    ("FPS equidistant", test_fps_selects_all_equidistant_points),
    ("FPS diversity", test_fps_prefers_diverse_points),
    ("FPS duplicates", test_fps_handles_duplicates),
    ("FPS size limit", test_fps_caps_at_dataset_size),

    ("GQD uniqueness", test_gqd_returns_unique_points),
    ("GQD quality preference", test_gqd_prefers_high_quality),
    ("GQD quality mode", test_gqd_quality_mode_clusters),
    ("GQD diversity mode", test_gqd_diversity_mode_spreads_clusters),

    ("DPP uniqueness", test_dpp_returns_unique_points),
    ("DPP duplicate avoidance", test_dpp_avoids_duplicates),
    ("DPP quality preference", test_dpp_prefers_quality),
    ("DPP size limit", test_dpp_handles_small_dataset),
]