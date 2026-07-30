# tests/test_quality.py
"""
Quality scorer correctness tests.

Tests:
- output range
- score assignment
- metric weighting
- blur degradation
- occlusion degradation
- area degradation
- deterministic behaviour
- monotonic quality response
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from tests.test_utils import (
    check,
    make_image,
    full_mask,
)


# ============================================================
# OBSERVATION HELPERS
# ============================================================

def make_observation(
    image=None,
    mask=None,
    object_hand=None,
    obs_id=0,
):
    """
    Construct synthetic Observation.
    """

    from data_io.observation import Observation

    if image is None:
        image = make_image()

    if mask is None:
        mask = full_mask(
            image.shape[0],
            image.shape[1],
        )

    return Observation(
        id=obs_id,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask,
        object_hand=object_hand,
    )


def make_scorer():
    """
    Standard quality scorer used by tests.
    """

    from quality.quality_scorer import QualityScorer
    from quality.blur import BlurQuality
    from quality.area import AreaQuality
    from quality.occlusion import OcclusionQuality

    return QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality(),
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2,
        },
    )


# ============================================================
# BASIC BEHAVIOUR
# ============================================================

def test_quality_range():
    scorer = make_scorer()

    obs = make_observation()

    q = scorer.score(obs)

    check(
        0.0 <= q <= 1.0,
        f"Quality range valid ({q:.4f})",
    )

    check(
        np.isclose(obs.quality, q),
        "Observation quality updated",
    )


def test_quality_weights_sum_to_one():
    scorer = make_scorer()

    check(
        np.isclose(
            sum(scorer.weights.values()),
            1.0,
        ),
        "Quality weights sum to 1",
    )


def test_quality_deterministic():

    scorer = make_scorer()

    obs = make_observation()

    q1 = scorer.score(obs)
    q2 = scorer.score(obs)

    check(
        np.isclose(q1, q2),
        f"Quality deterministic ({q1:.6f}, {q2:.6f})",
    )


# ============================================================
# QUALITY RESPONSE
# ============================================================

def test_perfect_observation_scores_high():

    scorer = make_scorer()

    obs = make_observation(
        image=make_image(),
        mask=full_mask(),
    )

    q = scorer.score(obs)

    check(
        q > 0.8,
        f"Clean observation scores high ({q:.4f})",
    )


def test_blur_reduces_quality():

    scorer = make_scorer()

    sharp = make_image()

    blurry = cv2.GaussianBlur(
        sharp,
        (31, 31),
        10,
    )

    obs_sharp = make_observation(
        image=sharp,
        obs_id=0,
    )

    obs_blurry = make_observation(
        image=blurry,
        obs_id=1,
    )

    q_sharp = scorer.score(obs_sharp)
    q_blurry = scorer.score(obs_blurry)

    check(
        q_blurry < q_sharp,
        (
            f"Blur lowers quality "
            f"({q_sharp:.4f} -> {q_blurry:.4f})"
        ),
    )


def test_occlusion_reduces_quality():

    scorer = make_scorer()

    mask = full_mask()

    clean = make_observation(
        mask=mask,
        object_hand=None,
        obs_id=0,
    )

    hand = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    hand[:, :50] = 255

    occluded = make_observation(
        mask=mask,
        object_hand=hand,
        obs_id=1,
    )

    q_clean = scorer.score(clean)
    q_occluded = scorer.score(occluded)

    check(
        q_occluded < q_clean,
        (
            f"Occlusion lowers quality "
            f"({q_clean:.4f} -> {q_occluded:.4f})"
        ),
    )


def test_small_area_reduces_quality():

    scorer = make_scorer()

    large_mask = full_mask()

    small_mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    small_mask[
        49:51,
        49:51,
    ] = 255

    large = make_observation(
        mask=large_mask,
        obs_id=0,
    )

    small = make_observation(
        mask=small_mask,
        obs_id=1,
    )

    q_large = scorer.score(large)
    q_small = scorer.score(small)

    check(
        q_small < q_large,
        (
            f"Small object lowers quality "
            f"({q_large:.4f} -> {q_small:.4f})"
        ),
    )


# ============================================================
# EXTREME CASES
# ============================================================

def test_empty_mask_quality():

    scorer = make_scorer()

    empty = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    obs = make_observation(
        mask=empty,
    )

    q = scorer.score(obs)

    check(
        0 <= q <= 1,
        f"Empty mask quality valid ({q:.4f})",
    )


def test_full_mask_quality():

    scorer = make_scorer()

    obs = make_observation(
        mask=full_mask(),
    )

    q = scorer.score(obs)

    check(
        q > 0,
        f"Full mask produces positive quality ({q:.4f})",
    )


# ============================================================
# EXPORT
# ============================================================

QUALITY_TESTS = [

    (
        "Quality range",
        test_quality_range,
    ),

    (
        "Quality weights",
        test_quality_weights_sum_to_one,
    ),

    (
        "Quality deterministic",
        test_quality_deterministic,
    ),

    (
        "Perfect quality",
        test_perfect_observation_scores_high,
    ),

    (
        "Blur degradation",
        test_blur_reduces_quality,
    ),

    (
        "Occlusion degradation",
        test_occlusion_reduces_quality,
    ),

    (
        "Area degradation",
        test_small_area_reduces_quality,
    ),

    (
        "Empty mask quality",
        test_empty_mask_quality,
    ),

    (
        "Full mask quality",
        test_full_mask_quality,
    ),
]