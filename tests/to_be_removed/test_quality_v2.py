# tests/test_quality.py
"""
Quality scorer correctness tests.

Tests:
- output range
- metric weighting behavior
- degradation response
- monotonicity with quality changes
"""

import numpy as np
import cv2

from data_io.observation import Observation
from quality.quality_scorer import QualityScorer
from quality.blur import BlurQuality
from quality.area import AreaQuality
from quality.occlusion import OcclusionQuality


def make_image(h=100, w=100):
    return (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)


def make_observation(
    image=None,
    mask=None,
    object_hand=None,
    obs_id=0
):
    from pathlib import Path

    h, w = mask.shape if mask is not None else image.shape[:2]

    if image is None:
        image = make_image(h, w)

    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255

    return Observation(
        id=obs_id,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask,
        object_hand=object_hand,
    )


def test_quality_range():
    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    obs = make_observation()

    q = scorer.score(obs)

    assert 0.0 <= q <= 1.0
    assert obs.quality == q


def test_perfect_observation_scores_high():
    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    img = make_image()

    obs = make_observation(
        image=img,
        mask=np.ones((100, 100), dtype=np.uint8) * 255
    )

    q = scorer.score(obs)

    assert q > 0.8


def test_blur_reduces_quality():

    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    sharp = make_image()

    blurry = cv2.GaussianBlur(
        sharp,
        (31, 31),
        10
    )

    obs_sharp = make_observation(
        image=sharp,
        obs_id=0
    )

    obs_blurry = make_observation(
        image=blurry,
        obs_id=1
    )

    q_sharp = scorer.score(obs_sharp)
    q_blurry = scorer.score(obs_blurry)

    assert q_blurry < q_sharp


def test_occlusion_reduces_quality():

    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8
    ) * 255

    no_hand = make_observation(
        mask=mask,
        object_hand=None,
        obs_id=0
    )

    heavy_hand = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    heavy_hand[:, :50] = 255

    occluded = make_observation(
        mask=mask,
        object_hand=heavy_hand,
        obs_id=1
    )

    q_clean = scorer.score(no_hand)
    q_occluded = scorer.score(occluded)

    assert q_occluded < q_clean


def test_small_object_reduces_area_quality():

    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    full_mask = np.ones(
        (100, 100),
        dtype=np.uint8
    ) * 255

    small_mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    small_mask[49:51, 49:51] = 255

    large_obs = make_observation(
        mask=full_mask,
        obs_id=0
    )

    small_obs = make_observation(
        mask=small_mask,
        obs_id=1
    )

    q_large = scorer.score(large_obs)
    q_small = scorer.score(small_obs)

    assert q_small < q_large


def test_quality_is_deterministic():

    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    obs = make_observation()

    q1 = scorer.score(obs)
    q2 = scorer.score(obs)

    assert np.isclose(q1, q2)


def test_quality_weights_sum_correctly():

    scorer = QualityScorer(
        metrics=[
            BlurQuality(),
            AreaQuality(),
            OcclusionQuality()
        ],
        weights={
            "blur": 0.5,
            "area": 0.3,
            "occlusion": 0.2
        }
    )

    assert np.isclose(
        sum(scorer.weights.values()),
        1.0
    )