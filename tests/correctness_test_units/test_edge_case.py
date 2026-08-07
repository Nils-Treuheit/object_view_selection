# tests/test_edge_cases.py
"""
Edge case correctness tests.

Tests:
- empty masks
- full masks
- degenerate inputs
- grayscale images
- tiny images
- duplicate embeddings
- invalid selection sizes
- zero-area objects
"""

import numpy as np

from pathlib import Path

from data_io.observation import Observation


def make_image(h=50, w=50):
    return (
        np.random.RandomState(123)
        .rand(h, w, 3)
        * 255
    ).astype(np.uint8)


def make_observation(
    image,
    mask,
    object_hand=None,
    obs_id=0
):
    return Observation(
        id=obs_id,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask,
        object_hand=object_hand,
    )


def test_empty_mask_area_filter():

    from preprocessing.legacy.area_filter import AreaFilter

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    obs = make_observation(
        make_image(),
        mask
    )

    score, passed, _ = AreaFilter(
        minimum_ratio=0.01
    ).evaluate(obs)

    assert not passed
    assert obs.metrics.area_ratio == 0.0


def test_full_mask_area_filter():

    from preprocessing.legacy.area_filter import AreaFilter

    mask = np.ones(
        (50, 50),
        dtype=np.uint8
    ) * 255

    obs = make_observation(
        make_image(),
        mask
    )

    score, passed, _ = AreaFilter(
        minimum_ratio=0.01
    ).evaluate(obs)

    assert passed
    assert np.isclose(
        obs.metrics.area_ratio,
        1.0
    )


def test_single_pixel_mask():

    from preprocessing.future_work.completeness_filter import CompletenessFilter

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    mask[25, 25] = 255

    obs = make_observation(
        make_image(),
        mask
    )

    score, passed, _ = CompletenessFilter(
        minimum_score=0.5
    ).evaluate(obs)

    assert np.isfinite(score)


def test_border_filter_all_pixels_on_border():

    from preprocessing.legacy.border_truncation import BorderFilter

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    mask[0, :] = 255
    mask[-1, :] = 255
    mask[:, 0] = 255
    mask[:, -1] = 255

    obs = make_observation(
        make_image(),
        mask
    )

    score, passed, _ = BorderFilter(
        maximum_ratio=0.01
    ).evaluate(obs)

    assert obs.metrics.border_ratio > 0
    assert not passed


def test_grayscale_image_handling():

    from preprocessing.legacy.blur_filter import BlurFilter

    gray = (
        np.random.RandomState(0)
        .rand(50, 50)
        * 255
    ).astype(np.uint8)

    gray_rgb = np.stack(
        [
            gray,
            gray,
            gray
        ],
        axis=-1
    )

    mask = np.ones(
        (50, 50),
        dtype=np.uint8
    ) * 255

    obs = make_observation(
        gray_rgb,
        mask
    )

    score, passed, _ = BlurFilter().evaluate(obs)

    assert np.isfinite(
        obs.metrics.laplacian
    )


def test_tiny_image():

    from preprocessing.legacy.blur_filter import BlurFilter

    image = make_image(
        5,
        5
    )

    mask = np.ones(
        (5, 5),
        dtype=np.uint8
    ) * 255

    obs = make_observation(
        image,
        mask
    )

    score, passed, _ = BlurFilter().evaluate(obs)

    assert np.isfinite(
        obs.metrics.laplacian
    )


def test_no_hand_occlusion():

    from preprocessing.future_work.occlusion_filter import OcclusionFilter

    mask = np.ones(
        (50, 50),
        dtype=np.uint8
    ) * 255

    obs = make_observation(
        make_image(),
        mask,
        object_hand=None
    )

    score, passed, _ = OcclusionFilter().evaluate(obs)

    assert passed
    assert obs.metrics.hand_overlap == 0.0


def test_hand_without_overlap():

    from preprocessing.future_work.occlusion_filter import OcclusionFilter

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    mask[10:40, 10:40] = 255

    hand = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    hand[0:5, 0:5] = 255

    obs = make_observation(
        make_image(),
        mask,
        object_hand=hand
    )

    score, passed, _ = OcclusionFilter().evaluate(obs)

    assert passed
    assert obs.metrics.hand_overlap == 0.0


def test_fps_more_requested_than_available():

    from selection.fps import FarthestPointSampling

    embeddings = np.random.RandomState(0).randn(
        3,
        8
    ).astype(np.float32)

    indices = FarthestPointSampling().select(
        embeddings,
        None,
        n=10
    )

    assert len(indices) == 3
    assert len(set(indices)) == 3


def test_fps_duplicate_embeddings():

    from selection.fps import FarthestPointSampling

    embeddings = np.ones(
        (5, 4),
        dtype=np.float32
    )

    indices = FarthestPointSampling().select(
        embeddings,
        None,
        n=3
    )

    assert len(indices) == 3
    assert 1 <= len(set(indices)) <= 3


def test_dpp_duplicate_embeddings():

    from selection.dpp import DPPSelector

    embeddings = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1]
        ],
        dtype=np.float32
    )

    quality = np.ones(
        3,
        dtype=np.float32
    )

    indices = DPPSelector(
        sigma=0.5
    ).select(
        embeddings,
        quality,
        n=2
    )

    assert len(indices) == 2
    assert len(set(indices)) == 2


def test_descriptor_empty_mask():

    from descriptors.hu import hu_moments
    from descriptors.fourier import fourier_descriptors

    empty = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    hu = hu_moments(empty)

    fd = fourier_descriptors(
        empty,
        num_descriptors=16
    )

    assert np.all(
        np.isfinite(hu)
    )

    assert np.all(
        np.isfinite(fd)
    )


def test_descriptor_repeatability():

    from descriptors.zernike import zernike_moments

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8
    )

    mask[10:40, 10:40] = 255

    z1 = zernike_moments(
        mask,
        degree=6
    )

    z2 = zernike_moments(
        mask,
        degree=6
    )

    assert np.allclose(
        z1,
        z2
    )