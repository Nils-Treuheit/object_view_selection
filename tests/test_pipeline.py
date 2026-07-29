# tests/test_pipeline.py
"""
Filter pipeline and integration correctness tests.

Tests:
- pipeline execution order
- cumulative rejection behavior
- multiple filters interacting correctly
- accepted and rejected observations
"""

import numpy as np

from data_io.observation import Observation
from preprocessing.filter_pipeline import FilterPipeline
from preprocessing.area_filter import AreaFilter
from preprocessing.border_truncation import BorderFilter
from preprocessing.blur_filter import BlurFilter

from pathlib import Path


def make_image(h=100, w=100):
    return (
        np.random.RandomState(42)
        .rand(h, w, 3)
        * 255
    ).astype(np.uint8)


def make_observation(mask, image=None, obs_id=0):
    if image is None:
        image = make_image(
            mask.shape[0],
            mask.shape[1]
        )

    return Observation(
        id=obs_id,
        image_path=Path(""),
        mask_path=Path(""),
        object_hand_path=None,
        image=image,
        mask=mask
    )


def test_pipeline_accepts_good_observation():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[25:75, 25:75] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is True


def test_pipeline_rejects_small_objects():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[49:51, 49:51] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_rejects_border_objects():

    pipeline = FilterPipeline(
        [
            AreaFilter(minimum_ratio=0.02),
            BorderFilter(maximum_ratio=0.01),
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[
        0:50,
        0:50
    ] = 255

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_runs_filters_in_order():

    class TrackingFilter:

        def __init__(self, name, result):
            self.name = name
            self.result = result
            self.called = False

        def evaluate(self, observation):
            self.called = True
            observation.test_order.append(self.name)
            return 1.0, self.result, {}

    f1 = TrackingFilter("first", True)
    f2 = TrackingFilter("second", True)

    pipeline = FilterPipeline(
        [
            f1,
            f2
        ]
    )

    mask = np.ones(
        (20, 20),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)
    obs.test_order = []

    result = pipeline.run(obs)

    assert result is True
    assert obs.test_order == [
        "first",
        "second"
    ]


def test_pipeline_stops_after_failure():

    class TrackingFilter:

        def __init__(self, name, result):
            self.name = name
            self.result = result

        def evaluate(self, observation):
            observation.test_order.append(self.name)
            return 0.0, self.result, {}

    f1 = TrackingFilter(
        "reject",
        False
    )

    f2 = TrackingFilter(
        "should_not_run",
        True
    )

    pipeline = FilterPipeline(
        [
            f1,
            f2
        ]
    )

    mask = np.ones(
        (20, 20),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)
    obs.test_order = []

    result = pipeline.run(obs)

    assert result is False

    assert obs.test_order == [
        "reject"
    ]


def test_pipeline_combines_blur_and_area():

    pipeline = FilterPipeline(
        [
            BlurFilter(
                laplacian_threshold=120,
                tenengrad_threshold=35
            ),
            AreaFilter(
                minimum_ratio=0.02
            )
        ]
    )

    sharp = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    yy, xx = np.ogrid[:100, :100]

    sharp[
        (xx - 50) ** 2 +
        (yy - 50) ** 2 < 30 ** 2
    ] = 255

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    mask[
        20:80,
        20:80
    ] = 255


    obs = make_observation(
        mask,
        image=sharp
    )

    result = pipeline.run(obs)

    assert result is True


def test_pipeline_rejects_empty_mask():

    pipeline = FilterPipeline(
        [
            AreaFilter(
                minimum_ratio=0.01
            )
        ]
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    obs = make_observation(mask)

    result = pipeline.run(obs)

    assert result is False


def test_pipeline_metrics_are_populated():

    pipeline = FilterPipeline(
        [
            AreaFilter(
                minimum_ratio=0.01
            ),
            BorderFilter(
                maximum_ratio=0.01
            )
        ]
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8
    ) * 255

    obs = make_observation(mask)

    pipeline.run(obs)

    assert obs.metrics.area_ratio > 0
    assert hasattr(
        obs.metrics,
        "border_ratio"
    )