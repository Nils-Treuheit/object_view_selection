"""
Correctness tests for the pre-filter threshold tuner webapp backend
(``embedding_explorer_tool/prefilter_app.py``).

Tests
- apply_knobs / config_payload (garbage + outlier knobs)
- run_prefilter on a synthetic dataset (accept/reject + reasons)
- build_report_text output
- auto_knobs / run_prefilter_auto (data-driven thresholds)
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig

from embedding_explorer_tool.prefilter_app import (
    apply_knobs,
    auto_knobs,
    build_report_text,
    config_payload,
    run_prefilter,
    run_prefilter_auto,
)

from tests.test_utils import check


def _make_synthetic_dataset():
    """Write 8 images+masks: 4 sharp (noise band) and 4 blurred (flat band)."""
    rng = np.random.default_rng(42)
    root = Path(tempfile.mkdtemp(prefix="prefilter_app_"))
    (root / "images").mkdir()
    (root / "masks").mkdir()

    def write(idx, img, mask):
        cv2.imwrite(str(root / "images" / f"{idx}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks" / f"{idx}.png"), mask)

    def circle_mask(cx, cy, r):
        yy, xx = np.mgrid[0:128, 0:128]
        return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255

    for i in range(4):
        mask = circle_mask(64 + i * 5, 64, 40)
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8).copy()
        img[mask == 0] = 0
        write(i, img, mask)
    for i in range(4, 8):
        mask = circle_mask(64, 64, 40)
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8).copy()
        img[mask == 0] = 0
        img = cv2.GaussianBlur(img, (41, 41), 0)
        img[mask == 0] = 0
        write(i, img, mask)

    return root


def test_apply_knobs():
    cfg = PipelineConfig(data_root="x")
    apply_knobs(cfg, {"blur_laplacian.hard_min_variance": 1234.0,
                      "blur_tenengrad.hard_min_tenengrad": 55.5},
                {"blur_laplacian.outlier_z": {"enabled": False, "value": 2.0},
                 "vincents_artefacts.outlier_z": {"enabled": True, "value": 4.5}},
                {"vincents_artefacts.kernel_size": 7.0,
                 "blur_laplacian.stroke_width": 5.0,
                 "vincents_area.softness": 0.25})
    check(cfg.filters.blur_laplacian.hard_min_variance == 1234.0,
          "garbage knob writes hard_min_variance")
    check(cfg.filters.blur_tenengrad.hard_min_tenengrad == 55.5,
          "garbage knob writes hard_min_tenengrad")
    check(cfg.filters.blur_laplacian.outlier_z is None,
          "disabled outlier knob sets z to None")
    check(cfg.filters.vincents_artefacts.outlier_z == 4.5,
          "enabled outlier knob sets z value")
    check(cfg.filters.vincents_artefacts.kernel_size == 7,
          "integer parameter knob stays an int")
    check(cfg.filters.blur_laplacian.stroke_width == 5,
          "integer stroke_width parameter stays an int")
    check(cfg.filters.vincents_area.softness == 0.25,
          "float parameter knob written as float")
    apply_knobs(cfg, {"does.not.exist": 1.0}, {"also.missing": {"enabled": True, "value": 2.0}},
                {"no.such_param": 3})
    check(True, "unknown knob keys are ignored")


def test_config_payload():
    cfg = PipelineConfig(data_root="x")
    apply_knobs(cfg, {}, {"blur_laplacian.outlier_z": {"enabled": False, "value": 0.0}})
    payload = config_payload(cfg)
    garbage = {k["key"]: k for k in payload["garbage"]}
    outlier = {k["key"]: k for k in payload["outlier"]}
    params = {k["key"]: k for k in payload["params"]}
    check(set(garbage) == {"blur_laplacian.hard_min_variance",
                           "blur_tenengrad.hard_min_tenengrad",
                           "vincents_artefacts.hard_max_fraction"},
          "default filter order only exposes the active floors/ceiling")
    check(set(outlier) == {"blur_laplacian.outlier_z",
                           "blur_tenengrad.outlier_z",
                           "vincents_artefacts.outlier_z"},
          "default filter order only exposes the active z-cutoffs")
    check(set(params) == {"blur_laplacian.stroke_width",
                          "blur_tenengrad.stroke_width",
                          "vincents_artefacts.kernel_size"},
          "default filter order exposes the active filter parameters")
    check(all("min" in k and "max" in k and "step" in k for k in payload["garbage"]),
          "knobs carry min/max/step bounds")
    check(outlier["blur_laplacian.outlier_z"]["enabled"] is False
          and outlier["blur_laplacian.outlier_z"]["value"] == 3.0,
          "disabled outlier serialized with fallback value 3.0")

    # Including the soft filters in the order exposes their knobs too.
    cfg.filters.filter_order = list(cfg.filters.filter_order) + ["vincents_area", "vincents_motion_blur"]
    payload = config_payload(cfg)
    keys = {k["key"] for k in payload["garbage"]}
    out_keys = {k["key"] for k in payload["outlier"]}
    param_keys = {k["key"] for k in payload["params"]}
    check({"vincents_area.hard_min_area_fraction",
           "vincents_motion_blur.hard_min_variance"} <= keys,
          "soft filters in the order add their garbage knobs")
    check({"vincents_area.outlier_z", "vincents_motion_blur.outlier_z"} <= out_keys,
          "soft filters in the order add their outlier knobs")
    check({"vincents_area.softness", "vincents_motion_blur.stroke_width",
           "vincents_motion_blur.softness"} <= param_keys,
          "soft filters in the order add their parameter knobs")


def test_run_prefilter_synthetic():
    root = _make_synthetic_dataset()
    garbage = {"blur_laplacian.hard_min_variance": 100000.0}
    text, accepted, rejected, reasons = run_prefilter(str(root), garbage)
    check(len(accepted) == 4, f"4 sharp observations accepted (got {len(accepted)})")
    check(len(rejected) == 4, f"4 blurred observations rejected (got {len(rejected)})")
    check(sorted(rejected) == [4, 5, 6, 7],
          "blurred observations are the rejected ones")
    check(reasons.get("blur_laplacian_threshold", 0) == 4,
          "rejections annotated with blur_laplacian_threshold")


def test_build_report_text():
    root = _make_synthetic_dataset()
    garbage = {"blur_laplacian.hard_min_variance": 100000.0}
    text, _acc, _rej, _reasons = run_prefilter(str(root), garbage)
    check("PRE-FILTER RUN" in text, "report header present")
    check("Filter order:" in text, "filter order section present")
    check("Garbage thresholds applied:" in text, "garbage section present")
    check("Outlier thresholds applied:" in text, "outlier section present")
    check("Filter parameters applied:" in text, "parameters section present")
    check("Rejected by filter:" in text, "rejected section present")
    check("Accepted raw stats" in text, "stats section present")
    check("100000.0000" in text, "applied knob value shown")
    check("observations: 8" in text, "observation count shown")
    check("kernel_size" in text, "active filter parameter shown")


def test_filter_order_selects_filters():
    """The soft filters only run when explicitly included in the order."""
    root = _make_synthetic_dataset()
    order_default = ["vincent_empty_mask", "vincent_border_pixel",
                     "blur_laplacian", "blur_tenengrad", "vincents_artefacts"]
    text, _acc, _rej, _reasons = run_prefilter(str(root), filter_order=order_default)
    check("vincents_area" not in text and "vincents_motion_blur" not in text,
          "default order runs no soft filters")
    check("blur_laplacian stat" in text
          and "vincents_artefacts stat" in text,
          "default order reports its own stats")
    check("vincents_area stat" not in text and "vincents_motion_blur stat" not in text,
          "default order hides the soft filter stats")

    order_with_soft = order_default + ["vincents_area", "vincents_motion_blur"]
    text2, _acc2, _rej2, _reasons2 = run_prefilter(str(root), filter_order=order_with_soft)
    check("vincents_area" in text2 and "vincents_motion_blur" in text2,
          "soft filters appear when included in the order")
    check("vincents_area stat" in text2 and "vincents_motion_blur stat" in text2,
          "soft filter stats shown when included in the order")


def test_auto_knobs():
    root = _make_synthetic_dataset()
    garbage, outlier = auto_knobs(str(root))
    check("blur_laplacian.hard_min_variance" in garbage
          and "blur_tenengrad.hard_min_tenengrad" in garbage,
          "auto knobs cover both blur floors")
    check(isinstance(garbage["blur_laplacian.hard_min_variance"], float),
          "auto laplacian floor is a float")
    check(isinstance(garbage["blur_tenengrad.hard_min_tenengrad"], float),
          "auto tenengrad floor is a float")
    check(outlier == {}, "auto tuning leaves outlier knobs untouched")

    text, accepted, rejected, reasons, garbage2, outlier2 = run_prefilter_auto(str(root))
    check(len(accepted) + len(rejected) == 8, "auto run processes all observations")
    check(garbage2 == garbage and outlier2 == outlier, "auto run reports applied knobs")
    check("PRE-FILTER RUN" in text, "auto run produces a report")
