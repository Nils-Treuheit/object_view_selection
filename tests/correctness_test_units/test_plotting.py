import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.test_utils import check


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n=50, dim=64, n_sel=5):
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float32)
    quality_scores = rng.rand(n)
    selected_idx = np.sort(rng.choice(n, n_sel, replace=False))
    return embeddings, quality_scores, selected_idx


class _MockMetrics:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _MockObs:
    def __init__(self, oid, quality, metrics_dict):
        self.id = oid
        self.quality = quality
        self.metrics = _MockMetrics(**metrics_dict)
        self.rejection_reason = None
        self.image_path = None
        self.mask_path = None


def _make_mock_observations(embeddings, quality_scores, selected_idx):
    keys = ["laplacian", "tenengrad", "area_ratio", "border_ratio",
            "edge_ratio", "hand_overlap", "completeness", "blur", "area",
            "occlusion", "confidence",
            "vincents_area", "vincents_artefacts", "vincents_motion_blur",
            "vincent_area_fraction", "vincent_artifact_fraction",
            "vincent_boundary_blur_variance"]
    accepted = []
    for i in range(len(embeddings)):
        d = {k: float(np.random.rand()) for k in keys}
        # unbounded raw stats (not [0, 1])
        d["laplacian"] = float(np.random.rand() * 500.0)
        d["tenengrad"] = float(np.random.rand() * 100.0)
        d["vincent_boundary_blur_variance"] = float(np.random.rand() * 10000.0)
        accepted.append(_MockObs(i, float(quality_scores[i]), d))
    sel_set = set(selected_idx)
    selected = [o for o in accepted if o.id in sel_set]
    rejected = []
    for i in range(10):
        rej = _MockObs(1000 + i, 0.0, {k: 0.0 for k in keys})
        rej.rejection_reason = "blur"
        rejected.append(rej)
    return accepted, rejected, selected


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_modules_import():
    """All plotting submodules can be imported."""
    try:
        from plotting_process.wrapper import plot_all
        from plotting_process.embedding_plots import run_all
        from plotting_process.embedding_plots.base import (
            _reduce_embeddings,
            draw_embedding_scatter_2d,
            draw_embedding_scatter_3d,
            render_embedding,
        )
        from plotting_process.quality_score_plots.violins import plot_quality_violins
        from plotting_process.misc_plot import plot_rejection_reasons
        check(True, "all modules imported successfully")
    except Exception as e:
        check(False, f"import failed: {e}")


def test_folder_structure_created():
    """plot_all creates the expected folder tree under plots/."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(50, 64, 5)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=False, single_set_plots=False,
        )

        plots = out / "plots"
        check(plots.is_dir(), "plots/ exists")
        check((plots / "pre-filter").is_dir(), "plots/pre-filter/ exists")
        check((plots / "selection").is_dir(), "plots/selection/ exists")
        check((plots / "selection" / "2D_DR_plots").is_dir(), "plots/selection/2D_DR_plots/ exists")
        check((plots / "selection" / "3D_DR_plots").is_dir(), "plots/selection/3D_DR_plots/ exists")


def test_debug_false_only_pca_mds():
    """Without debug, only PCA (selection_embedding*) and MDS plots appear."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(30, 64, 4)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=False, single_set_plots=False,
        )

        d2 = out / "plots" / "selection" / "2D_DR_plots"

        check((d2 / "selection_embedding.png").exists(), "PCA 2D present")
        check((d2 / "selection_embedding_scaled.png").exists(), "PCA scaled present")
        check((d2 / "embedding_mds.png").exists(), "MDS present")
        check(not (d2 / "embedding_tsne.png").exists(), "t-SNE absent without debug")
        check(not (d2 / "embedding_umap.png").exists(), "UMAP absent without debug")
        check(not (d2 / "embedding_isomap.png").exists(), "Isomap absent without debug")
        check(not (d2 / "embedding_lle.png").exists(), "LLE absent without debug")


def test_debug_true_all_methods():
    """With debug, all DR methods produce output files."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(30, 64, 4)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=True, single_set_plots=False,
        )

        d2 = out / "plots" / "selection" / "2D_DR_plots"

        for name in ["selection_embedding.png", "selection_embedding_scaled.png",
                      "embedding_mds.png", "embedding_tsne.png", "embedding_umap.png",
                      "embedding_isomap.png", "embedding_lle.png"]:
            check((d2 / name).exists(), f"{name} present (debug=True)")


def test_output_files_non_trivial_size():
    """All generated PNG/HTML files have non-trivial size (> 1 KB)."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(30, 64, 4)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=True, single_set_plots=False,
        )

        for f in out.rglob("*"):
            if f.is_file() and f.suffix in (".png", ".html"):
                sz = f.stat().st_size
                check(sz > 1024, f"{f.name} size {sz} > 1 KB")


def test_standalone_loader_valid_input():
    """_load_from_disk reads pipeline output correctly."""
    from plotting_process.wrapper import _load_from_disk

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        n, dim = 30, 64
        rng = np.random.RandomState(42)
        embeddings = rng.randn(n, dim).astype(np.float32)
        quality_scores = rng.rand(n)
        sel_idx = np.sort(rng.choice(n, 4, replace=False))

        np.save(out / "embeddings.npy", embeddings)
        np.save(out / "selected_indices.npy", sel_idx)

        import pandas as pd
        ids = list(range(n))
        df = pd.DataFrame({
            "id": ids,
            "quality": quality_scores,
            "laplacian": rng.rand(n),
            "tenengrad": rng.rand(n),
            "area_ratio": rng.rand(n),
            "border_ratio": rng.rand(n),
            "hand_overlap": rng.rand(n),
            "completeness": rng.rand(n),
            "blur": rng.rand(n),
            "area": rng.rand(n),
            "occlusion": rng.rand(n),
            "confidence": rng.rand(n),
        })
        df.to_csv(out / "quality.csv", index=False)

        import json
        report = {
            "accepted_ids": ids,
            "selected_ids": [int(ids[i]) for i in sel_idx],
        }
        with open(out / "report.json", "w") as f:
            json.dump(report, f)

        rej_data = [{"id": 999, "reason": "test"}]
        with open(out / "rejected.json", "w") as f:
            json.dump(rej_data, f)

        pd.DataFrame([{"id": 999, "laplacian": 0.1, "tenengrad": 0.2,
                        "area_ratio": 0.3, "border_ratio": 0.4,
                        "hand_overlap": 0.5, "completeness": 0.6}]
                     ).to_csv(out / "rejected_metrics.csv", index=False)

        accepted, rejected, selected, emb2, sidx, qs2, pool_obs = _load_from_disk(out)

        check(len(accepted) == n, f"loaded {len(accepted)} accepted")
        check(len(rejected) == 1, f"loaded {len(rejected)} rejected")
        check(len(selected) == 4, f"loaded {len(selected)} selected")
        check(emb2.shape == (n, dim), f"embeddings shape {emb2.shape}")
        check(len(sidx) == 4, f"selected_idx length {len(sidx)}")
        check(len(qs2) == n, f"quality_scores length {len(qs2)}")
        check(len(pool_obs) == n, f"pool_obs aligned with embeddings ({len(pool_obs)})")
        check([o.id for o in pool_obs] == ids, "pool_obs in selection-pool order")


def test_standalone_loader_missing_file():
    """_load_from_disk raises FileNotFoundError without report.json."""
    from plotting_process.wrapper import _load_from_disk

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _load_from_disk(tmp)
            check(False, "should have raised FileNotFoundError")
        except FileNotFoundError:
            check(True, "correctly raised FileNotFoundError")


def test_standalone_plot_all_runs():
    """plot_all with input_dir= produces output without crashing."""
    from plotting_process.wrapper import plot_all, _load_from_disk

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        n, dim = 20, 64
        rng = np.random.RandomState(42)
        embeddings = rng.randn(n, dim).astype(np.float32)
        qs = rng.rand(n)
        sel_idx = np.sort(rng.choice(n, 3, replace=False))

        np.save(out / "embeddings.npy", embeddings)
        np.save(out / "selected_indices.npy", sel_idx)

        import pandas as pd
        ids = list(range(n))
        df = pd.DataFrame({
            "id": ids, "quality": qs,
            "laplacian": rng.rand(n), "tenengrad": rng.rand(n),
            "area_ratio": rng.rand(n), "border_ratio": rng.rand(n),
            "hand_overlap": rng.rand(n), "completeness": rng.rand(n),
            "blur": rng.rand(n), "area": rng.rand(n),
            "occlusion": rng.rand(n), "confidence": rng.rand(n),
        })
        df.to_csv(out / "quality.csv", index=False)

        import json
        report = {"accepted_ids": ids, "selected_ids": [int(ids[i]) for i in sel_idx]}
        with open(out / "report.json", "w") as f:
            json.dump(report, f)

        out2 = Path(tempfile.mkdtemp())
        try:
            plot_all(input_dir=out, output_dir=out2, debug=False, single_set_plots=False)
            check((out2 / "plots" / "selection" / "2D_DR_plots" / "selection_embedding.png").exists(),
                  "standalone plot_all produced output")
        finally:
            shutil.rmtree(out2, ignore_errors=True)


def test_lda_fallback_single_component():
    """LDA with 2 classes produces 1 component — handled gracefully."""
    from plotting_process.embedding_plots.base import _reduce_embeddings

    rng = np.random.RandomState(0)
    embeddings = rng.randn(10, 8).astype(np.float32)
    sel_idx = np.array([0, 1, 2])

    coords, extra = _reduce_embeddings(embeddings, "lda", selected_idx=sel_idx, n_components=2)
    check(coords.shape[1] == 1, f"LDA with 2 classes → 1 component (got {coords.shape[1]})")


def test_mds_no_future_warning():
    """MDS is called with n_init=1 to suppress FutureWarning."""
    from plotting_process.embedding_plots.base import _reduce_embeddings
    import warnings

    rng = np.random.RandomState(0)
    embeddings = rng.randn(15, 8).astype(np.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coords, extra = _reduce_embeddings(embeddings, "mds", n_components=2)

    fw = [x for x in w if issubclass(x.category, FutureWarning) and "n_init" in str(x.message)]
    check(len(fw) == 0, f"no FutureWarning about n_init (got {len(fw)})")
    check(coords.shape == (15, 2), f"MDS output shape {coords.shape}")


def test_rejection_reasons_saved():
    """rejection_reasons.png is created when rejected.json exists."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(20, 8, 3)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        import json
        with open(out / "rejected.json", "w") as f:
            json.dump([{"id": 999, "reason": "blur"}], f)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=False, single_set_plots=False,
        )

        f = out / "plots" / "pre-filter" / "rejection_reasons.png"
        check(f.exists(), "rejection_reasons.png saved")
        check(f.stat().st_size > 1024, "rejection_reasons.png has content")


def test_render_embedding_unknown_method():
    """Unknown method raises ValueError."""
    from plotting_process.embedding_plots.base import _reduce_embeddings

    try:
        _reduce_embeddings(np.zeros((5, 4)), "nonexistent_method")
        check(False, "should have raised ValueError")
    except ValueError:
        check(True, "unknown method raises ValueError")


def test_feature_overview_plots_saved():
    """Per-feature images land in data_set_overview/ and bad_examples/ folders."""
    from plotting_process.wrapper import plot_all

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        embeddings, qs, sel_idx = _make_synthetic_data(40, 64, 5)
        accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

        plot_all(
            accepted=accepted, rejected=rejected, selected=selected,
            embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
            output_dir=out, debug=False, single_set_plots=False,
        )

        pre = out / "plots" / "pre-filter"
        sel = out / "plots" / "selection"
        raw_overview = pre / "data_set_overview"
        qual_overview = sel / "data_set_overview"
        bad = out / "bad_examples"
        pre_stage = bad / "pre-filter_stage"
        sel_stage = bad / "selection_stage"
        check(raw_overview.is_dir(), "pre-filter/data_set_overview/ exists")
        check(qual_overview.is_dir(), "selection/data_set_overview/ exists")
        check(pre_stage.is_dir(), "bad_examples/pre-filter_stage/ exists")
        check(sel_stage.is_dir(), "bad_examples/selection_stage/ exists")

        # raw stats -> pre-filter/data_set_overview/<feature>_filter_*.png;
        # bounded features get both variants (fixed + relative); unbounded
        # counting stats (laplacian, tenengrad, boundary-blur variance) get
        # the relative plot only.
        for name in ["laplacian_filter", "tenengrad_filter",
                     "vincent_boundary_blur_variance_filter"]:
            check(not (raw_overview / f"{name}_fixed.png").exists(),
                  f"no fixed variant for unbounded {name}")
            check((raw_overview / f"{name}_relative.png").exists(),
                  f"pre-filter/data_set_overview/{name}_relative.png saved")
        for name in ["area_ratio_filter"]:
            check((raw_overview / f"{name}_fixed.png").exists(),
                  f"pre-filter/data_set_overview/{name}_fixed.png saved")
            check((raw_overview / f"{name}_relative.png").exists(),
                  f"pre-filter/data_set_overview/{name}_relative.png saved")

        # quality scores -> selection/data_set_overview/quality_score_*.png
        for name in ["quality_score_blur", "quality_score_occlusion", "quality_score_score"]:
            check((qual_overview / f"{name}_fixed.png").exists(),
                  f"selection/data_set_overview/{name}_fixed.png saved")
            check((qual_overview / f"{name}_relative.png").exists(),
                  f"selection/data_set_overview/{name}_relative.png saved")

        # mock rejected frames all carry reason "blur" -> laplacian/tenengrad
        # get *_filtered.png; every other stat falls back to lower_*_quality.png
        check((pre_stage / "laplacian_filtered.png").exists(),
              "pre-filter_stage/laplacian_filtered.png saved (reason blur fired)")
        check(not (pre_stage / "area_ratio_filtered.png").exists(),
              "no small_object rejections -> no area_ratio_filtered.png")
        check((pre_stage / "lower_area_ratio_quality.png").exists(),
              "pre-filter_stage/lower_area_ratio_quality.png saved")
        check((sel_stage / "lower_blur_quality.png").exists(),
              "selection_stage/lower_blur_quality.png saved")
        check((sel_stage / "lower_score_quality.png").exists(),
              "selection_stage/lower_score_quality.png saved")

        # no legacy combined images / no old raw_filter_* names / no flat
        # bad_examples images / bad_examples not inside plots/pre-filter
        check(not (pre / "dataset_overview_raw.png").exists(), "no legacy dataset_overview_raw.png")
        check(not (pre / "bad_examples_raw.png").exists(), "no legacy bad_examples_raw.png")
        check(not (raw_overview / "raw_filter_laplacian.png").exists(), "no non-suffixed overview image")
        check(not (raw_overview / "raw_filter_laplacian_relative.png").exists(),
              "no old raw_filter_* names")
        check(not (pre / "bad_examples").exists(), "bad_examples is not inside plots/pre-filter")
        check(not (bad / "raw_filter_laplacian.png").exists(), "no flat bad_examples image")
        check(not (bad / "quality_score_score.png").exists(), "no flat quality bad_examples image")

        files = (list(raw_overview.glob("*.png")) + list(qual_overview.glob("*.png"))
                 + list(pre_stage.glob("*.png")) + list(sel_stage.glob("*.png")))
        for f in files:
            check(f.stat().st_size > 1024, f"{f.name} has content")


def test_bad_examples_select_rejected_only():
    """bad_examples selects only filtered-out frames, worst first, capped at 5."""
    from plotting_process.feature_plots import _curated_bad_examples

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

    for i, o in enumerate(rejected):
        o.metrics.area_ratio = float(i + 1) / 10.0

    picked = _curated_bad_examples(rejected, "area_ratio", 1, 5)
    check(len(picked) == 5, f"capped at 5 (got {len(picked)})")
    check(all(o in rejected for o, _ in picked), "only filtered-out frames are selected")
    # the very worst frame is always the first pick
    check(picked[0][0].id == rejected[0].id, "absolute worst frame comes first")

    picked_all = _curated_bad_examples(rejected, "area_ratio", 1, 100)
    check(len(picked_all) == len(rejected),
          f"k > available returns all (got {len(picked_all)})")


def test_bad_examples_curated_avoid_near_duplicates():
    """Curated picks skip a run of near-identical frames when distinct ones exist."""
    from plotting_process.feature_plots import _curated_bad_examples

    n = 10
    rng = np.random.RandomState(7)
    rejected = []
    for i in range(n):
        o = _MockObs(i, 0.0, {"area_ratio": (i + 1) / 10.0})
        o.rejection_reason = "area"
        rejected.append(o)

    # ids 0-2 share the same image, ids 3-9 all share a different image
    img_a = np.full((64, 64, 3), 10, dtype=np.uint8)
    img_b = np.full((64, 64, 3), 200, dtype=np.uint8)
    for i, o in enumerate(rejected):
        o.image = img_a if i < 3 else img_b

    picked = _curated_bad_examples(rejected, "area_ratio", 1, 5)
    picked_ids = [o.id for o, _ in picked]
    check(len(picked_ids) == 5, f"picked 5 (got {len(picked_ids)})")
    # a pure worst-first run would give ids [0,1,2,3,4]; curation must not
    # return three identical-looking frames
    check(picked_ids != [0, 1, 2, 3, 4],
          f"curated picks avoid a near-identical run (got {picked_ids})")


def test_bad_examples_placeholders_when_few_rejected():
    """bad_examples writes a placeholder tile when fewer than 5 are rejected."""
    from plotting_process.feature_plots import plot_bad_examples

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)
    rejected = rejected[:2]  # fewer than the default 5

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pre-filter"
        out.mkdir(parents=True)
        plot_bad_examples(accepted, rejected, selected, out)

        pre_stage = out / "bad_examples" / "pre-filter_stage"
        sel_stage = out / "bad_examples" / "selection_stage"
        check((pre_stage / "laplacian_filtered.png").exists(), "pre-filter_stage bad_examples saved")
        check((sel_stage / "lower_score_quality.png").exists(), "selection_stage bad_examples saved")
        check((pre_stage / "laplacian_filtered.png").stat().st_size > 1024, "bad_examples has content")


def test_report_value_inverts_lower_is_better_features():
    """_report_value reports 1 - value for lower-is-better features."""
    from plotting_process.feature_plots import _report_value, _FEATURE_DIRECTION

    for attr, gd in [("border_ratio", -1), ("edge_ratio", -1),
                     ("hand_overlap", -1), ("vincent_artifact_fraction", -1)]:
        check(_FEATURE_DIRECTION[attr] == -1, f"{attr} is lower-is-better")
        check(abs(_report_value(attr, 0.06) - 0.94) < 1e-9, f"{attr} reported as 1 - value")
        check(_report_value(attr, 0.0) == 1.0, f"{attr} best case reports 1.0")

    for attr, gd in [("laplacian", 1), ("area_ratio", 1), ("score", 1)]:
        check(_FEATURE_DIRECTION.get(attr, 1) == 1, f"{attr} is higher-is-better")
        check(abs(_report_value(attr, 0.06) - 0.06) < 1e-9, f"{attr} reported unchanged")


def test_format_tick_never_uses_scientific_notation():
    """_format_tick writes numbers fully (no e-notation)."""
    from plotting_process.feature_plots import _format_tick

    check(_format_tick(4571.75) == "4571.75", "laplacian tick fully written")
    check(_format_tick(50.9) == "50.9", "large-ish value written out")
    check(_format_tick(0.064) == "0.064", "small fraction written out")
    check(_format_tick(1.0) == "1", "whole number has no trailing zeros")
    check(_format_tick(0.0) == "0", "zero")
    check("e+" not in _format_tick(123456.789), "no e+ notation for big values")


def test_bad_examples_unhashable_observations():
    """_curated_bad_examples works with unhashable observations (real Observation)."""
    from dataclasses import dataclass, field
    from plotting_process.feature_plots import _curated_bad_examples

    @dataclass
    class _UnhashableObs:
        id: int
        image: object = None
        rejection_reason: str = "area"

        @property
        def metrics(self):
            return _MockMetrics(area_ratio=float(self.id + 1) / 10.0)

    rejected = [_UnhashableObs(i) for i in range(6)]
    picked = _curated_bad_examples(rejected, "area_ratio", 1, 5)
    check(len(picked) == 5, f"unhashable observations picked 5 (got {len(picked)})")
    check(picked[0][0].id == 0, "worst unhashable frame first")


def test_bad_examples_filtered_reason_scoped():
    """*_filtered.png shows only frames rejected for that feature's reason."""
    from unittest import mock
    from plotting_process.feature_plots import plot_bad_examples

    accepted = [_MockObs(i, 0.8, {"area_ratio": 0.9, "border_ratio": 0.2, "laplacian": 50.0})
                for i in range(20)]
    rejected = []
    for i in range(3):
        o = _MockObs(100 + i, 0.0, {"area_ratio": 0.01, "laplacian": 10.0})
        o.rejection_reason = "small_object"
        rejected.append(o)
    for i in range(4):
        o = _MockObs(200 + i, 0.0, {"area_ratio": 0.8, "laplacian": 2.0})
        o.rejection_reason = "blur"
        rejected.append(o)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with mock.patch("plotting_process.feature_plots._save_example_row") as mock_save:
            plot_bad_examples(accepted, rejected, [], out)
            calls = {Path(c.args[-1]).name: c.args[0] for c in mock_save.call_args_list}

        check("area_ratio_filtered.png" in calls, "area_ratio_filtered.png produced")
        check("laplacian_filtered.png" in calls, "laplacian_filtered.png produced")
        check({o.id for o, _ in calls["area_ratio_filtered.png"]} == {100, 101, 102},
              "area_ratio_filtered only small_object frames")
        check({o.id for o, _ in calls["laplacian_filtered.png"]} == {200, 201, 202, 203},
              "laplacian_filtered only blur frames")
        # a feature whose reason never fired must not get a *_filtered.png,
        # and must instead fall back to lower_*_quality.png from accepted
        check("border_ratio_filtered.png" not in calls, "no border_ratio_filtered.png")
        check("lower_border_ratio_quality.png" in calls, "border_ratio fell back to lower_*_quality.png")
        check(all(o.id in range(20) for o, _ in calls["lower_border_ratio_quality.png"]),
              "lower_*_quality samples accepted frames, not rejected")
        check("lower_area_ratio_quality.png" not in calls,
              "no lower_*_quality when the reason fired")


def test_bad_examples_status_lines_and_rel_range():
    """Status lines follow the per-stage format and borders use the rel range."""
    from unittest import mock
    from plotting_process.feature_plots import plot_bad_examples, REL_CMAP, _goodness

    accepted = [_MockObs(i, 0.8, {"area_ratio": 0.9, "border_ratio": 0.2, "laplacian": 50.0})
                for i in range(20)]
    rejected = []
    for i in range(2):
        o = _MockObs(100 + i, 0.0, {"area_ratio": 0.01, "laplacian": 10.0})
        o.rejection_reason = "small_object"
        rejected.append(o)

    with mock.patch("plotting_process.feature_plots._save_example_row") as mock_save:
        with tempfile.TemporaryDirectory() as tmp:
            plot_bad_examples(accepted, rejected, [], Path(tmp))
        calls = {Path(c.args[-1]).name: c for c in mock_save.call_args_list}
        by_path = {str(Path(c.args[-1])): c for c in mock_save.call_args_list}

    # filtered stage: rejected - <reason>, with the id/QoS format applied later
    c = calls["area_ratio_filtered.png"]
    obs = c.args[0][0][0]
    status = c.kwargs["status_line"](obs, 0.01)
    check(status == f"rejected - {obs.rejection_reason}", f"filtered status line (got '{status}')")

    # pre-filter fallback (accepted frames): accepted - <feature label>
    c = calls["lower_border_ratio_quality.png"]
    acc = c.args[0][0][0]
    status = c.kwargs["status_line"](acc, 0.8)
    check(status == "accepted - Border-Free Ratio", f"pre-filter fallback status line (got '{status}')")

    # selection stage: accepted but not selected
    sel_calls = [cc for path, cc in by_path.items() if "selection_stage" in path]
    check(len(sel_calls) > 0, "selection-stage images produced")
    status = sel_calls[0].kwargs["status_line"](accepted[0], 0.5)
    check(status == "accepted but not selected", f"selection-stage status line (got '{status}')")

    # rel_range spans the reported values of ALL samples (accepted + rejected)
    lo, hi = calls["area_ratio_filtered.png"].kwargs["rel_range"]
    check(lo <= 0.01 and hi >= 0.9, f"area_ratio rel range covers all samples (got {(lo, hi)})")
    lo, hi = calls["lower_border_ratio_quality.png"].kwargs["rel_range"]
    check(lo <= 0.8 and hi >= 0.8 and lo <= hi,
          f"border_ratio rel range covers accepted values (got {(lo, hi)})")
    # border colour is the viridis colour of the relative score
    color = REL_CMAP(_goodness(0.8, 0.0, 1.0, 1))
    check(len(color) == 4, "border colour from the viridis cmap")


def test_prob_sample_low_quality_prefers_worst():
    """Probability sampling favours the lowest-quality frames."""
    from plotting_process.feature_plots import _prob_sample_low_quality

    pool = [_MockObs(i, 0.0, {"laplacian": v}) for i, v in enumerate([3.0, 1.0, 0.0, 2.0])]
    picked_vals = []
    for seed in range(400):
        picked = _prob_sample_low_quality(pool, "laplacian", 1, 1,
                                          rng=np.random.default_rng(seed))
        picked_vals.append(picked[0][1])
    # uniform sampling would average 1.5; the prob sample must sit far below
    check(float(np.mean(picked_vals)) < 1.1,
          f"sampled frames are low-quality on average (mean {np.mean(picked_vals):.3f})")

    picked2 = _prob_sample_low_quality(pool, "laplacian", 1, 2,
                                       rng=np.random.default_rng(0))
    check(len(picked2) == 2, "sample size respected")
    check(len({o.id for o, _ in picked2}) == 2, "sampled without replacement")


def test_histogram_bars_centered_on_bin_values():
    """Bounded-feature histograms centre bars on the fixed 0.0..1.0 grid."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from plotting_process.feature_plots import _plot_metric_row

    fixed_bins = np.linspace(0.0, 1.0 + 0.025, 42)
    # accepted values span the full 0..1 range; one rejected frame sits at 0.0
    accepted = [_MockObs(i, 0.5, {"area_ratio": float(i) / 9.0}) for i in range(10)]
    rej = _MockObs(100, 0.0, {"area_ratio": 0.0})
    rej.rejection_reason = "small_object"
    rejected = [rej]

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2)
    _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, [],
                     "area_ratio", "Area Ratio", 1,
                     plt.cm.coolwarm, Normalize(0.0, 1.0), "goodness",
                     bins=fixed_bins)
    centers = np.array(sorted(p.get_x() + p.get_width() / 2.0 for p in ax_hist.patches))
    check(len(centers) > 0, "histogram produced bars")
    w = 1.0 / 40
    # every bar sits on the 0.0, 0.025, ..., 1.0 grid
    check(np.allclose(centers / w, np.round(centers / w), atol=1e-6),
          "all bars centred on the 0..1 value grid")
    check(abs(centers.min()) < 1e-9, f"0.0 bar centred on 0.0 (got {centers.min():.6f})")
    check(abs(centers.max() - 1.0) < 1e-9, f"1.0 bar centred on 1.0 (got {centers.max():.6f})")
    plt.close(fig)

    # without explicit bins (unbounded features) the grid is data-driven but
    # still centred on the data minimum (the rejected 0.0 frame)
    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted2, rejected2, _ = _make_mock_observations(embeddings, qs, sel_idx)
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2)
    _plot_metric_row(ax_hist, ax_scatter, accepted2, rejected2, [],
                     "area_ratio", "Area Ratio", 1,
                     plt.cm.coolwarm, Normalize(0.0, 1.0), "goodness")
    centers2 = np.array(sorted(p.get_x() + p.get_width() / 2.0 for p in ax_hist.patches))
    check(abs(centers2.min()) < 1e-6, f"data-driven bars centred at the data min (got {centers2.min():.6f})")
    plt.close(fig)


def test_dataset_overview_hist_xlim():
    """Histogram x-axis is [-0.05, 1.05] for [0, 1]-bounded features only."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from plotting_process.feature_plots import _plot_metric_row, FIXED_HIST_XLIM

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

    # 0-1 bounded feature (area_ratio) -> fixed xlim
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2)
    _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                     "area_ratio", "Area Ratio", 1,
                     plt.cm.coolwarm, Normalize(0.0, 1.0), "goodness")
    lo, hi = ax_hist.get_xlim()
    check(abs(lo - FIXED_HIST_XLIM[0]) < 1e-9, f"area_ratio lower xlim {lo}")
    check(abs(hi - FIXED_HIST_XLIM[1]) < 1e-9, f"area_ratio upper xlim {hi}")
    plt.close(fig)

    # unbounded feature (laplacian variance) -> not fixed to [0, 1]
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2)
    _plot_metric_row(ax_hist, ax_scatter, accepted, rejected, selected,
                     "laplacian", "Laplacian", 1,
                     plt.cm.coolwarm, Normalize(0.0, 1.0), "goodness")
    lo, hi = ax_hist.get_xlim()
    check(hi > 1.05, f"laplacian upper xlim {hi} not clipped to [0, 1]")
    plt.close(fig)


def test_plot_metric_row_colorbar_ticks():
    """colorbar_ticks replace the default tick labels on the colourbar."""
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from plotting_process.feature_plots import _plot_metric_row

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2)
    _plot_metric_row(
        ax_hist, ax_scatter, accepted, rejected, selected,
        "area_ratio", "Area Ratio", 1,
        plt.cm.viridis, Normalize(0.0, 1.0), "goodness (relative scale)",
        color_of=lambda v: v,
        colorbar_ticks=[(0.0, "0"), (0.5, "0.5"), (1.0, "1")],
    )
    cbar_ax = fig.axes[-1]
    labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    check("0" in labels and "1" in labels,
          f"colorbar shows custom raw-value ticks (got {labels})")
    plt.close(fig)


def test_dataset_overview_inverts_flipped_features():
    """Goodness is 1.0=good / 0.0=bad even for lower-is-better features."""
    from plotting_process.feature_plots import _goodness

    # lower-is-better: a low raw value (good) must map to goodness ~1.0
    check(_goodness(0.0, 0.0, 1.0, -1) == 1.0, "border_ratio=0 (good) -> goodness 1")
    check(_goodness(1.0, 0.0, 1.0, -1) == 0.0, "border_ratio=1 (bad) -> goodness 0")
    # higher-is-better: a high raw value (good) maps to goodness ~1.0
    check(_goodness(1.0, 0.0, 1.0, 1) == 1.0, "laplacian=max (good) -> goodness 1")
    check(_goodness(0.0, 0.0, 1.0, 1) == 0.0, "laplacian=min (bad) -> goodness 0")


def test_fixed_scale_uses_absolute_value_for_bounded_features():
    """Fixed 0..1 plots colour bounded features by their absolute value."""
    from plotting_process.feature_plots import _abs_goodness, BOUNDED_FEATURES

    for attr in ["blur", "occlusion", "confidence", "score", "area_ratio"]:
        check(attr in BOUNDED_FEATURES, f"{attr} is a bounded feature")
    for attr in ["laplacian", "tenengrad", "vincent_boundary_blur_variance"]:
        check(attr not in BOUNDED_FEATURES, f"{attr} is unbounded")

    # higher-is-better: the colour equals the raw value, so a 0.99 quality dot
    # is coloured near the top (warm end) of the fixed 0..1 colourbar
    check(abs(_abs_goodness(0.99, 1) - 0.99) < 1e-9, "quality 0.99 -> colour 0.99")
    check(abs(_abs_goodness(0.2, 1) - 0.2) < 1e-9, "quality 0.2 -> colour 0.2")
    # lower-is-better (border_ratio): a small bad value -> large goodness
    check(abs(_abs_goodness(0.06, -1) - 0.94) < 1e-9, "border_ratio 0.06 -> colour 0.94")
    check(_abs_goodness(1.5, 1) == 1.0, "value > 1 clips to 1.0")
    check(_abs_goodness(-0.5, 1) == 0.0, "value < 0 clips to 0.0")


def test_relative_colorbar_range_rounded():
    """Relative colourbar range rounds min down / max up to 2 decimal places."""
    from plotting_process.feature_plots import _round_range

    lo, hi = _round_range(0.0042, 0.9698)
    check(lo == 0.0, f"min rounded down to 2 dp (got {lo})")
    check(hi == 0.97, f"max rounded up to 2 dp (got {hi})")

    lo, hi = _round_range(6.486, 50.938)
    check(lo == 6.48 and hi == 50.94, f"2-dp outward rounding (got {lo}, {hi})")

    # degenerate all-identical values still produce a usable range
    lo, hi = _round_range(0.5, 0.5)
    check(hi > lo, f"degenerate range widened (got {lo}, {hi})")


def test_relative_colorbar_ticks_max_3_decimals():
    """Relative colourbar tick labels never exceed 3 decimal places."""
    from plotting_process.feature_plots import _format_relative_tick

    for v in [0.0, 0.5, 0.475, 0.975, 50.938, 0.0001]:
        label = _format_relative_tick(v)
        check("e+" not in label and "e-" not in label, f"{label} not e-notation")
        frac = label.split(".")[1] if "." in label else ""
        check(len(frac) <= 3, f"{label} has at most 3 decimal places")
    check(_format_relative_tick(0.5) == "0.5", "trailing zeros trimmed")


def test_relative_colorbar_range_rounded_and_ticks():
    """plot_dataset_overview passes the rounded range to the relative colourbar."""
    from unittest import mock
    from plotting_process import feature_plots

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

    with mock.patch.object(feature_plots, "_plot_metric_row") as mr:
        with tempfile.TemporaryDirectory() as tmp:
            feature_plots.plot_dataset_overview(accepted, rejected, selected,
                                                Path(tmp) / "plots" / "pre-filter",
                                                Path(tmp) / "plots" / "selection")
        rel_calls = [c for c in mr.call_args_list
                     if c.kwargs.get("colorbar_ticks") is not None]
        check(len(rel_calls) > 0, "relative variants produced with ticks")
        positions = [pos for pos, _ in rel_calls[0].kwargs["colorbar_ticks"]]
        check(sorted(positions) == [0.0, 0.5, 1.0], f"tick positions on goodness scale (got {positions})")
        for _, label in rel_calls[0].kwargs["colorbar_ticks"]:
            frac = label.split(".")[1] if "." in label else ""
            check(len(frac) <= 3, f"tick label {label} has <= 3 decimals")


def test_feature_plots_warm_cold_colormap():
    """The feature plots use matplotlib's coolwarm (cold=low, warm=high)."""
    from plotting_process.feature_plots import WARM_COLD_CMAP

    cold = WARM_COLD_CMAP(0.0)
    mid = WARM_COLD_CMAP(0.5)
    warm = WARM_COLD_CMAP(1.0)
    check(cold[2] > cold[0], "cold end is blue-dominant")
    check(warm[0] > warm[2], "warm end is red-dominant")
    check(abs(cold[0] - mid[0]) > 0.3, "cold and mid differ clearly")


def test_3d_html_has_content():
    """_draw_embedding_scatter_3d produces non-trivial HTML."""
    from plotting_process.embedding_plots.base import draw_embedding_scatter_3d

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_3d.html"
        n = 30
        rng = np.random.RandomState(42)
        coords = rng.randn(n, 3)
        qs = rng.rand(n)
        sel_idx = np.sort(rng.choice(n, 4, replace=False))

        draw_embedding_scatter_3d(
            path, coords, qs, sel_idx,
            "Test 3D Plot",
        )

        check(path.exists(), "3D HTML file exists")
        content = path.read_text(encoding="utf-8")
        check(len(content) > 5000, f"3D HTML content length {len(content)} > 5000")
        check("Plotly" in content, "HTML contains Plotly JS")


def test_neighbor_plots_debug_gated():
    """selected_neighbors_* and selected_clusters_pca appear only with debug."""
    from plotting_process.wrapper import plot_all

    for debug in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            embeddings, qs, sel_idx = _make_synthetic_data(30, 64, 4)
            accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

            plot_all(
                accepted=accepted, rejected=rejected, selected=selected,
                embeddings=embeddings, selected_idx=sel_idx, quality_scores=qs,
                output_dir=out, debug=debug, single_set_plots=False,
            )

            sel_dir = out / "plots" / "selection"
            for name in ["selected_neighbors_knn.png",
                         "selected_neighbors_kmeans.png",
                         "selected_clusters_pca.png"]:
                f = sel_dir / name
                if debug:
                    check(f.exists(), f"{name} present with debug=True")
                    check(f.stat().st_size > 1024, f"{name} has content")
                else:
                    check(not f.exists(), f"{name} absent without debug")


def test_selected_samples_export():
    """save_selected_samples copies the selected tuples into selected_samples/<obj_id>/."""
    from data_io.observation import Observation
    from run import save_selected_samples

    img = np.full((32, 32, 3), 120, dtype=np.uint8)
    msk = np.full((32, 32), 255, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bottle_obj"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)
        (root / "object_hands").mkdir(parents=True)
        (root / "depth").mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(root / "images" / f"{i:05d}.png"), img)
            cv2.imwrite(str(root / "masks" / f"{i:05d}.png"), msk)
            cv2.imwrite(str(root / "object_hands" / f"{i:05d}.png"), msk)
            np.save(root / "depth" / f"{i:05d}.npy", img.astype(np.float32))

        out = Path(tmp) / "results"
        selected = [
            Observation(
                id=i,
                image_path=root / "images" / f"{i:05d}.png",
                mask_path=root / "masks" / f"{i:05d}.png",
                object_hand_path=root / "object_hands" / f"{i:05d}.png",
                image=cv2.cvtColor(cv2.imread(str(root / "images" / f"{i:05d}.png")),
                                   cv2.COLOR_BGR2RGB),
                mask=msk,
                object_hand=msk,
            )
            for i in range(3)
        ]

        base = save_selected_samples(selected, str(root), out)
        check(base.name == "bottle_obj", f"obj_id folder named after data_root (got {base.name})")
        for i in range(3):
            check((base / "rgb" / f"{i:05d}.png").exists(), f"rgb/{i:05d}.png saved")
            check((base / "mask" / f"{i:05d}.png").exists(), f"mask/{i:05d}.png saved")
            check((base / "depth" / f"{i:05d}.npy").exists(), f"depth/{i:05d}.npy saved")
            check((base / "hand_mask" / f"{i:05d}.png").exists(), f"hand_mask/{i:05d}.png saved")

    # dataset without depth / hand data: those subfolders are skipped
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "nodepth"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)
        cv2.imwrite(str(root / "images" / "00000.png"), img)
        cv2.imwrite(str(root / "masks" / "00000.png"), msk)

        out = Path(tmp) / "out2"
        selected = [Observation(
            id=0,
            image_path=root / "images" / "00000.png",
            mask_path=root / "masks" / "00000.png",
            object_hand_path=None,
        )]
        base = save_selected_samples(selected, str(root), out)
        check((base / "rgb" / "00000.png").exists(), "rgb saved without depth data")
        check(not (base / "depth").exists(), "no depth folder without depth data")
        check(not (base / "hand_mask").exists(), "no hand_mask folder without hand data")


def test_filter_order_cli_override():
    """--filter_order reorders the pre-filter pipeline; default is the config order."""
    import run
    from config import PipelineConfig

    cfg = PipelineConfig()
    check(cfg.filters.filter_order == [
        "vincent_empty_mask", "vincent_border_pixel",
        "border", "area", "confidence", "blur", "occlusion", "completeness",
    ], "default filter order is the current setup")

    built = run.build_filters(cfg)
    names = [type(f).__name__ for f in built.filters]
    check(names == ["VincentEmptyMaskFilter", "VincentBorderPixelFilter",
                    "BorderFilter", "AreaFilter", "ConfidenceFilter",
                    "BlurFilter", "OcclusionFilter", "CompletenessFilter"],
          f"default pipeline built in order (got {names})")

    cfg.filters.filter_order = ["blur", "area"]
    built = run.build_filters(cfg)
    check([type(f).__name__ for f in built.filters] == ["BlurFilter", "AreaFilter"],
          "overridden order respected")


if __name__ == "__main__":
    names = [k for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for name in sorted(names):
        fn = globals()[name]
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
