import sys
import shutil
import tempfile
from pathlib import Path

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

        accepted, rejected, selected, emb2, sidx, qs2 = _load_from_disk(out)

        check(len(accepted) == n, f"loaded {len(accepted)} accepted")
        check(len(rejected) == 1, f"loaded {len(rejected)} rejected")
        check(len(selected) == 4, f"loaded {len(selected)} selected")
        check(emb2.shape == (n, dim), f"embeddings shape {emb2.shape}")
        check(len(sidx) == 4, f"selected_idx length {len(sidx)}")
        check(len(qs2) == n, f"quality_scores length {len(qs2)}")


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
        overview = pre / "data_set_overview"
        bad = pre / "bad_examples"
        check(overview.is_dir(), "data_set_overview/ exists")
        check(bad.is_dir(), "bad_examples/ exists")

        # two overview variants per feature (fixed + relative) and one
        # bad_examples image per feature, prefixed raw_filter_ / quality_score_
        for name in ["raw_filter_laplacian", "raw_filter_area_ratio",
                     "raw_filter_vincent_boundary_blur_variance",
                     "quality_score_blur", "quality_score_occlusion", "quality_score_score"]:
            check((overview / f"{name}_fixed.png").exists(), f"data_set_overview/{name}_fixed.png saved")
            check((overview / f"{name}_relative.png").exists(), f"data_set_overview/{name}_relative.png saved")
            check((bad / f"{name}.png").exists(), f"bad_examples/{name}.png saved")

        # no legacy combined images / no non-suffixed overview images
        check(not (pre / "dataset_overview_raw.png").exists(), "no legacy dataset_overview_raw.png")
        check(not (pre / "bad_examples_raw.png").exists(), "no legacy bad_examples_raw.png")
        check(not (overview / "raw_filter_laplacian.png").exists(), "no non-suffixed overview image")

        for f in list(overview.glob("*.png")) + list(bad.glob("*.png")):
            check(f.stat().st_size > 1024, f"{f.name} has content")


def test_bad_examples_select_rejected_only():
    """bad_examples selects only filtered-out frames, worst-first, capped at 5."""
    from plotting_process.feature_plots import _worst_rejected

    embeddings, qs, sel_idx = _make_synthetic_data(30, 8, 4)
    accepted, rejected, selected = _make_mock_observations(embeddings, qs, sel_idx)

    for i, o in enumerate(rejected):
        o.metrics.area_ratio = float(i + 1) / 10.0

    picked = _worst_rejected(rejected, "area_ratio", 1, 5)
    check(len(picked) == 5, f"capped at 5 (got {len(picked)})")
    check(all(o in rejected for o, _ in picked), "only filtered-out frames are selected")
    vals = [v for _, v in picked]
    check(vals == sorted(vals), "worst (lowest) frames come first")

    picked_all = _worst_rejected(rejected, "area_ratio", 1, 100)
    check(len(picked_all) == len(rejected),
          f"k > available returns all (got {len(picked_all)})")


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

        pre = out / "bad_examples"
        check((pre / "raw_filter_laplacian.png").exists(), "raw bad_examples saved")
        check((pre / "quality_score_score.png").exists(), "quality bad_examples saved")
        check((pre / "raw_filter_laplacian.png").stat().st_size > 1024, "bad_examples has content")


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


def test_dataset_overview_inverts_flipped_features():
    """Goodness is 1.0=good / 0.0=bad even for lower-is-better features."""
    from plotting_process.feature_plots import _goodness

    # lower-is-better: a low raw value (good) must map to goodness ~1.0
    check(_goodness(0.0, 0.0, 1.0, -1) == 1.0, "border_ratio=0 (good) -> goodness 1")
    check(_goodness(1.0, 0.0, 1.0, -1) == 0.0, "border_ratio=1 (bad) -> goodness 0")
    # higher-is-better: a high raw value (good) maps to goodness ~1.0
    check(_goodness(1.0, 0.0, 1.0, 1) == 1.0, "laplacian=max (good) -> goodness 1")
    check(_goodness(0.0, 0.0, 1.0, 1) == 0.0, "laplacian=min (bad) -> goodness 0")


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


if __name__ == "__main__":
    names = [k for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for name in sorted(names):
        fn = globals()[name]
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
