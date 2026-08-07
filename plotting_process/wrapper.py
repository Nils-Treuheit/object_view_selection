import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*random_state")
warnings.filterwarnings("ignore", message="The TBB threading layer")
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation")

from .embedding_plots import run_all as run_embedding_plots
from .embedding_plots import run_criteria_dr
from .embedding_plots.base import kmeans_cluster_labels
from .feature_plots import plot_bad_examples, plot_dataset_overview
from .misc_plot import plot_rejection_reasons
from .pre_filter_plots import plot_pre_filter_distributions
from .quality_score_plots.violins import plot_quality_violins

_DEFAULT_CLUSTERS = 10

_CRITERIA_COLUMNS = [
    "laplacian", "tenengrad",
    "area_ratio", "border_ratio", "edge_ratio",
    "hand_overlap",
    "vincent_area_fraction", "vincent_artifact_fraction",
    "vincent_boundary_blur_variance",
    "solidity", "extent", "convexity", "completeness",
    "blur", "area", "vincents_artefacts", "centerness", "confidence",
]


def _criteria_matrix(pool_obs):
    """Normalised [0, 1] metric matrix of the selection pool.

    Only criteria present on every observation are kept, so it degrades
    gracefully in standalone mode and with partial metric dicts. Each column
    is min-max scaled so no single criterion dominates the DR distances.
    """
    rows = []
    for obs in pool_obs:
        m = getattr(obs, "metrics", None)
        row = {}
        if m is not None:
            for col in _CRITERIA_COLUMNS:
                v = getattr(m, col, None)
                if v is not None:
                    row[col] = float(v)
        rows.append(row)
    if not rows:
        return None
    cols = [c for c in _CRITERIA_COLUMNS if all(c in r for r in rows)]
    if not cols:
        return None
    mat = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    lo = mat.min(axis=0)
    hi = mat.max(axis=0)
    span = hi - lo
    span[span < 1e-12] = 1.0
    return (mat - lo) / span


def _indexed_files(dirpath, suffix=".png"):
    """Map integer observation ids to their actual file paths.

    The dataset stores images as zero-padded stems (``00000.png``), so
    ``f"{oid}.png"`` is wrong. Scanning the directory once maps every id to
    its real file regardless of padding.
    """
    out = {}
    d = Path(dirpath)
    if d.is_dir():
        for p in d.glob(f"*{suffix}"):
            if p.stem.isdigit():
                out[int(p.stem)] = p
    return out


def _load_from_disk(input_dir):
    """Rebuild plot arguments from saved pipeline outputs."""
    input_dir = Path(input_dir)

    report_path = input_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No report.json found in {input_dir}")

    import json
    with open(report_path) as f:
        report = json.load(f)

    embeddings = np.load(input_dir / "embeddings.npy")
    selected_idx = np.load(input_dir / "selected_indices.npy")

    import pandas as pd
    df = pd.read_csv(input_dir / "quality.csv")
    id_to_quality = dict(zip(df["id"], df["quality"]))
    id_to_metrics = {}
    for _, row in df.iterrows():
        id_to_metrics[row["id"]] = row.to_dict()

    # embeddings.npy / selected_indices.npy are aligned to the selection pool
    # (accepted observations above the quality floor), not to all accepted.
    pool_ids = report.get("selection_pool_ids", report["accepted_ids"])
    quality_scores = np.array([id_to_quality[oid] for oid in pool_ids])
    data_root = Path(report.get("data_root", ""))
    img_map = _indexed_files(data_root / "images")
    msk_map = _indexed_files(data_root / "masks")

    class _MockMetrics:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    class _MockObs:
        def __init__(self, oid, quality, metrics_dict, rejection_reason=None):
            self.id = oid
            self.quality = quality
            self.metrics = _MockMetrics(metrics_dict)
            self.rejection_reason = rejection_reason
            self.image_path = img_map.get(oid)
            self.mask_path = msk_map.get(oid)

    accepted = [
        _MockObs(oid, id_to_quality[oid], id_to_metrics[oid])
        for oid in report["accepted_ids"]
    ]

    selected_ids_set = set(report["selected_ids"])
    selected = [o for o in accepted if o.id in selected_ids_set]

    # pool observations aligned row-for-row with embeddings.npy
    accepted_by_id = {o.id: o for o in accepted}
    pool_obs = [accepted_by_id[oid] for oid in pool_ids if oid in accepted_by_id]

    rejected = []
    rej_path = input_dir / "rejected.json"
    if rej_path.exists():
        with open(rej_path) as f:
            rej_data = json.load(f)
        rej_metrics_path = input_dir / "rejected_metrics.csv"
        if rej_metrics_path.exists():
            df_rej = pd.read_csv(rej_metrics_path)
            rej_id_to_metrics = dict(zip(df_rej["id"], [r.to_dict() for _, r in df_rej.iterrows()]))
        else:
            rej_id_to_metrics = {}
            for r in rej_data:
                rej_id_to_metrics[r["id"]] = {}
        for r in rej_data:
            rejected.append(_MockObs(
                r["id"], 0.0, rej_id_to_metrics.get(r["id"], {}),
                rejection_reason=r.get("reason"),
            ))

    return accepted, rejected, selected, embeddings, selected_idx, quality_scores, pool_obs


def _ensure_dirs(base):
    pre = base / "pre-filter"
    sel = base / "selection"
    emb_2d = sel / "embedding_space" / "2D_DR_plots"
    emb_3d = sel / "embedding_space" / "3D_DR_plots"
    crit_2d = sel / "quality_criteria" / "DR_plots" / "2D_DR_plots"
    crit_3d = sel / "quality_criteria" / "DR_plots" / "3D_DR_plots"
    for d in (pre, emb_2d, emb_3d, crit_2d, crit_3d):
        d.mkdir(parents=True, exist_ok=True)
    return pre, emb_2d, emb_3d, crit_2d, crit_3d


def plot_all(
    accepted=None, rejected=None, selected=None,
    embeddings=None, selected_idx=None, quality_scores=None,
    output_dir=None, input_dir=None,
    debug=False, single_set_plots=False,
    pool_obs=None, n_clusters=_DEFAULT_CLUSTERS,
):
    """
    Main plotting entry point.

    Call with in-memory Observation objects (accepted, rejected, selected)
    OR with input_dir pointing to saved pipeline outputs.

    output_dir is where the plots/ folder is created. If not given, uses
    input_dir (or current directory as last resort).
    """
    if input_dir is not None:
        accepted, rejected, selected, embeddings, selected_idx, quality_scores, pool_obs = \
            _load_from_disk(input_dir)

    if output_dir is None:
        output_dir = Path(input_dir) if input_dir else Path.cwd()
    output_dir = Path(output_dir)
    # the pipeline-results dir (report.json, rejected.json, ...) is the
    # input_dir when running standalone, otherwise the output_dir itself
    results_dir = Path(input_dir) if input_dir else output_dir

    if pool_obs is None:
        pool_obs = accepted

    plots_root = output_dir / "plots"
    dir_pre, dir_emb_2d, dir_emb_3d, dir_crit_2d, dir_crit_3d = _ensure_dirs(plots_root)

    print("Generating pipeline plots...")

    plot_quality_violins(
        accepted, rejected, selected,
        dir_pre, plots_root / "selection",
        single_set_plots=single_set_plots,
    )

    plot_pre_filter_distributions(accepted, rejected, dir_pre)

    plot_dataset_overview(accepted, rejected, selected, dir_pre, plots_root / "selection")
    plot_bad_examples(accepted, rejected, selected, output_dir)

    if embeddings is not None and len(embeddings) >= 2:
        cluster_labels = kmeans_cluster_labels(embeddings, n_clusters)
        run_embedding_plots(
            embeddings, selected_idx, quality_scores,
            dir_emb_2d, dir_emb_3d,
            debug=debug, cluster_labels=cluster_labels,
        )

        criteria = _criteria_matrix(pool_obs)
        if criteria is not None and len(criteria) == len(embeddings):
            run_criteria_dr(
                criteria, selected_idx, quality_scores,
                dir_crit_2d, dir_crit_3d,
                debug=debug, n_clusters=n_clusters,
            )

    if debug and pool_obs:
        from .embedded_samples_plots import plot_embedded_samples
        plot_embedded_samples(pool_obs, output_dir)

    if debug and embeddings is not None and len(embeddings) >= 2 and len(selected_idx) > 0:
        from .neighbor_plots import plot_neighbor_analysis
        plot_neighbor_analysis(
            embeddings, selected_idx, pool_obs, quality_scores,
            plots_root / "selection",
        )

    plot_rejection_reasons(results_dir, dir_pre)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Standalone plotting for obj_view_selection")
    parser.add_argument("--input_dir", required=True, help="Pipeline results directory")
    parser.add_argument("--output_dir", default=None, help="Where to create plots/ (default: input_dir)")
    parser.add_argument("--debug", action="store_true", help="Show all DR methods (not just PCA+MDS)")
    parser.add_argument("--n_clusters", type=int, default=_DEFAULT_CLUSTERS,
                        help="k-means clusters used for LDA labels and cluster-coloured plots")
    args = parser.parse_args()

    plot_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        single_set_plots=True,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
