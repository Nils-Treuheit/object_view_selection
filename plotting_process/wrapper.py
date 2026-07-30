import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*random_state")
warnings.filterwarnings("ignore", message="The TBB threading layer")
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation")

from .embedding_plots import run_all as run_embedding_plots
from .quality_score_plots.violins import plot_quality_violins
from .misc_plot import plot_rejection_reasons


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

    quality_scores = np.array([id_to_quality[oid] for oid in report["accepted_ids"]])

    class _MockMetrics:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    class _MockObs:
        def __init__(self, oid, quality, metrics_dict):
            self.id = oid
            self.quality = quality
            self.metrics = _MockMetrics(metrics_dict)

    accepted = [
        _MockObs(oid, id_to_quality[oid], id_to_metrics[oid])
        for oid in report["accepted_ids"]
    ]

    selected_ids_set = set(report["selected_ids"])
    selected = [o for o in accepted if o.id in selected_ids_set]

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
            rejected.append(_MockObs(r["id"], 0.0, rej_id_to_metrics.get(r["id"], {})))

    return accepted, rejected, selected, embeddings, selected_idx, quality_scores


def _ensure_dirs(base):
    pre = base / "pre-filter"
    sel_2d = base / "selection" / "2D_DR_plots"
    sel_3d = base / "selection" / "3D_DR_plots"
    pre.mkdir(parents=True, exist_ok=True)
    sel_2d.mkdir(parents=True, exist_ok=True)
    sel_3d.mkdir(parents=True, exist_ok=True)
    return pre, sel_2d, sel_3d


def plot_all(
    accepted=None, rejected=None, selected=None,
    embeddings=None, selected_idx=None, quality_scores=None,
    output_dir=None, input_dir=None,
    debug=False, single_set_plots=False,
):
    """
    Main plotting entry point.

    Call with in-memory Observation objects (accepted, rejected, selected)
    OR with input_dir pointing to saved pipeline outputs.

    output_dir is where the plots/ folder is created. If not given, uses
    input_dir (or current directory as last resort).
    """
    if input_dir is not None:
        accepted, rejected, selected, embeddings, selected_idx, quality_scores = \
            _load_from_disk(input_dir)

    if output_dir is None:
        output_dir = Path(input_dir) if input_dir else Path.cwd()
    output_dir = Path(output_dir)

    plots_root = output_dir / "plots"
    dir_pre, dir_sel_2d, dir_sel_3d = _ensure_dirs(plots_root)

    print("Generating pipeline plots...")

    plot_quality_violins(
        accepted, rejected, selected,
        dir_pre, plots_root / "selection",
        single_set_plots=single_set_plots,
    )

    if embeddings is not None and len(embeddings) >= 2:
        run_embedding_plots(
            embeddings, selected_idx, quality_scores,
            dir_sel_2d, dir_sel_3d,
            debug=debug,
        )

    plot_rejection_reasons(output_dir, dir_pre)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Standalone plotting for obj_view_selection")
    parser.add_argument("--input_dir", required=True, help="Pipeline results directory")
    parser.add_argument("--output_dir", default=None, help="Where to create plots/ (default: input_dir)")
    parser.add_argument("--debug", action="store_true", help="Show all DR methods (not just PCA+MDS)")
    args = parser.parse_args()

    plot_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        single_set_plots=True,
    )


if __name__ == "__main__":
    main()
