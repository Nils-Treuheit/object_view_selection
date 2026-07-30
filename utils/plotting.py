import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def _build_df(observations, score_keys, label):
    rows = []
    for obs in observations:
        row = {"id": obs.id}
        m = obs.metrics
        for k in score_keys:
            row[k] = getattr(m, k, None)
        row["quality"] = obs.quality
        rows.append(row)
    df = pd.DataFrame(rows)
    df["group"] = label
    return df


def _plot_violin_set(
    df,
    score_keys,
    title,
    path,
    group_col=None,
    palette=None,
):
    n = len(score_keys)
    fig, axes = plt.subplots(1, n, figsize=(3 * n + 1, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, score_keys):
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            positions = []
            data = []
            labels = []
            for i, g in enumerate(groups):
                vals = df.loc[df[group_col] == g, key].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(i + 1)
                    labels.append(g)
            if data:
                vp = ax.violinplot(data, positions, showmeans=True, showmedians=False)
                for j, body in enumerate(vp["bodies"]):
                    color = palette.get(labels[j], "#4ecdc4") if palette else "#4ecdc4"
                    body.set_facecolor(color)
                    body.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
        else:
            vals = df[key].dropna().values
            if len(vals) > 0:
                ax.violinplot(vals, positions=[1], showmeans=True, showmedians=False)
                ax.set_xticks([1])
                ax.set_xticklabels([key], fontsize=8)

        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Score" if key == score_keys[0] else "")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_quality_violins(accepted, rejected, selected, output_dir):
    score_keys = ["blur", "area", "occlusion", "completeness", "confidence", "quality"]
    palette = {"selected": "#e76f51", "non-selected": "#264653"}
    rejected_palette = {"rejected": "#e9c46a"}

    selected_ids = {s.id for s in selected}
    non_sel = [o for o in accepted if o.id not in selected_ids]

    # ---- Plot 1: non-selected quality distribution ----
    df_ns = _build_df(non_sel, score_keys, "non-selected")
    if len(df_ns) > 1:
        _plot_violin_set(
            df_ns, score_keys,
            "Non-Selected — Quality Score Distribution",
            output_dir / "violin_non_selected.png",
        )

    # ---- Plot 2: selected quality distribution ----
    df_sel = _build_df(selected, score_keys, "selected")
    if len(df_sel) > 1:
        _plot_violin_set(
            df_sel, score_keys,
            "Selected — Quality Score Distribution",
            output_dir / "violin_selected.png",
        )

    # ---- Combined: selected vs non-selected side-by-side per score ----
    combined = pd.concat([df_ns, df_sel], ignore_index=True)
    if len(combined) > 1:
        fig, axes = plt.subplots(1, len(score_keys), figsize=(3 * len(score_keys) + 1, 4.5))
        for ax, key in zip(axes, score_keys):
            groups = combined["group"].unique()
            data = []
            positions = []
            labels = []
            for i, g in enumerate(["non-selected", "selected"]):
                vals = combined.loc[combined["group"] == g, key].dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(i + 1)
                    labels.append(g)
            if data:
                vp = ax.violinplot(data, positions, showmeans=True, showmedians=False)
                for j, body in enumerate(vp["bodies"]):
                    body.set_facecolor(palette.get(labels[j], "#4ecdc4"))
                    body.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(key, fontsize=10)
            ax.set_ylabel("Score" if key == score_keys[0] else "")
            ax.set_ylim(-0.05, 1.05)
        fig.suptitle("Selected vs Non-Selected — Quality Scores", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / "violin_selected_vs_non_selected.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir / 'violin_selected_vs_non_selected.png'}")

    # ---- Plot 3: rejected raw metrics ----
    rejected_raw_keys = ["laplacian", "tenengrad", "area_ratio", "border_ratio", "hand_overlap", "completeness"]
    readable_labels = {
        "laplacian": "Laplacian", "tenengrad": "Tenengrad",
        "area_ratio": "Area Ratio", "border_ratio": "Border Ratio",
        "hand_overlap": "Hand Overlap", "completeness": "Completeness",
    }
    df_rej = _build_df(rejected, rejected_raw_keys, "rejected")
    df_acc = _build_df(accepted, rejected_raw_keys, "accepted")
    combined_raw = pd.concat([df_rej, df_acc], ignore_index=True)

    if len(combined_raw) > 1:
        n_raw = len(rejected_raw_keys)
        fig, axes = plt.subplots(1, n_raw, figsize=(3 * n_raw + 1, 4.5))
        for ax, key in zip(axes, rejected_raw_keys):
            for i, (g, color) in enumerate([("rejected", "#e9c46a"), ("accepted", "#2a9d8f")]):
                vals = combined_raw.loc[combined_raw["group"] == g, key].dropna().values
                if len(vals) > 1:
                    vp = ax.violinplot(vals, positions=[i + 1], showmeans=False, showmedians=True)
                    vp["bodies"][0].set_facecolor(color)
                    vp["bodies"][0].set_alpha(0.6)
                    vp["cmedians"].set_color(color)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["rejected", "accepted"], fontsize=8)
            ax.set_title(readable_labels.get(key, key), fontsize=10)
        fig.suptitle("Rejected vs Accepted — Pre-Filter Raw Metrics", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / "violin_rejected_vs_accepted.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir / 'violin_rejected_vs_accepted.png'}")


def plot_selection_process(accepted, selected, embeddings, selected_idx, quality_scores, output_dir):
    from sklearn.decomposition import PCA

    n = len(embeddings)
    if n < 2:
        return

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(embeddings)
    var_ratio = pca.explained_variance_ratio_

    selected_set = set(selected_idx)
    sel_coords = coords[list(selected_idx)]
    sel_qual = quality_scores[list(selected_idx)]
    non_sel_mask = np.ones(n, dtype=bool)
    non_sel_mask[list(selected_idx)] = False
    non_sel_coords = coords[non_sel_mask]
    non_sel_qual = quality_scores[non_sel_mask]

    dist_to_sel = pairwise_distances(coords, sel_coords, metric="cosine")
    nearest_sel = dist_to_sel.argmin(axis=1)

    # ---- Figure: embedding scatter + selection ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sc = ax.scatter(
        non_sel_coords[:, 0], non_sel_coords[:, 1],
        c=non_sel_qual, cmap="viridis", s=20, alpha=0.5, vmin=0, vmax=1,
        label="non-selected",
    )

    cmap = plt.get_cmap("viridis")
    sel_colors = cmap(sel_qual)
    ax.scatter(
        sel_coords[:, 0], sel_coords[:, 1],
        c=sel_colors, s=100, marker="o", edgecolors="red", linewidths=1.5,
        label="selected", zorder=5,
    )

    for i, idx in enumerate(selected_idx):
        ax.annotate(
            str(i + 1),
            coords[idx],
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold", color="red",
        )

    sel_idx_arr = list(selected_idx)
    for i in range(len(coords)):
        if i in selected_set:
            continue
        ns = nearest_sel[i]
        target = sel_coords[ns]
        ax.plot(
            [coords[i, 0], target[0]], [coords[i, 1], target[1]],
            color="gray", alpha=0.08, linewidth=0.5,
        )

    cbar = fig.colorbar(sc, ax=ax, label="Quality Score")
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)")
    ax.set_title("View Selection — Embedding Space (PCA)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "selection_embedding.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'selection_embedding.png'}")

    # ---- Summary bar: rejection reasons ----
    accepted_ids_set = {o.id for o in accepted}
    from pathlib import Path
    rej_path = output_dir / "rejected.json"
    if rej_path.exists():
        import json
        with open(rej_path) as f:
            rej_data = json.load(f)
        reasons = {}
        for r in rej_data:
            reason = r.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            labels = list(reasons.keys())
            counts = list(reasons.values())
            colors_bars = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            ax2.barh(labels, counts, color=colors_bars)
            ax2.set_xlabel("Count")
            ax2.set_title("Rejection Reasons")
            for i, v in enumerate(counts):
                ax2.text(v + 0.3, i, str(v), va="center", fontsize=9)
            fig2.tight_layout()
            fig2.savefig(output_dir / "rejection_reasons.png", dpi=150)
            plt.close(fig2)
            print(f"  Saved {output_dir / 'rejection_reasons.png'}")


def plot_all(accepted, rejected, selected, embeddings, selected_idx, quality_scores, output_dir):
    print("Generating pipeline plots...")
    plot_quality_violins(accepted, rejected, selected, output_dir)
    plot_selection_process(accepted, selected, embeddings, selected_idx, quality_scores, output_dir)
    print("Done.")
