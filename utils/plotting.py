import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def _build_df(observations, score_keys, label):
    rows = []
    for obs in observations:
        row = {"id": obs.id}
        m = obs.metrics
        for k in score_keys:
            if k == "border_free":
                row[k] = 1.0 - getattr(m, "border_ratio", 0.0)
            elif k == "hand_overlap_free":
                row[k] = 1.0 - getattr(m, "hand_overlap", 0.0)
            else:
                row[k] = getattr(m, k, None)
        row["quality"] = obs.quality
        row["score"] = obs.quality
        rows.append(row)
    df = pd.DataFrame(rows)
    df["group"] = label
    return df


def _create_violin_figure(n, sep_index):
    has_spacer = sep_index is not None and sep_index < n - 1
    if has_spacer:
        n_cols = n + 1
        ratios = [1.0] * n_cols
        ratios[sep_index + 1] = 0.08
    else:
        n_cols = n
        ratios = None

    extra_w = 0.08 if has_spacer else 0
    fig = plt.figure(figsize=(3 * n + 1 + extra_w, 4.5))

    if has_spacer:
        gs = GridSpec(1, n_cols, width_ratios=ratios, figure=fig)
        axes_raw = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        sp = axes_raw[sep_index + 1]
        for spine in sp.spines.values():
            spine.set_visible(False)
        sp.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        sp.set_xlim(0, 1)
        sp.set_ylim(0, 1)
        sp.axvline(0.5, color="gray", linewidth=0.8, linestyle="-")
        axes = []
        for i in range(n):
            if i <= sep_index:
                axes.append(axes_raw[i])
            else:
                axes.append(axes_raw[i + 1])
        fig._ax_spacer = axes_raw[sep_index + 1]
    else:
        axes_raw = [fig.add_subplot(1, n_cols, i + 1) for i in range(n_cols)]
        axes = axes_raw
        fig._ax_spacer = None

    return fig, axes


def _plot_violin_set(
    df,
    score_keys,
    title,
    path,
    group_col=None,
    palette=None,
    sep_index=None,
):
    n = len(score_keys)
    fig, axes = _create_violin_figure(n, sep_index)
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


def _plot_violin_set_scaled(
    df,
    score_keys,
    title,
    path,
    palette=None,
    sep_index=None,
):
    n = len(score_keys)
    fig, axes = _create_violin_figure(n, sep_index)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, score_keys):
        vals = df[key].dropna().values
        if len(vals) == 0:
            continue
        ax.violinplot(vals, positions=[1], showmeans=True, showmedians=False)

        vmin, vmax = vals.min(), vals.max()
        margin = max((vmax - vmin) * 0.15, 0.01)
        ax.set_ylim(vmin - margin, vmax + margin)

        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Score" if key == score_keys[0] else "")
        ax.set_xticks([1])
        ax.set_xticklabels([key], fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_quality_violins(accepted, rejected, selected, output_dir, single_set_plots=True):
    score_keys = ["blur", "area", "occlusion", "completeness", "confidence", "score"]
    palette = {"selected": "#e76f51", "non-selected": "#264653"}

    selected_ids = {s.id for s in selected}
    non_sel = [o for o in accepted if o.id not in selected_ids]

    sep_idx = 3

    df_ns = _build_df(non_sel, score_keys, "non-selected")
    df_sel = _build_df(selected, score_keys, "selected")

    if single_set_plots:
        if len(df_ns) > 1:
            _plot_violin_set(
                df_ns, score_keys,
                "Non-Selected — Quality Score Distribution",
                output_dir / "violin_non_selected.png",
                sep_index=sep_idx,
            )
            _plot_violin_set_scaled(
                df_ns, score_keys,
                "Non-Selected — Quality Score Distribution (scaled)",
                output_dir / "violin_non_selected_scaled.png",
                sep_index=sep_idx,
            )
        if len(df_sel) > 1:
            _plot_violin_set(
                df_sel, score_keys,
                "Selected — Quality Score Distribution",
                output_dir / "violin_selected.png",
                sep_index=sep_idx,
            )
            _plot_violin_set_scaled(
                df_sel, score_keys,
                "Selected — Quality Score Distribution (scaled)",
                output_dir / "violin_selected_scaled.png",
                sep_index=sep_idx,
            )

    # ---- Combined: selected vs non-selected side-by-side per score ----
    combined = pd.concat([df_ns, df_sel], ignore_index=True)
    if len(combined) > 1:
        n = len(score_keys)
        fig, axes = _create_violin_figure(n, sep_idx)
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, score_keys):
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

    # ---- Scaled: zoomed-in per-subplot y-lim ----
    combined_scaled = pd.concat([df_ns, df_sel], ignore_index=True)
    if len(combined_scaled) > 1:
        n = len(score_keys)
        fig, axes = _create_violin_figure(n, sep_idx)
        if n == 1:
            axes = [axes]
        for ax, key in zip(axes, score_keys):
            data = []
            positions = []
            labels = []
            for i, g in enumerate(["non-selected", "selected"]):
                vals = combined_scaled.loc[combined_scaled["group"] == g, key].dropna().values
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

            all_key_vals = combined_scaled[key].dropna().values
            if len(all_key_vals) > 0:
                vmin, vmax = all_key_vals.min(), all_key_vals.max()
                margin = max((vmax - vmin) * 0.15, 0.005)
                ax.set_ylim(vmin - margin, vmax + margin)

            ax.set_title(key, fontsize=10)
            ax.set_ylabel("Score" if key == score_keys[0] else "")
        fig.suptitle("Selected vs Non-Selected — Quality Scores (scaled)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / "violin_selected_vs_non_selected_scaled.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir / 'violin_selected_vs_non_selected_scaled.png'}")

    # ---- Rejected vs accepted — normalized pre-filter metrics ----
    rejected_raw_keys = ["laplacian", "tenengrad", "area_ratio", "border_free", "hand_overlap_free", "completeness"]
    readable_labels = {
        "laplacian": "Laplacian", "tenengrad": "Tenengrad",
        "area_ratio": "Area Ratio", "border_free": "Border-Free Ratio",
        "hand_overlap_free": "Hand-Free Ratio", "completeness": "Completeness",
    }
    df_rej = _build_df(rejected, rejected_raw_keys, "rejected")
    df_acc = _build_df(accepted, rejected_raw_keys, "accepted")
    combined_raw = pd.concat([df_rej, df_acc], ignore_index=True)

    if len(combined_raw) > 1:
        for suffix, scaled in [("", False), ("_scaled", True)]:
            n_raw = len(rejected_raw_keys)
            fig, axes = plt.subplots(1, n_raw, figsize=(3 * n_raw + 1, 4.5))
            for ax, key in zip(axes, rejected_raw_keys):
                all_vals = combined_raw[key].dropna().values
                vmin_all, vmax_all = all_vals.min(), all_vals.max()
                for i, (g, color) in enumerate([("rejected", "#e9c46a"), ("accepted", "#2a9d8f")]):
                    vals = combined_raw.loc[combined_raw["group"] == g, key].dropna().values
                    if len(vals) > 1:
                        if vmax_all > vmin_all:
                            vals_norm = (vals - vmin_all) / (vmax_all - vmin_all)
                        else:
                            vals_norm = vals * 0.0
                        vp = ax.violinplot(vals_norm, positions=[i + 1], showmeans=False, showmedians=True)
                        vp["bodies"][0].set_facecolor(color)
                        vp["bodies"][0].set_alpha(0.6)
                        vp["cmedians"].set_color(color)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["rejected", "accepted"], fontsize=8)
                ax.set_title(readable_labels.get(key, key), fontsize=10)

                if scaled:
                    local_vals = combined_raw[key].dropna().values
                    if vmax_all > vmin_all:
                        local_norm = (local_vals - vmin_all) / (vmax_all - vmin_all)
                    else:
                        local_norm = local_vals * 0.0
                    vmin_z, vmax_z = local_norm.min(), local_norm.max()
                    margin = max((vmax_z - vmin_z) * 0.15, 0.005)
                    ax.set_ylim(vmin_z - margin, vmax_z + margin)
                else:
                    ax.set_ylim(-0.05, 1.05)

            title_text = "Rejected vs Accepted — Pre-Filter Metrics (normalised [0,1])"
            if scaled:
                title_text += " — scaled"
            fig.suptitle(title_text, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fname = f"violin_rejected_vs_accepted{suffix}.png"
            fig.savefig(output_dir / fname, dpi=150)
            plt.close(fig)
            print(f"  Saved {output_dir / fname}")


def _reduce_embeddings(embeddings, method, selected_idx=None, n_components=2, random_state=0, **kwargs):
    n = len(embeddings)
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components, random_state=random_state)
        coords = model.fit_transform(embeddings)
        extra = {"explained_variance_ratio": getattr(model, "explained_variance_ratio_", None)}
        return coords, extra
    elif method == "mds":
        from sklearn.manifold import MDS
        model = MDS(n_components=n_components, random_state=random_state, normalized_stress=False, **kwargs)
        coords = model.fit_transform(embeddings)
        extra = {"stress": getattr(model, "stress_", None)}
        return coords, extra
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = kwargs.get("perplexity", min(30, max(5, n - 1)))
        model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        coords = model.fit_transform(embeddings)
        return coords, {}
    elif method == "isomap":
        from sklearn.manifold import Isomap
        n_neighbors = kwargs.get("n_neighbors", min(15, max(5, n - 1)))
        model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        coords = model.fit_transform(embeddings)
        extra = {"reconstruction_error": getattr(model, "reconstruction_error", None)}
        return coords, extra
    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = kwargs.get("n_neighbors", min(15, max(5, n - 1)))
        model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
        coords = model.fit_transform(embeddings)
        extra = {"reconstruction_error": getattr(model, "reconstruction_error_", None)}
        return coords, extra
    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        if selected_idx is None:
            raise ValueError("LDA requires selected_idx")
        labels = np.zeros(len(embeddings), dtype=int)
        labels[list(selected_idx)] = 1
        n_classes = len(np.unique(labels))
        nc = min(n_components, n_classes - 1)
        model = LinearDiscriminantAnalysis(n_components=nc)
        coords = model.fit_transform(embeddings, labels)
        extra = {"explained_variance_ratio": getattr(model, "explained_variance_ratio_", None)}
        return coords, extra
    elif method == "umap":
        import umap
        model = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        coords = model.fit_transform(embeddings)
        return coords, {}
    else:
        raise ValueError(f"Unknown method: {method}")


def _draw_embedding_scatter_2d(path, coords, quality_scores, selected_idx, title,
                                cmap="viridis", vmin=None, vmax=None, var_ratio=None,
                                draw_nearest_lines=False):
    n = len(coords)
    selected_set = set(selected_idx)
    sel_coords = coords[list(selected_idx)]
    sel_qual = quality_scores[list(selected_idx)]
    non_sel_mask = np.ones(n, dtype=bool)
    non_sel_mask[list(selected_idx)] = False
    non_sel_coords = coords[non_sel_mask]
    non_sel_qual = quality_scores[non_sel_mask]

    if vmin is None:
        vmin = quality_scores.min()
    if vmax is None:
        vmax = quality_scores.max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sc = ax.scatter(
        non_sel_coords[:, 0], non_sel_coords[:, 1],
        c=non_sel_qual, cmap=cmap, s=20, alpha=0.5,
        vmin=vmin, vmax=vmax, label="non-selected",
    )

    cmap_obj = plt.get_cmap(cmap)
    norm_sel = (sel_qual - vmin) / (vmax - vmin) if vmax > vmin else sel_qual * 0
    sel_colors = cmap_obj(norm_sel)
    ax.scatter(
        sel_coords[:, 0], sel_coords[:, 1],
        c=sel_colors, s=100, marker="o", edgecolors="black", linewidths=1.5,
        label="selected", zorder=5,
    )

    for i, idx in enumerate(selected_idx):
        ax.annotate(
            str(i + 1), coords[idx],
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold", color="black",
        )

    if draw_nearest_lines:
        dist_to_sel = pairwise_distances(coords, sel_coords, metric="cosine")
        nearest_sel = dist_to_sel.argmin(axis=1)
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

    if var_ratio is not None and len(var_ratio) >= 2:
        ax.set_xlabel(f"Component 1 ({var_ratio[0]:.1%} variance)")
        ax.set_ylabel(f"Component 2 ({var_ratio[1]:.1%} variance)")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    ax.set_title(title)
    ax.legend(loc="best")

    if draw_nearest_lines:
        ax.text(
            0.98, 0.02,
            "Grey lines: each non-selected view → its nearest selected view (cosine similarity)",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="gray", style="italic",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _draw_embedding_scatter_3d(path, coords, quality_scores, selected_idx, title,
                                var_ratio=None):
    import plotly.graph_objects as go

    sel = list(selected_idx)
    selected_set = set(sel)
    non_sel = [i for i in range(len(coords)) if i not in selected_set]

    vmin = quality_scores.min()
    vmax = quality_scores.max()

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=coords[non_sel, 0], y=coords[non_sel, 1], z=coords[non_sel, 2],
        mode="markers",
        marker=dict(
            size=4, color=quality_scores[non_sel],
            colorscale="Viridis", cmin=vmin, cmax=vmax,
            showscale=True, colorbar=dict(title="Quality Score"),
            opacity=0.5,
        ),
        text=[f"idx={i}<br>quality={quality_scores[i]:.3f}" for i in non_sel],
        hoverinfo="text",
        name="non-selected",
    ))

    fig.add_trace(go.Scatter3d(
        x=coords[sel, 0], y=coords[sel, 1], z=coords[sel, 2],
        mode="markers+text",
        marker=dict(
            size=10, color=quality_scores[sel],
            colorscale="Viridis", cmin=vmin, cmax=vmax,
            line=dict(color="black", width=2),
            opacity=1,
        ),
        text=[str(i + 1) for i in range(len(sel))],
        textposition="top center",
        textfont=dict(size=12, color="black", family="Arial Black"),
        hovertext=[f"selected #{i+1}<br>quality={quality_scores[s]:.3f}" for i, s in enumerate(sel)],
        hoverinfo="text",
        name="selected",
    ))

    if var_ratio is not None and len(var_ratio) >= 3:
        ax_labels = [
            f"PC1 ({var_ratio[0]:.1%})",
            f"PC2 ({var_ratio[1]:.1%})",
            f"PC3 ({var_ratio[2]:.1%})",
        ]
    else:
        ax_labels = ["Component 1", "Component 2", "Component 3"]

    def _axis(title):
        return dict(title=title, showgrid=True, gridcolor="lightgray", zeroline=False)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=_axis(ax_labels[0]),
            yaxis=_axis(ax_labels[1]),
            zaxis=_axis(ax_labels[2]),
            aspectmode="cube",
        ),
        width=900, height=800,
        legend=dict(x=0.8, y=1),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.write_html(path)
    print(f"  Saved {path}")
    plt.close("all")


def plot_selection_process(accepted, selected, embeddings, selected_idx, quality_scores, output_dir):
    n = len(embeddings)
    if n < 2:
        return

    # ---- PCA original (jet, 0-1) ----
    coords_pca_2d, extra = _reduce_embeddings(embeddings, "pca", n_components=2)
    _draw_embedding_scatter_2d(
        output_dir / "selection_embedding.png",
        coords_pca_2d, quality_scores, selected_idx,
        "View Selection — Embedding Space (PCA)",
        cmap="jet", vmin=0, vmax=1,
        var_ratio=extra["explained_variance_ratio"],
        draw_nearest_lines=True,
    )

    # ---- PCA scaled (viridis, min-max) ----
    _draw_embedding_scatter_2d(
        output_dir / "selection_embedding_scaled.png",
        coords_pca_2d, quality_scores, selected_idx,
        "View Selection — Embedding Space (PCA, scaled)",
        cmap="viridis", vmin=None, vmax=None,
        var_ratio=extra["explained_variance_ratio"],
    )

    # ---- PCA 3D interactive ----
    coords_pca_3d, extra_3d = _reduce_embeddings(embeddings, "pca", n_components=3)
    _draw_embedding_scatter_3d(
        output_dir / "selection_embedding_3d.html",
        coords_pca_3d, quality_scores, selected_idx,
        "View Selection — Embedding Space (PCA 3D)",
        var_ratio=extra_3d["explained_variance_ratio"],
    )

    # ---- Other dimensionality reduction methods ----
    method_configs = [
        ("mds",    "MDS",    {}),
        ("tsne",   "t-SNE",  {}),
        ("umap",   "UMAP",   {}),
        ("isomap", "Isomap", {}),
        ("lle",    "LLE",    {}),
    ]
    if len(selected_idx) >= 2 and n - len(selected_idx) >= 1:
        method_configs.append(("lda", "LDA", {}))

    for method_key, method_label, kwargs in method_configs:
        try:
            coords_2d, extra_2d = _reduce_embeddings(
                embeddings, method_key, selected_idx=selected_idx,
                n_components=2, **kwargs,
            )
            _draw_embedding_scatter_2d(
                output_dir / f"embedding_{method_key}.png",
                coords_2d, quality_scores, selected_idx,
                f"View Selection — Embedding Space ({method_label})",
                cmap="viridis", vmin=None, vmax=None,
            )

            if n > 3:
                try:
                    coords_3d, _ = _reduce_embeddings(
                        embeddings, method_key, selected_idx=selected_idx,
                        n_components=3, **kwargs,
                    )
                    _draw_embedding_scatter_3d(
                        output_dir / f"embedding_{method_key}_3d.html",
                        coords_3d, quality_scores, selected_idx,
                        f"View Selection — Embedding Space ({method_label} 3D)",
                    )
                except Exception as e:
                    print(f"  Note: 3D {method_label} skipped ({e})")
        except Exception as e:
            print(f"  Skipping {method_label}: {e}")

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


def plot_all(accepted, rejected, selected, embeddings, selected_idx, quality_scores, output_dir, single_set_plots=True):
    print("Generating pipeline plots...")
    plot_quality_violins(accepted, rejected, selected, output_dir, single_set_plots=single_set_plots)
    plot_selection_process(accepted, selected, embeddings, selected_idx, quality_scores, output_dir)
    print("Done.")
