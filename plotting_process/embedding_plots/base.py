import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


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
        model = MDS(n_components=n_components, random_state=random_state, normalized_stress=False, n_init=1, **kwargs)
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


def draw_embedding_scatter_2d(
    path, coords, quality_scores, selected_idx, title,
    cmap="viridis", vmin=None, vmax=None, var_ratio=None,
    draw_nearest_lines=False,
):
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


def draw_embedding_scatter_3d(path, coords, quality_scores, selected_idx, title, var_ratio=None):
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


def render_embedding(
    method_key, method_label, has_original,
    embeddings, selected_idx, quality_scores,
    output_dir_2d, output_dir_3d,
):
    n = len(embeddings)
    if n < 2:
        return

    if method_key == "pca":
        coords_2d, extra = _reduce_embeddings(embeddings, "pca", n_components=2)
        var_ratio = extra["explained_variance_ratio"]

        if has_original:
            draw_embedding_scatter_2d(
                output_dir_2d / "selection_embedding.png",
                coords_2d, quality_scores, selected_idx,
                "View Selection — Embedding Space (PCA)",
                cmap="jet", vmin=0, vmax=1,
                var_ratio=var_ratio, draw_nearest_lines=True,
            )

            draw_embedding_scatter_2d(
                output_dir_2d / "selection_embedding_scaled.png",
                coords_2d, quality_scores, selected_idx,
                "View Selection — Embedding Space (PCA, scaled)",
                cmap="viridis", vmin=None, vmax=None,
                var_ratio=var_ratio,
            )

        try:
            coords_3d, extra_3d = _reduce_embeddings(embeddings, "pca", n_components=3)
            draw_embedding_scatter_3d(
                output_dir_3d / "selection_embedding_3d.html",
                coords_3d, quality_scores, selected_idx,
                "View Selection — Embedding Space (PCA 3D)",
                var_ratio=extra_3d["explained_variance_ratio"],
            )
        except Exception as e:
            print(f"  Note: 3D {method_label} skipped ({e})")

        extra = _reduce_embeddings(embeddings, "pca", n_components=2)[1]
        return coords_2d, extra["explained_variance_ratio"]

    try:
        coords_2d, extra = _reduce_embeddings(
            embeddings, method_key, selected_idx=selected_idx, n_components=2,
        )
    except Exception as e:
        print(f"  Skipping {method_label}: {e}")
        return

    if coords_2d.shape[1] < 2:
        print(f"  Skipping 2D {method_label}: only {coords_2d.shape[1]} component(s) available")
        return

    stem = f"embedding_{method_key}"
    draw_embedding_scatter_2d(
        output_dir_2d / f"{stem}.png",
        coords_2d, quality_scores, selected_idx,
        f"View Selection — Embedding Space ({method_label})",
        cmap="viridis", vmin=None, vmax=None,
    )

    if n > 3:
        try:
            coords_3d, _ = _reduce_embeddings(
                embeddings, method_key, selected_idx=selected_idx, n_components=3,
            )
            if coords_3d.shape[1] < 3:
                print(f"  Skipping 3D {method_label}: only {coords_3d.shape[1]} component(s) available")
            else:
                draw_embedding_scatter_3d(
                    output_dir_3d / f"{stem}_3d.html",
                    coords_3d, quality_scores, selected_idx,
                    f"View Selection — Embedding Space ({method_label} 3D)",
                )
        except Exception as e:
            print(f"  Note: 3D {method_label} skipped ({e})")
