import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


def kmeans_cluster_labels(embeddings, n_clusters):
    """k-means cluster assignment over the rows of ``embeddings``.

    ``n_clusters`` is clamped to the valid range [2, n] so the result is
    always a usable set of class labels for LDA / cluster-coloured plots.
    """
    from sklearn.cluster import KMeans

    n = len(embeddings)
    k = max(2, min(n_clusters or 2, n))
    return KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embeddings)


def _reduce_embeddings(embeddings, method, selected_idx=None, n_components=2, random_state=0, labels=None, **kwargs):
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
        # Class labels: when provided (k-means clusters over the embedding
        # space) LDA gets n_classes - 1 components, so 2D/3D work for
        # k >= 3. Without labels it falls back to selected vs. non-selected,
        # which is capped at a single component.
        if labels is None:
            if selected_idx is None:
                raise ValueError("LDA requires selected_idx")
            labels = np.zeros(len(embeddings), dtype=int)
            labels[list(selected_idx)] = 1
        labels = np.asarray(labels)
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


def draw_cluster_scatter_2d(path, coords, cluster_labels, selected_idx, title, var_ratio=None):
    """2D scatter coloured by discrete cluster labels, selected marked as stars."""
    n = len(coords)
    labels = np.asarray(cluster_labels)
    k = len(np.unique(labels))
    cmap = plt.get_cmap("tab20", max(k, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=25, alpha=0.7,
        vmin=-0.5, vmax=max(k, 1) - 0.5,
    )

    sel = list(selected_idx)
    sel_coords = coords[sel]
    ax.scatter(
        sel_coords[:, 0], sel_coords[:, 1], marker="*", s=260,
        c="gold", edgecolors="black", linewidths=1.2, zorder=5,
        label="selected",
    )
    for i, pos in enumerate(sel):
        ax.annotate(str(i + 1), coords[pos], xytext=(6, 6),
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color="black")

    if var_ratio is not None and len(var_ratio) >= 2:
        ax.set_xlabel(f"Component 1 ({var_ratio[0]:.1%} variance)")
        ax.set_ylabel(f"Component 2 ({var_ratio[1]:.1%} variance)")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    ax.set_title(title)
    ax.legend(loc="best")

    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(k))
    cbar.ax.set_yticklabels([str(i) for i in range(k)])
    cbar.set_label("k-means cluster")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def draw_cluster_scatter_3d(path, coords, cluster_labels, selected_idx, title, var_ratio=None):
    """3D scatter coloured by discrete cluster labels (plotly)."""
    import plotly.graph_objects as go
    import matplotlib

    labels = np.asarray(cluster_labels)
    k = len(np.unique(labels))
    cmap = matplotlib.colormaps["tab20"].resampled(max(k, 1))
    colors = [matplotlib.colors.to_hex(cmap(labels[i])) for i in range(len(labels))]

    sel = list(selected_idx)
    selected_set = set(sel)
    non_sel = [i for i in range(len(coords)) if i not in selected_set]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=coords[non_sel, 0], y=coords[non_sel, 1], z=coords[non_sel, 2],
        mode="markers",
        marker=dict(
            size=4, color=[colors[i] for i in non_sel],
            opacity=0.6,
        ),
        text=[f"idx={i}<br>cluster={labels[i]}" for i in non_sel],
        hoverinfo="text",
        name="non-selected",
    ))

    fig.add_trace(go.Scatter3d(
        x=coords[sel, 0], y=coords[sel, 1], z=coords[sel, 2],
        mode="markers+text",
        marker=dict(
            size=10, color="gold",
            line=dict(color="black", width=2),
            opacity=1,
        ),
        text=[str(i + 1) for i in range(len(sel))],
        textposition="top center",
        textfont=dict(size=12, color="black", family="Arial Black"),
        hovertext=[f"selected #{i+1}<br>cluster={labels[s]}" for i, s in enumerate(sel)],
        hoverinfo="text",
        name="selected",
    ))

    if var_ratio is not None and len(var_ratio) >= 3:
        ax_labels = [
            f"Component 1 ({var_ratio[0]:.1%})",
            f"Component 2 ({var_ratio[1]:.1%})",
            f"Component 3 ({var_ratio[2]:.1%})",
        ]
    else:
        ax_labels = ["Component 1", "Component 2", "Component 3"]

    def _axis(t):
        return dict(title=t, showgrid=True, gridcolor="lightgray", zeroline=False)

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
    data, selected_idx, quality_scores,
    output_dir_2d, output_dir_3d,
    cluster_labels=None,
    stem="embedding",
    space_label="Embedding Space",
):
    n = len(data)
    if n < 2:
        return

    try:
        coords_2d, extra = _reduce_embeddings(
            data, method_key, selected_idx=selected_idx,
            n_components=2, labels=cluster_labels,
        )
    except Exception as e:
        print(f"  Skipping {method_label}: {e}")
        return

    if coords_2d.shape[1] < 2:
        print(f"  Skipping 2D {method_label}: only {coords_2d.shape[1]} component(s) available")
        return

    var_ratio = extra.get("explained_variance_ratio") if extra else None

    if method_key == "pca" and has_original:
        draw_embedding_scatter_2d(
            output_dir_2d / f"selection_{stem}.png",
            coords_2d, quality_scores, selected_idx,
            f"View Selection — {space_label} (PCA)",
            cmap="jet", vmin=0, vmax=1,
            var_ratio=var_ratio, draw_nearest_lines=True,
        )

        draw_embedding_scatter_2d(
            output_dir_2d / f"selection_{stem}_scaled.png",
            coords_2d, quality_scores, selected_idx,
            f"View Selection — {space_label} (PCA, scaled)",
            cmap="viridis", vmin=None, vmax=None,
            var_ratio=var_ratio,
        )

    draw_embedding_scatter_2d(
        output_dir_2d / f"{stem}_{method_key}.png",
        coords_2d, quality_scores, selected_idx,
        f"View Selection — {space_label} ({method_label})",
        cmap="viridis", vmin=None, vmax=None,
        var_ratio=var_ratio,
    )

    if cluster_labels is not None:
        draw_cluster_scatter_2d(
            output_dir_2d / f"clusters_{stem}_{method_key}.png",
            coords_2d, cluster_labels, selected_idx,
            f"Clusters — {space_label} ({method_label})",
            var_ratio=var_ratio,
        )

    if n > 3:
        try:
            coords_3d, extra_3d = _reduce_embeddings(
                data, method_key, selected_idx=selected_idx,
                n_components=3, labels=cluster_labels,
            )
            if coords_3d.shape[1] < 3:
                print(f"  Skipping 3D {method_label}: only {coords_3d.shape[1]} component(s) available")
            else:
                var_ratio_3d = extra_3d.get("explained_variance_ratio") if extra_3d else None
                if method_key == "pca" and has_original:
                    draw_embedding_scatter_3d(
                        output_dir_3d / f"selection_{stem}_3d.html",
                        coords_3d, quality_scores, selected_idx,
                        f"View Selection — {space_label} (PCA 3D)",
                        var_ratio=var_ratio_3d,
                    )
                draw_embedding_scatter_3d(
                    output_dir_3d / f"{stem}_{method_key}_3d.html",
                    coords_3d, quality_scores, selected_idx,
                    f"View Selection — {space_label} ({method_label} 3D)",
                    var_ratio=var_ratio_3d,
                )
                if cluster_labels is not None:
                    draw_cluster_scatter_3d(
                        output_dir_3d / f"clusters_{stem}_{method_key}_3d.html",
                        coords_3d, cluster_labels, selected_idx,
                        f"Clusters — {space_label} ({method_label} 3D)",
                        var_ratio=var_ratio_3d,
                    )
        except Exception as e:
            print(f"  Note: 3D {method_label} skipped ({e})")

    return coords_2d, var_ratio
