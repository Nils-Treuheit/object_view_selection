"""
Debug-only neighbourhood visualisations for the final selected views.

The point of these plots is to check whether the embedding space actually
groups *similar-looking* frames together: for every final selected candidate we
show its 5 nearest neighbours in embedding space (cosine distance), so a human
can eyeball whether the neighbours really look like the candidate.

Three images are produced (all debug-gated):

  selected_neighbors_knn.png     plain 5-NN in embedding space, one row per
                                 selected candidate (candidate + 5 neighbours).
  selected_neighbors_kmeans.png  the same, but the neighbours are taken from
                                 the candidate's k-means cluster first (k-means
                                 over the pool embeddings), only falling back to
                                 the overall nearest neighbours when the
                                 cluster has fewer than 5 members.
  selected_clusters_pca.png      PCA 2D scatter of the pool coloured by k-means
                                 cluster assignment, selected candidates marked
                                 with gold stars.

All three are diagnostics: nothing here changes the pipeline output.
"""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .feature_plots import _load_image

# one row per selected candidate: the candidate itself plus this many neighbours
NEIGHBORS_PER_CANDIDATE = 5

# thumbnail resolution for the grid cells
THUMB_SIZE = 96


def _thumbnail(obs):
    """Return a square RGB thumbnail for ``obs`` or None when unavailable."""
    if obs is None:
        return None
    image, _ = _load_image(obs)
    if image is None:
        return None
    return cv2.resize(image, (THUMB_SIZE, THUMB_SIZE))


def _neighbor_positions(embeddings, selected_idx, k, prefer_cluster=False, n_clusters=None):
    """For each selected position return the k nearest neighbour positions.

    Neighbours are the k closest positions by cosine distance, never including
    the candidate itself. When ``prefer_cluster`` is set, positions from the
    candidate's k-means cluster are taken first and the ranking is filled up
    with the overall nearest remaining positions.
    """
    n = len(embeddings)
    dist = pairwise_distances(embeddings, metric="cosine")

    labels = None
    if prefer_cluster:
        kc = max(2, min(n_clusters or len(selected_idx), n))
        labels = KMeans(n_clusters=kc, random_state=0, n_init=10).fit_predict(embeddings)

    rows = []
    for pos in selected_idx:
        order = np.argsort(dist[pos])
        order = [p for p in order if p != pos]

        if labels is not None:
            cluster = labels[pos]
            same_cluster = [p for p in order if labels[p] == cluster]
            order = same_cluster + [p for p in order if labels[p] != cluster]

        rows.append((pos, order[:k]))
    return rows


def _render_neighbor_grid(embeddings, selected_idx, pool_obs, out_path, k,
                          prefer_cluster, n_clusters, title):
    """One row per selected candidate: candidate + k neighbours as thumbnails."""
    rows = _neighbor_positions(
        embeddings, selected_idx, k,
        prefer_cluster=prefer_cluster, n_clusters=n_clusters,
    )

    n_rows = len(rows)
    if n_rows == 0:
        return
    n_cols = 1 + k

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    dist = pairwise_distances(embeddings, metric="cosine")

    for r, (sel_pos, neigh_pos) in enumerate(rows):
        sel_obs = pool_obs[sel_pos] if sel_pos < len(pool_obs) else None

        ax = axes[r, 0]
        ax.axis("off")
        thumb = _thumbnail(sel_obs)
        if thumb is None:
            ax.imshow(np.full((THUMB_SIZE, THUMB_SIZE, 3), 210, dtype=np.uint8))
            ax.set_title(f"Selected #{r + 1}\nid={getattr(sel_obs, 'id', '?')}",
                         fontsize=8, color="dimgray")
        else:
            ax.imshow(thumb)
            ax.add_patch(Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                fill=False, edgecolor="gold", linewidth=4,
            ))
            ax.set_title(f"Selected #{r + 1}\nid={getattr(sel_obs, 'id', '?')}",
                         fontsize=8, fontweight="bold")

        for c, npos in enumerate(neigh_pos):
            ax = axes[r, c + 1]
            ax.axis("off")
            n_obs = pool_obs[npos] if npos < len(pool_obs) else None
            thumb = _thumbnail(n_obs)
            if thumb is None:
                ax.imshow(np.full((THUMB_SIZE, THUMB_SIZE, 3), 240, dtype=np.uint8))
                ax.set_title(f"id={getattr(n_obs, 'id', '?')}", fontsize=8,
                             color="gray")
            else:
                ax.imshow(thumb)
                ax.set_title(
                    f"id={getattr(n_obs, 'id', '?')}\ncos={dist[sel_pos, npos]:.3f}",
                    fontsize=8,
                )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_kmeans_pca_scatter(embeddings, selected_idx, quality_scores, out_path, n_clusters):
    """PCA 2D scatter coloured by k-means cluster, selected marked as stars."""
    n = len(embeddings)
    kc = max(2, min(n_clusters or len(selected_idx), n))
    labels = KMeans(n_clusters=kc, random_state=0, n_init=10).fit_predict(embeddings)
    coords = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", kc)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap,
                         s=25, alpha=0.7, vmin=-0.5, vmax=kc - 0.5)

    sel = list(selected_idx)
    sel_coords = coords[sel]
    ax.scatter(sel_coords[:, 0], sel_coords[:, 1], marker="*", s=260,
               c="gold", edgecolors="black", linewidths=1.2, zorder=5,
               label="selected")
    for i, pos in enumerate(sel):
        ax.annotate(str(i + 1), coords[pos], xytext=(6, 6),
                    textcoords="offset points", fontsize=8,
                    fontweight="bold", color="black")

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"k-means clustering over the selection pool (k={kc})")
    ax.legend(loc="best")

    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(kc))
    cbar.ax.set_yticklabels([str(i) for i in range(kc)])
    cbar.set_label("k-means cluster")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_neighbor_analysis(embeddings, selected_idx, pool_obs, quality_scores,
                           output_dir, n_neighbors=NEIGHBORS_PER_CANDIDATE,
                           n_clusters=None):
    """All three debug neighbourhood visualisations (see module docstring).

    ``pool_obs`` is the list of observations aligned with ``embeddings`` (one
    per row). When a position has no observation the cell renders a placeholder.
    """
    n = len(embeddings)
    if n < 2 or len(selected_idx) == 0:
        return
    if pool_obs is None:
        pool_obs = []
    if n_clusters is None:
        n_clusters = min(len(selected_idx), n)

    output_dir = Path(output_dir)

    _render_neighbor_grid(
        embeddings, selected_idx, pool_obs, output_dir / "selected_neighbors_knn.png",
        k=n_neighbors, prefer_cluster=False, n_clusters=None,
        title=f"5 nearest neighbours per selected view (cosine distance, embedding space)",
    )

    _render_neighbor_grid(
        embeddings, selected_idx, pool_obs, output_dir / "selected_neighbors_kmeans.png",
        k=n_neighbors, prefer_cluster=True, n_clusters=n_clusters,
        title=f"Nearest neighbours inside each selected view's k-means cluster (k={n_clusters})",
    )

    _plot_kmeans_pca_scatter(
        embeddings, selected_idx, quality_scores, output_dir / "selected_clusters_pca.png",
        n_clusters=n_clusters,
    )
