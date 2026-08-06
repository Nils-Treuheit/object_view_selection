from . import pca, mds, tsne, umap, isomap, lle, lda
from .base import render_embedding, kmeans_cluster_labels

_METHODS = [
    (pca,  True),
    (mds,  True),
    (tsne, False),
    (umap, False),
    (isomap, False),
    (lle,  False),
    (lda,  False),
]

_DEFAULT_CLUSTERS = 10


def _run_methods(data, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
                 debug=False, cluster_labels=None, stem="embedding",
                 space_label="Embedding Space"):
    for mod, always in _METHODS:
        if not always and not debug:
            continue
        mod.render(data, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
                   cluster_labels=cluster_labels, stem=stem, space_label=space_label)


def run_all(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
            debug=False, cluster_labels=None, n_clusters=_DEFAULT_CLUSTERS,
            stem="embedding", space_label="Embedding Space"):
    """DR plots of the embedding space.

    ``cluster_labels`` are k-means clusters over the space's rows; when omitted
    they are computed here. They feed LDA as class labels (so LDA's 2D/3D
    reductions actually render) and colour the ``clusters_*`` variants.
    """
    if cluster_labels is None:
        cluster_labels = kmeans_cluster_labels(embeddings, n_clusters)
    _run_methods(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
                 debug=debug, cluster_labels=cluster_labels, stem=stem, space_label=space_label)


def run_criteria_dr(metrics_matrix, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
                    debug=False, n_clusters=_DEFAULT_CLUSTERS):
    """DR plots of the quality-criteria space.

    Rows of ``metrics_matrix`` must align with the selection pool (same order
    as ``embeddings`` / ``selected_idx``). Each row is the normalised metric
    vector of one observation.
    """
    run_all(metrics_matrix, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
            debug=debug, n_clusters=n_clusters, stem="criteria", space_label="Quality Criteria")
