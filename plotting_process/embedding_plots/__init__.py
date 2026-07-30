from . import pca, mds, tsne, umap, isomap, lle, lda
from .base import render_embedding

_METHODS = [
    (pca,  True),
    (mds,  True),
    (tsne, False),
    (umap, False),
    (isomap, False),
    (lle,  False),
    (lda,  False),
]


def run_all(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d, debug=False):
    for mod, always in _METHODS:
        if not always and not debug:
            continue
        mod.render(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d)
