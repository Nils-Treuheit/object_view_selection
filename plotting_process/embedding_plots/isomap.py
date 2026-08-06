from .base import render_embedding

def render(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d,
           cluster_labels=None, stem="embedding", space_label="Embedding Space"):
    render_embedding("isomap", "Isomap", False, embeddings, selected_idx, quality_scores,
                     output_dir_2d, output_dir_3d, cluster_labels=cluster_labels,
                     stem=stem, space_label=space_label)
