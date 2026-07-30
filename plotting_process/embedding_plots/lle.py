from .base import render_embedding

def render(embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d):
    render_embedding("lle", "LLE", False, embeddings, selected_idx, quality_scores, output_dir_2d, output_dir_3d)
