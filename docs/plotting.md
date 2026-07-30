# Plotting Module

The `plotting_process/` submodule generates diagnostic visualisations for pipeline results. It is organised into three sub-packages covering quality-score violin plots, embedding-space dimensionality reduction plots, and miscellaneous charts.

## Quick Start

```bash
# From inside the pipeline (--plot flag):
python run.py --data_root /path/to/bottle --num_views 10 --output_dir ./outputs --plot --debug

# Standalone after a previous run:
python -m plotting_process.wrapper --input_dir ./outputs --debug
```

## Output Structure

```
outputs/
└── plots/
    ├── pre-filter/
    │   ├── violin_rejected_vs_accepted.png
    │   ├── violin_rejected_vs_accepted_scaled.png
    │   └── rejection_reasons.png
    │
    └── selection/
        ├── violin_non_selected.png
        ├── violin_non_selected_scaled.png
        ├── violin_selected.png
        ├── violin_selected_scaled.png
        ├── violin_selected_vs_non_selected.png
        ├── violin_selected_vs_non_selected_scaled.png
        │
        ├── 2D_DR_plots/
        │   ├── selection_embedding.png            (PCA, jet, 0-1)
        │   ├── selection_embedding_scaled.png      (PCA, viridis, min-max)
        │   ├── embedding_mds.png
        │   ├── embedding_tsne.png                 (debug only)
        │   ├── embedding_umap.png                 (debug only)
        │   ├── embedding_isomap.png               (debug only)
        │   ├── embedding_lle.png                  (debug only)
        │   ├── embedding_lda.png                  (debug only)
        │
        └── 3D_DR_plots/
            ├── selection_embedding_3d.html        (PCA, interactive plotly)
            ├── embedding_mds_3d.html
            ├── embedding_tsne_3d.html             (debug only)
            ├── embedding_umap_3d.html             (debug only)
            ├── embedding_isomap_3d.html           (debug only)
            ├── embedding_lle_3d.html              (debug only)
            ├── embedding_lda_3d.html              (debug only)
```

## Plot Descriptions

### Quality-Score Violins (`selection/`)

Compare the quality-score distributions of selected vs non-selected (and non-selected vs selected) for each of the six quality components:

| Score | Definition |
|-------|-----------|
| `blur` | Normalised sharpness (Laplacian / 2× threshold) |
| `area` | Mask pixel area ratio, capped at 20 % |
| `occlusion` | 1 − hand-overlap ratio |
| `completeness` | Solidity / extent / convexity availability |
| `confidence` | Weakest-link score (min of all components) |
| `score` | Weighted arithmetic mean of the five components |

Each violin is a kernel-density estimate; the horizontal bar marks the mean. *Scaled* variants zoom the y-axis to the observed data range so that small differences become visible.

### Pre-Filter Violins (`pre-filter/`)

Compare raw metrics of rejected vs accepted observations for the six pre-filter criteria, normalised to [0, 1] across the combined pool. The *scaled* variant zooms per subplot.

| Metric | What it measures |
|--------|-----------------|
| Laplacian | Image sharpness (pixel-level variance) |
| Tenengrad | Gradient magnitude sharpness |
| Area Ratio | Mask size relative to image area |
| Border-Free Ratio | 1 − border-touching-pixel fraction |
| Hand-Free Ratio | 1 − hand-overlap fraction |
| Completeness | Combined solidity/extent/convexity score |

### Rejection Reasons (`pre-filter/`)

Horizontal bar chart counting how many observations were rejected by each filter module.

### 2D Embedding Scatter Plots (`selection/2D_DR_plots/`)

Each point is an observation projected into 2D via a dimensionality reduction technique. Non-selected points are colour-mapped by quality score; selected points are marked with numbered black-outlined circles. Grey connection lines (PCA only) trace each non-selected view to its nearest selected neighbour by cosine similarity.

| File | Method | Colormap | Color Range |
|------|--------|----------|-------------|
| `selection_embedding.png` | PCA | jet | [0, 1] |
| `selection_embedding_scaled.png` | PCA | viridis | [min, max] |
| `embedding_mds.png` | MDS | viridis | [min, max] |
| `embedding_*.png` | t-SNE / UMAP / Isomap / LLE / LDA | viridis | [min, max] |

### 3D Interactive Scatter Plots (`selection/3D_DR_plots/`)

Plotly HTML files with the same colour scheme as the 2D variants but in three dimensions. Hover over points to see index and quality score; selected points are labelled with their rank number.

## Debug Mode

By default only **PCA** and **MDS** plots are generated. Pass `--debug` (or set `cfg.debug = True`) to include:

- t-SNE
- UMAP
- Isomap
- LLE
- LDA (only when both selected and non-selected classes have ≥ 1 sample)

## Standalone Plotting

Re-generate plots from a previous pipeline run without re-running the pipeline:

```bash
python -m plotting_process.wrapper --input_dir /path/to/pipeline/outputs [--output_dir /path/to/plots] [--debug]
```

If `--output_dir` is omitted, the `plots/` folder is created inside `--input_dir`.

### Requirements for Standalone Mode

The pipeline must have saved these files (all are saved by default):

| File | Required for |
|------|-------------|
| `quality.csv` | All violin plots |
| `embeddings.npy` | Embedding scatter plots |
| `selected_indices.npy` | Embedding scatter plots |
| `report.json` | Observation ID mapping |
| `rejected.json` | Rejection reasons |
| `rejected_metrics.csv` | Pre-filter violin plots |

## Dimensionality Reduction Methods

| Method | Library | Properties |
|--------|---------|------------|
| **PCA** | sklearn.decomposition | Linear, orthogonal, maximises variance |
| **MDS** | sklearn.manifold | Metric, preserves pairwise distances |
| **t-SNE** | sklearn.manifold | Non-linear, preserves local neighbourhoods |
| **UMAP** | umap-learn | Non-linear, preserves topology |
| **Isomap** | sklearn.manifold | Non-linear, geodesic distances |
| **LLE** | sklearn.manifold | Non-linear, local linear patches |
| **LDA** | sklearn.discriminant_analysis | Supervised, maximises class separation (selected vs non-selected) |

## Submodule Architecture

```
plotting_process/
├── __init__.py
├── wrapper.py                   # plot_all() entry point + standalone CLI
├── misc_plot.py                 # rejection_reasons bar chart
│
├── embedding_plots/
│   ├── __init__.py              # debug-gated run_all()
│   ├── base.py                  # _reduce_embeddings(), draw_2d(), draw_3d()
│   ├── pca.py, mds.py, tsne.py, umap.py, isomap.py, lle.py, lda.py
│
└── quality_score_plots/
    ├── __init__.py
    └── violins.py               # all violin-plot logic
```
