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
    │   ├── pre_filter_raw_stats.png
    │   ├── pre_filter_soft_weights.png
    │   ├── rejection_reasons.png
    │   ├── data_set_overview/
    │   │   ├── raw_filter_<feature>_fixed.png      (coolwarm, goodness 0..1)
    │   │   ├── raw_filter_<feature>_relative.png   (viridis, data-relative goodness)
    │   │   ├── quality_score_<feature>_fixed.png
    │   │   └── quality_score_<feature>_relative.png
    │   └── bad_examples/
    │       ├── raw_filter_<feature>.png
    │       └── quality_score_<feature>.png
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

Compare the quality-score distributions of selected vs non-selected (and non-selected vs selected) for each of the quality components:

| Score | Definition |
|-------|-----------|
| `blur` | Normalised sharpness (Laplacian / 2× threshold) |
| `area` | Mask pixel area ratio, capped at 20 % |
| `occlusion` | 1 − hand-overlap ratio |
| `vincents_area` | Population-adapted mask area weight |
| `vincents_artefacts` | Population-adapted mask artifact weight |
| `vincents_motion_blur` | Population-adapted boundary blur weight |
| `completeness` | Solidity / extent / convexity availability |
| `confidence` | Weakest-link score (min of all components) |
| `score` | Weighted arithmetic mean of all components |

Each violin is a kernel-density estimate; the horizontal bar marks the mean. *Scaled* variants zoom the y-axis to the observed data range so that small differences become visible.

### Pre-Filter Violins (`pre-filter/`)

Compare raw metrics of rejected vs accepted observations for the pre-filter criteria, normalised to [0, 1] across the combined pool. The *scaled* variant zooms per subplot.

| Metric | What it measures |
|--------|-----------------|
| Laplacian | Image sharpness (pixel-level variance) |
| Tenengrad | Gradient magnitude sharpness |
| Area Ratio | Mask size relative to image area |
| Border-Free Ratio | 1 − border-touching-pixel fraction |
| Hand-Free Ratio | 1 − hand-overlap fraction |
| Completeness | Combined solidity/extent/convexity score |
| Mask Area Fraction | Vincent soft pre-filter raw stat (mask/canvas area) |
| Artifact Fraction | Vincent soft pre-filter raw stat (open⊕close / mask pixels) |
| Boundary Blur Variance | Vincent soft pre-filter raw stat (Laplacian variance in boundary band) |

### Pre-Filter Distribution Plots (`pre-filter/`)

Histograms for **every pre-filter element** for debugging:

- `pre_filter_raw_stats.png` — grid of histograms of each raw stat, accepted (teal) vs rejected (gold). Covers the classic filters (Laplacian, Tenengrad, area, border, edge, hand overlap, completeness) and the Vincent elements (mask pixel count, touches-border flag, area fraction, artifact fraction, boundary blur variance).
- `pre_filter_soft_weights.png` — histograms of the population-adapted soft weights (`vincents_area`, `vincents_artefacts`, `vincents_motion_blur`) over the accepted set.

### Feature Sample Distribution Plots (`pre-filter/`)

Per-feature diagnostics covering **every feature** (all pre-filter raw stats plus all quality component scores), generated by `plotting_process/feature_plots.py`. One dedicated image per feature, split across two folders with the prefixes `raw_filter_` (pre-filter raw stats) and `quality_score_` (quality components):

- `data_set_overview/` — **two variants per feature**, each with a distribution histogram on the left (rejected stacked by reason, accepted in teal, selected frames marked by black dashed vertical lines) and the feature value over the frame sequence on the right, selected frames highlighted in gold. The histogram x-axis is fixed to `[-0.05, 1.05]` for features whose values are bounded to `[0, 1]`.

  All sample colouring uses a **persistent meaning**: `1.0` is always *good* and `0.0` is always *bad*, regardless of the feature. For features where a high value is bad (border ratio, hand overlap, artifact fraction) the value is inverted so a good frame is always warm/bright. The two variants differ only in how that goodness value is mapped to a colour scale:

  - `*_fixed.png` — `coolwarm` with the colorbar pinned to the full `0..1` goodness range, so colours are directly comparable across every feature and dataset.
  - `*_relative.png` — `viridis` with the colorbar adjusted to this dataset's observed goodness `min..max`, which makes small relative differences easier to see.

- `bad_examples/` — the worst frames **that were actually filtered out** per feature (up to 5, sampled worst-first). Each has the mask overlaid (artifact pixels highlighted for `vincent_artifact_fraction`, the boundary band for `vincent_boundary_blur_variance`, border pixels for the border/edge features). Every thumbnail is framed by a `coolwarm` border proportional to its goodness (warm = good, cold = bad) and labelled with the frame id, feature value and rejection reason. If fewer than 5 frames were rejected, the remaining slots are drawn as placeholder tiles.

### Rejection Reasons (`pre-filter/`)

Horizontal bar chart counting how many observations were rejected by each filter module, sorted largest first and using descriptive labels. **Occlusion** (hand or other object covering the object) and **truncation** (object cut off at the frame edge) are always kept as two separate bars so the two failure modes never merge. The truncation bar aggregates both truncation detectors (`border` and `vincent_border_pixel`); all other raw reasons (`occlusion`, `small_object`, `blur`, `incomplete_shape`, ...) each keep their own bar.

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
├── pre_filter_plots.py          # per-element pre-filter histograms
├── feature_plots.py             # per-feature overview + bad-example plots
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
