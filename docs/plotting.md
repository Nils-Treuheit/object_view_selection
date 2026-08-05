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
├── bad_examples/                # Per-stage example frames (sibling of plots/)
│   ├── pre-filter_stage/
│   │   ├── <feature>_filtered.png          (frames rejected for that feature's reason)
│   │   └── lower_<feature>_quality.png     (prob-sampled lowest-quality accepted frames)
│   └── selection_stage/
│       └── lower_<feature>_quality.png     (prob-sampled lowest-quality accepted frames)
│
└── plots/
    ├── pre-filter/
    │   ├── violin_rejected_vs_accepted.png
    │   ├── violin_rejected_vs_accepted_scaled.png
    │   ├── pre_filter_raw_stats.png
    │   ├── pre_filter_soft_weights.png
    │   ├── rejection_reasons.png
    │   └── data_set_overview/
    │       ├── <feature>_filter_fixed.png      (bounded features only, coolwarm, 0..1)
    │       └── <feature>_filter_relative.png   (viridis, data-relative)
    │
    └── selection/
        ├── violin_non_selected.png
        ├── violin_non_selected_scaled.png
        ├── violin_selected.png
        ├── violin_selected_scaled.png
        ├── violin_selected_vs_non_selected.png
        ├── violin_selected_vs_non_selected_scaled.png
        ├── data_set_overview/
        │   ├── quality_score_<feature>_fixed.png
        │   └── quality_score_<feature>_relative.png
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
        ├── 3D_DR_plots/
        │   ├── selection_embedding_3d.html        (PCA, interactive plotly)
        │   ├── embedding_mds_3d.html
        │   ├── embedding_tsne_3d.html             (debug only)
        │   ├── embedding_umap_3d.html             (debug only)
        │   ├── embedding_isomap_3d.html           (debug only)
        │   ├── embedding_lle_3d.html              (debug only)
        │   ├── embedding_lda_3d.html              (debug only)
        │
        └── debug (--debug only):
            ├── selected_neighbors_knn.png         (5 nearest neighbours per selected view)
            ├── selected_neighbors_kmeans.png      (5 neighbours from the view's k-means cluster)
            └── selected_clusters_pca.png          (PCA scatter coloured by k-means cluster)
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

### Feature Sample Distribution Plots (`pre-filter/` + `selection/`)

Per-feature diagnostics covering **every feature** (all pre-filter raw stats plus all quality component scores), generated by `plotting_process/feature_plots.py`. One dedicated image per feature, split across two folders: the raw pre-filter stats land in `plots/pre-filter/data_set_overview/` under the `<feature>_filter_{fixed,relative}.png` names (e.g. `area_ratio_filter_fixed.png`, `laplacian_filter_relative.png`), and the quality component scores land in `plots/selection/data_set_overview/` under `quality_score_<feature>_{fixed,relative}.png`:

- `data_set_overview/` — **one or two variants per feature**, each with a distribution histogram on the left (rejected stacked by reason, accepted in teal, selected frames marked by black dashed vertical lines) and the feature value over the frame sequence on the right, selected frames highlighted in gold. Histogram bars are **centred on their bin values** — the bar for value `0.0` is centred at `0.0`, not at half a bin-width above it. For features whose values are bounded to `[0, 1]` the bars are pinned to a **fixed `0.0..1.0` grid** (centres at `0.0, 0.025, ..., 1.0`, half a bar-width overhanging each end), and the histogram x-axis is fixed to `[-0.05, 1.05]`. Unbounded stats keep data-driven bins. The overview titles are `<Label> (statistical pre-filter)` for the raw stats and `<Label> Score` for the quality scores.

  Reported values use a persistent meaning: the lower-is-better raw stats (border ratio, edge ratio, hand overlap, artifact fraction) are reported **inverted as `1 - value`** — the "free" share — so a higher reported value is always better, and the plots are titled just `Border-Free Ratio`, `Edge-Free Ratio`, `Hand-Free Ratio`, `Artifact-Free Fraction`.

  All sample colouring uses a **persistent meaning**: warm/bright is always *good* and cold/dark is always *bad*, regardless of the feature. The variants differ only in the colour scale:

  - `*_fixed.png` — `coolwarm` with the colorbar pinned to `0..1`, generated **only for features whose values are naturally bounded to `[0, 1]`** (all quality scores plus the ratio stats). The colour is the **absolute reported value**, so a dot's colour matches the colourbar tick labels exactly — a quality score of `0.99` always renders warm, never cold. Unbounded counting stats (Laplacian, Tenengrad, boundary-blur variance) have no fixed `0..1` meaning and are **only** reported relative.
  - `*_relative.png` — `viridis` with the colorbar adjusted to **this dataset's observed value range** rounded outward to the second decimal place (min rounded down, max rounded up), with ticks at `[0, 0.5, 1]` positions on the goodness scale labelled in the feature's **reported units at most 3 decimal places, written out in full** (e.g. Laplacian `6.48`…`50.94`, Border-Free Ratio `0.94`…`1`), so the relative plot reads as "relative to this dataset" and small differences within a narrow band become visible.

  Titles and colourbars carry no `(fixed 0..1)` / `(relative …)` suffix — the ticks with numbers are the colourbar's only labelling.

- `bad_examples/` — lives at the **top level of the output directory** (a sibling of `plots/`), split by pipeline stage:

  - `pre-filter_stage/<feature>_filtered.png` — up to 5 frames **actually rejected for that feature's own reason** (e.g. `area_ratio_filtered.png` shows only frames rejected as "small object (low mask area)"; `laplacian_filtered.png` and `tenengrad_filtered.png` show only blur-rejected frames; the border/edge stats cover the truncation detectors `border` and `vincent_border_pixel`). The absolute worst frame is always shown; the remaining slots are filled worst-first with frames that are **visually distinct** from the ones already shown (thumbnail-level difference), so a run of near-identical consecutive video frames never fills the whole row. If fewer than 5 frames were rejected for that reason, the remaining slots are placeholder tiles.
  - `pre-filter_stage/lower_<feature>_quality.png` — produced **only when a feature's reason never fired**: the lowest-quality **accepted** frames per that stat, **probability-sampled** (worst = highest likelihood). Vincent soft stats that can never hard-reject always take this form.
  - `selection_stage/lower_<feature>_quality.png` — for every quality score: the lowest-quality accepted frames, **probability-sampled** the same way.

  Each has the mask overlaid (artifact pixels highlighted for `vincent_artifact_fraction`, the boundary band for `vincent_boundary_blur_variance`, border pixels for the border/edge features). Every thumbnail is framed by a **`viridis` border coloured by the relative score** over the min/max of all samples — the same scale as the `*_relative.png` overview colourbar — and labelled with a status line plus `#<frame id> | QS: <reported value>` on the second line: filtered frames show `rejected - <reason>`, pre-filter lower-quality frames show `accepted - <feature label>`, and selection-stage frames show `accepted but not selected`. Titles are `Filtered-out examples: <label>` and `Lowest-quality accepted frames: <label>`.

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

### Embedding Neighbour Diagnostics (`selection/`)

`--debug` also enables three diagnostics (in `plotting_process/neighbor_plots.py`) that check whether the embedding space actually groups *similar-looking* frames together. For every final selected candidate, `NEIGHBORS_PER_CANDIDATE = 5` neighbours are shown:

- `selected_neighbors_knn.png` — one row per selected candidate: the candidate plus its **5 nearest neighbours by cosine distance** in embedding space. Each cell shows the frame id and, for neighbours, the cosine distance to the candidate; if the embeddings are meaningful, the neighbours should look like the candidate.
- `selected_neighbors_kmeans.png` — the same grid, but neighbours are taken **from the candidate's k-means cluster** first (k-means is fit over the pool embeddings with `k` = number of selected views), only falling back to the overall nearest neighbours when the cluster has fewer than 5 members. This highlights whether k-means cluster membership matches visual similarity.
- `selected_clusters_pca.png` — a PCA 2D scatter of the whole selection pool coloured by k-means cluster assignment, with the final selected candidates marked as numbered gold stars.

These are pure diagnostics: they never change the pipeline output.

## Standalone Plotting

Re-generate plots from a previous pipeline run without re-running the pipeline:

```bash
python -m plotting_process.wrapper --input_dir /path/to/pipeline/outputs [--output_dir /path/to/plots] [--debug]
```

If `--output_dir` is omitted, the `plots/` folder (and the top-level `bad_examples/` folder) is created inside `--input_dir`. The pipeline-result files (`report.json`, `rejected.json`, `quality.csv`, …) are always read from `--input_dir`.

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
├── neighbor_plots.py            # debug k-means / k-NN neighbour diagnostics
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
