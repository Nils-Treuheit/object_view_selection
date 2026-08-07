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
        ├── embedding_space/                    (DR of the embedding space)
        │   ├── 2D_DR_plots/
        │   │   ├── selection_embedding.png      (PCA, jet, 0-1)
        │   │   ├── selection_embedding_scaled.png  (PCA, viridis, min-max)
        │   │   ├── embedding_mds.png
        │   │   ├── embedding_tsne.png          (debug only)
        │   │   ├── embedding_umap.png          (debug only)
        │   │   ├── embedding_isomap.png        (debug only)
        │   │   ├── embedding_lle.png           (debug only)
        │   │   ├── embedding_lda.png           (debug only — now renders via k-means cluster pseudo-labels)
        │   │   └── clusters_embedding_<method>.png  (same coords, coloured by k-means cluster)
        │   └── 3D_DR_plots/
        │       ├── selection_embedding_3d.html (PCA, interactive plotly)
        │       ├── embedding_mds_3d.html
        │       ├── embedding_tsne_3d.html      (debug only)
        │       ├── embedding_umap_3d.html      (debug only)
        │       ├── embedding_isomap_3d.html    (debug only)
        │       ├── embedding_lle_3d.html       (debug only)
        │       ├── embedding_lda_3d.html       (debug only)
        │       └── clusters_embedding_<method>_3d.html
        │
        ├── quality_criteria/                   (DR of the normalised metric space)
        │   └── DR_plots/
        │       ├── 2D_DR_plots/
        │       │   ├── selection_criteria.png     (PCA, jet, 0-1)
        │       │   ├── selection_criteria_scaled.png
        │       │   ├── criteria_mds.png
        │       │   ├── criteria_tsne.png          (debug only)
        │       │   ├── criteria_umap.png          (debug only)
        │       │   ├── criteria_isomap.png        (debug only)
        │       │   ├── criteria_lle.png           (debug only)
        │       │   ├── criteria_lda.png           (debug only)
        │       │   └── clusters_criteria_<method>.png
        │       └── 3D_DR_plots/
        │           ├── selection_criteria_3d.html
        │           ├── criteria_mds_3d.html
        │           ├── criteria_tsne_3d.html      (debug only)
        │           ├── criteria_umap_3d.html      (debug only)
        │           ├── criteria_isomap_3d.html    (debug only)
        │           ├── criteria_lle_3d.html       (debug only)
        │           ├── criteria_lda_3d.html       (debug only)
        │           └── clusters_criteria_<method>_3d.html
        │
        └── debug (--debug only):
            ├── selected_neighbors_knn.png         (5 nearest neighbours per selected view)
            ├── selected_neighbors_kmeans.png      (5 neighbours from the view's k-means cluster)
            ├── selected_clusters_pca.png          (PCA scatter coloured by k-means cluster)
            └── embedded_samples/samples_<NN>.png  (original, mask, 224×224 cut-out on contrast bg, + original mask)
```

## Plot Descriptions

### Quality-Score Violins (`selection/`)

Compare the quality-score distributions of selected vs non-selected (and non-selected vs selected) for each of the quality components:

| Score | Definition |
|-------|-----------|
| `blur` | Boundary-band sharpness (Laplacian variance / global anchor) |
| `area` | Mask pixel area ratio, capped at 20 % |
| `vincents_artefacts` | Mask artifact fraction, anchored by `artifacts_max_fraction` |
| `centerness` | Mask centredness in the frame |
| `confidence` | Weakest-link score (min of all components) |
| `score` | Weighted arithmetic mean of all components |

Each violin is a kernel-density estimate; the horizontal bar marks the mean. *Scaled* variants zoom the y-axis to the observed data range so that small differences become visible.

### Pre-Filter Violins (`pre-filter/`)

Compare raw metrics of rejected vs accepted observations for the pre-filter criteria, normalised to [0, 1] across the combined pool. The *scaled* variant zooms per subplot.

| Metric | What it measures |
|--------|-----------------|
| Boundary Laplacian Variance | Boundary-band sharpness (default blur_laplacian pre-filter stat) |
| Boundary Tenengrad | Boundary-band gradient sharpness (default blur_tenengrad pre-filter stat) |
| Artifact Fraction | Mask artifact stat (default vincents_artefacts pre-filter) |
| Boundary Blur Variance | Vincent motion-blur soft stat (Laplacian variance in boundary band) |
| Mask Area Fraction | Vincent soft stat (mask/canvas area) |
| Touches Border (hard) | Binary: mask touches the image frame |
| Mask Pixel Count | Number of mask foreground pixels |
| Area Ratio / Border Ratio / Edge Ratio / Hand Overlap / Completeness | Legacy filter stats (kept for diagnostics) |

### Pre-Filter Distribution Plots (`pre-filter/`)

Histograms for **every pre-filter element** for debugging:

- `pre_filter_raw_stats.png` — grid of histograms of each raw stat, accepted (teal) vs rejected (gold). Covers the default pre-filter stats (boundary Laplacian variance, boundary Tenengrad, artifact fraction, boundary blur variance, mask area fraction, touches-border flag, mask pixel count) and the legacy stats (area ratio, border ratio, edge ratio, hand overlap, completeness).
- `pre_filter_soft_weights.png` — histograms of the population-adjusted scores over the accepted set: the two remaining soft weights (`vincents_area`, `vincents_motion_blur`) and the artifact score (`vincents_artefacts`).

### Feature Sample Distribution Plots (`pre-filter/` + `selection/`)

Per-feature diagnostics covering **every feature** (all pre-filter raw stats plus all quality component scores), generated by `plotting_process/feature_plots.py`. One dedicated image per feature, split across two folders: the raw pre-filter stats land in `plots/pre-filter/data_set_overview/` under the `<feature>_filter_{fixed,relative}.png` names (e.g. `area_ratio_filter_fixed.png`, `laplacian_filter_relative.png`), and the quality component scores land in `plots/selection/data_set_overview/` under `quality_score_<feature>_{fixed,relative}.png`:

- `data_set_overview/` — **one or two variants per feature**, each with a distribution histogram on the left (rejected stacked by reason, accepted in teal, selected frames marked by black dashed vertical lines) and the feature value over the frame sequence on the right, selected frames highlighted in gold. Histogram bars are **centred on their bin values** — the bar for value `0.0` is centred at `0.0`, not at half a bin-width above it. For features whose values are bounded to `[0, 1]` the bars are pinned to a **fixed `0.0..1.0` grid** (centres at `0.0, 0.025, ..., 1.0`, half a bar-width overhanging each end), and the histogram x-axis is fixed to `[-0.05, 1.05]`. Unbounded stats keep data-driven bins. The overview titles are `<Label> (statistical pre-filter)` for the raw stats and `<Label> Score` for the quality scores.

  Reported values use a persistent meaning: the lower-is-better raw stats (border ratio, edge ratio, hand overlap, artifact fraction) are reported **inverted as `1 - value`** — the "free" share — so a higher reported value is always better, and the plots are titled just `Border-Free Ratio`, `Edge-Free Ratio`, `Hand-Free Ratio`, `Artifact-Free Fraction`.

  All sample colouring uses a **persistent meaning**: warm/bright is always *good* and cold/dark is always *bad*, regardless of the feature. The variants differ only in the colour scale:

  - `*_fixed.png` — `coolwarm` with the colorbar pinned to `0..1`, generated **only for features whose values are naturally bounded to `[0, 1]`** (all quality scores plus the ratio stats). The colour is the **absolute reported value**, so a dot's colour matches the colourbar tick labels exactly — a quality score of `0.99` always renders warm, never cold. Unbounded counting stats (Laplacian, Tenengrad, boundary-blur variance) have no fixed `0..1` meaning and are **only** reported relative.
  - `*_relative.png` — `viridis` with the colorbar adjusted to **this dataset's observed value range** rounded outward to the second decimal place (min rounded down, max rounded up), with ticks at `[0, 0.5, 1]` positions on the goodness scale labelled in the feature's **reported units at most 3 decimal places, written out in full** (e.g. Laplacian `6.48`…`50.94`, Border-Free Ratio `0.94`…`1`), so the relative plot reads as "relative to this dataset" and small differences within a narrow band become visible.

  Titles and colourbars carry no `(fixed 0..1)` / `(relative …)` suffix — the ticks with numbers are the colourbar's only labelling.

- `bad_examples/` — lives at the **top level of the output directory** (a sibling of `plots/`), split by pipeline stage:

  - `pre-filter_stage/<feature>_filtered.png` — up to 5 frames **actually rejected for that feature's own reason(s)** (e.g. `area_ratio_filtered.png` shows only frames rejected as "small object (low mask area)"; `laplacian_filtered.png` shows only frames rejected for `blur_laplacian_threshold` / `blur_laplacian_outlier`; `vincent_artifact_fraction_filtered.png` shows only `vincents_artefacts_threshold` / `vincents_artefacts_outlier`; the border/edge stats cover the truncation detectors `border` and `vincent_border_pixel`). The absolute worst frame is always shown; the remaining slots are filled worst-first with frames that are **visually distinct** from the ones already shown (thumbnail-level difference), so a run of near-identical consecutive video frames never fills the whole row. If fewer than 5 frames were rejected for that reason, the remaining slots are placeholder tiles.
  - `pre-filter_stage/lower_<feature>_quality.png` — produced **only when a feature's reason never fired**: the lowest-quality **accepted** frames per that stat, **probability-sampled** (worst = highest likelihood). Vincent soft stats that can never hard-reject always take this form.
  - `selection_stage/lower_<feature>_quality.png` — for every quality score: the lowest-quality accepted frames, **probability-sampled** the same way.

  Each has the mask overlaid (artifact pixels highlighted for `vincent_artifact_fraction`, the boundary band for `vincent_boundary_blur_variance`, border pixels for the border/edge features). Every thumbnail is framed by a **`viridis` border coloured by the relative score** over the min/max of all samples — the same scale as the `*_relative.png` overview colourbar — and labelled with a status line plus `#<frame id> | QS: <reported value>` on the second line: filtered frames show `rejected - <reason>`, pre-filter lower-quality frames show `accepted - <feature label>`, and selection-stage frames show `accepted but not selected`. Titles are `Filtered-out examples: <label>` and `Lowest-quality accepted frames: <label>`.

### Rejection Reasons (`pre-filter/`)

Horizontal bar chart counting how many observations were rejected by each filter module, sorted largest first and using descriptive labels. **Occlusion** (hand or other object covering the object) and **truncation** (object cut off at the frame edge) are always kept as two separate bars so the two failure modes never merge. The truncation bar aggregates both truncation detectors (`border` and `vincent_border_pixel`); all other raw reasons (`blur_laplacian_threshold`, `blur_laplacian_outlier`, `blur_tenengrad_*`, `vincents_artefacts_*`, `small_object`, `incomplete_shape`, ...) each keep their own bar — the `_threshold` and `_outlier` variants get their own descriptive labels so below-the-floor rejections stay distinct from extreme-bad-outlier rejections.

### 2D Embedding Scatter Plots (`selection/embedding_space/2D_DR_plots/`)

Each point is an observation projected into 2D via a dimensionality reduction technique. Non-selected points are colour-mapped by quality score; selected points are marked with numbered black-outlined circles. Grey connection lines (PCA only) trace each non-selected view to its nearest selected neighbour by cosine similarity.

| File | Method | Colormap | Color Range |
|------|--------|----------|-------------|
| `selection_embedding.png` | PCA | jet | [0, 1] |
| `selection_embedding_scaled.png` | PCA | viridis | [min, max] |
| `embedding_mds.png` | MDS | viridis | [min, max] |
| `embedding_*.png` | t-SNE / UMAP / Isomap / LLE / LDA | viridis | [min, max] |

Every method also gets a **cluster-coloured twin** (`clusters_embedding_<method>.png`) that reuses the exact same coordinates but colours each point by its k-means cluster (discrete `tab20` colourbar) and marks the selected views as gold stars. k-means is fit over the embedding rows with `k = --n_clusters` (default 10, from the selection's `--kmeans_xnn_k` when run via `run.py`).

### 3D Interactive Scatter Plots (`selection/embedding_space/3D_DR_plots/`)

Plotly HTML files with the same colour scheme as the 2D variants but in three dimensions. Hover over points to see index and quality score; selected points are labelled with their rank number. The cluster-coloured variants (`clusters_embedding_<method>_3d.html`) use the same discrete k-means colouring.

### Quality-Criteria DR Plots (`selection/quality_criteria/DR_plots/`)

The same dimensionality-reduction method set run over the **normalised quality-criteria matrix** instead of the embedding vectors. Each observation is a vector of its per-criterion metrics (Laplacian, Tenengrad, area ratio, border ratio, edge ratio, hand overlap, Vincent area/artifact fractions, boundary-blur variance, solidity, extent, convexity, completeness, and the quality components `blur`/`area`/`vincents_artefacts`/`centerness`/`confidence`), min-max scaled column-wise to `[0, 1]` so no single criterion dominates. Columns missing on any observation are dropped, so the plot degrades gracefully with partial metric data. Cluster labels are k-means over the criteria matrix itself (default `k = --n_clusters`). LDA renders here too because the k-means clusters provide the class labels LDA needs for 2D/3D.

| File | Meaning |
|------|---------|
| `selection_criteria.png` | PCA, quality-score colour, jet [0, 1] |
| `selection_criteria_scaled.png` | PCA, quality-score colour, viridis [min, max] |
| `criteria_<method>.png` / `criteria_<method>_3d.html` | quality-coloured DR per method |
| `clusters_criteria_<method>.png` / `_3d.html` | cluster-coloured DR per method |

## Debug Mode

By default only **PCA** and **MDS** plots are generated. Pass `--debug` (or set `cfg.debug = True`) to include:

- t-SNE
- UMAP
- Isomap
- LLE
- LDA (class labels are k-means clusters over the space, so the 2D/3D reductions always render; without enough clusters it degrades to the selected-vs-non-selected single component and is skipped gracefully)

### Embedding Neighbour Diagnostics (`selection/`)

`--debug` also enables three diagnostics (in `plotting_process/neighbor_plots.py`) that check whether the embedding space actually groups *similar-looking* frames together. For every final selected candidate, `NEIGHBORS_PER_CANDIDATE = 5` neighbours are shown:

- `selected_neighbors_knn.png` — one row per selected candidate: the candidate plus its **5 nearest neighbours by cosine distance** in embedding space. Each cell shows the frame id and, for neighbours, the cosine distance to the candidate; if the embeddings are meaningful, the neighbours should look like the candidate.
- `selected_neighbors_kmeans.png` — the same grid, but neighbours are taken **from the candidate's k-means cluster** first (k-means is fit over the pool embeddings with `k` = number of selected views), only falling back to the overall nearest neighbours when the cluster has fewer than 5 members. This highlights whether k-means cluster membership matches visual similarity.
- `selected_clusters_pca.png` — a PCA 2D scatter of the whole selection pool coloured by k-means cluster assignment, with the final selected candidates marked as numbered gold stars.

These are pure diagnostics: they never change the pipeline output.

## Standalone Plotting

Re-generate plots from a previous pipeline run without re-running the pipeline:

```bash
python -m plotting_process.wrapper --input_dir /path/to/pipeline/outputs [--output_dir /path/to/plots] [--debug] [--n_clusters K]
```

`--n_clusters` controls the k-means clusters used for LDA labels and the `clusters_*` plots (default 10; match the pipeline's `--kmeans_xnn_k` for consistency).

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
| **LDA** | sklearn.discriminant_analysis | Supervised, maximises class separation (classes = k-means clusters over the space) |

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
