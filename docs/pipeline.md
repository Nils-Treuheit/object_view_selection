# Object View Selection Pipeline

## Overview

The pipeline selects the best **N image/mask pairs** from a set of observations of a single object, maximizing **object identifiability** — the selected views are high-quality, diverse, and non-redundant. The pipeline is fully modular: filters, quality metrics, embedding models, and subset selectors can be swapped independently.

```
  ┌─────────────┐
  │   Dataset   │  images/, masks/, object_hands/
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Pre-filter  │  Reject blurry, truncated, occluded, tiny, incomplete
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Quality    │  Score each surviving observation [0,1]
  │  Scorer     │  Weighted combination of component metrics
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Embeddings  │  DINOv3 / DINOv2 / SigLIP / CLIP / EVA-CLIP
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Selection  │  FPS / GQD / Facility Location / DPP / NBV
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Outputs   │  selected images, masks, report.json, quality.csv
  └─────────────┘
```

---

## Dataset Format

```
bottle/
├── images/          # 00000.png, 00001.png, ...   (RGB)
├── masks/           # 00000.png, 00001.png, ...   (binary, same filenames)
├── object_hands/    # 00000.png, 00001.png, ...   (binary, optional)
└── depth/           # 00000.png, 00001.png, ...   (optional, copied through to outputs)
```

All files are aligned by numeric stem. The `object_hands/` directory is optional — when absent, the occlusion filter passes all observations. `depth/` is not read by the pipeline, but matching depth files are copied into the sample outputs when present (see Stage 5).

---

## Stage 1: Pre-Filtering

Per-filter detail (algorithms, metrics, config, rejection behaviour) is in the
[`docs/pre-filter/`](pre-filter/README.md) reference.

The pre-filter runs the configured pipeline in order. The **default set** is deliberately small and conservative (port of `nit_view_selection/select_best_views.py`, reworked). Every non-binary default filter implements two `BaseFilter` rejection criteria, both on its **raw stat** in natural units:

- an **absolute garbage threshold** that filters out complete unusable garbage (`hard_min` floor / `hard_max` ceiling, reason `<reason>_threshold`), and
- a **population-based extreme-bad-outlier** rejection (`outlier_z`, robust median/MAD z over the population, reason `<reason>_outlier`).

Both are implemented once by the shared `ScoreFilter` base (`preprocessing/base.py` + `preprocessing/filter_utils.py`). The binary hard filters (empty mask, border pixel) reject on a structural condition with a bare reason. Every default filter returns `(score, passed, reason)` with `score ∈ [0, 1]`:

```python
score   ∈ [0, 1]    # pass/fail score
passed  ∈ {True, False}
reason  ∈ str       # rejection label (bare reason, or <reason>_threshold / <reason>_outlier)
```

### VincentEmptyMaskFilter (hard)

Ported from `nit_view_selection/select_best_views.py`. Rejects observations whose mask contains no foreground pixels at all.

**Metric:** `vincent_pixel_count = number of mask pixels > 0`

**Decision:** reject if `pixel_count <= 0` (reason `vincent_empty_mask`)

### VincentBorderPixelFilter (hard)

Ported from `nit_view_selection/select_best_views.py`. Rejects objects whose mask touches the image frame.

**Metric:** `vincent_touches_border` — true if any foreground pixel lies on the first/last row or column

**Decision:** reject if the mask touches the frame (reason `vincent_border_pixel`)

### BorderLaplacianBlurFilter (blur_laplacian)

Measures sharpness of the **object boundary band** — `band = dilate(mask) XOR erode(mask)` with an elliptical kernel of `stroke_width` pixels.

**Metric:** `laplacian = variance of the Laplacian of the grayscale image restricted to the band`

**Score:** `min(laplacian / max_variance, 1.0)`

**Rejection:** if the raw Laplacian variance falls below the absolute garbage floor `hard_min_variance` (reason `blur_laplacian_threshold`), or is an extreme bad outlier relative to the population (reason `blur_laplacian_outlier`).

**Config:** `LaplacianBlurConfig(stroke_width=9, max_variance=20000, hard_min_variance=4000, outlier_z=3.0)`

### BorderTenengradBlurFilter (blur_tenengrad)

Complementary boundary-band sharpness measure based on the gradient magnitude.

**Metric:** `tenengrad = mean Sobel magnitude of the grayscale image restricted to the band`

**Score:** `min(tenengrad / max_tenengrad, 1.0)`

**Rejection:** same garbage-floor / population-outlier scheme (reasons `blur_tenengrad_threshold` / `blur_tenengrad_outlier`).

**Config:** `TenengradBlurConfig(stroke_width=9, max_tenengrad=150, hard_min_tenengrad=33, outlier_z=3.0)`

### VincentsArtifactsFilter (vincents_artefacts)

Ported from `nit_view_selection/select_best_views.py`. Penalises speckled, holed, or ragged mask edges.

**Metric:** `vincent_artifact_fraction = |open(mask) XOR close(mask)| / mask_pixels` (kernel size 3)

**Score:** `min(1 - artifact_fraction / max_fraction, 0)` clamped — i.e. the fraction anchored by `max_fraction`.

**Rejection:** both `BaseFilter` criteria on the raw artifact fraction — an absolute garbage ceiling `hard_max_fraction` (reason `vincents_artefacts_threshold`) and population-outlier removal (reason `vincents_artefacts_outlier`).

**Config:** `VincentsArtifactsConfig(kernel_size=3, max_fraction=0.05, hard_max_fraction=0.15, outlier_z=3.0)`

### Rejection Layering

Rejections are grouped on disk under `rejected_samples/<reason>/threshold-based/` (the `_threshold` variants, plus the pure hard filters) and `rejected_samples/<reason>/outlier-based/` (the `_outlier` variants). See `preprocessing/base.py` (`ScoreFilter`) and `save_rejected_samples_by_reason` in `run.py`. The legacy filters — which implement only their own absolute cutoff — get the population outlier layered on by `OutlierFilter` (`preprocessing/variants.py`) when their config sets `outlier_z`.

### Soft pre-filters (population-adapted weights, optional)

Two Vincent soft filters remain available (`VincentsAreaFilter`, `VincentsMotionBlurFilter`). They compute a raw stat per observation, then a population pass converts it into a selection weight in (0, 1] using a robust median/MAD typical scale and a one-sided half-Gaussian falloff on the "bad" side. Raw stats are computed for all observations; the weight pass is fit **on the accepted set only**:

```
weight = exp(-0.5 * (z / softness)^2),   z = (stat - median) / (MAD * 1.4826)
```

- **VincentsAreaFilter** — `stat = vincent_area_fraction = mask_area / canvas_area`. Small masks are penalized (`low_bad`, `softness=0.3`). Weight stored as `vincents_area`. As a `ScoreFilter` it also implements the absolute garbage floor (`hard_min_area_fraction`, disabled by default) and the `outlier_z` population removal, so it can act as a working pre-filter.
- **VincentsMotionBlurFilter** — `stat = vincent_boundary_blur_variance` = variance of the Laplacian restricted to the boundary band. Blurred boundaries are penalized (`low_bad`, `softness=0.3`, `stroke_width=9`). Weight stored as `vincents_motion_blur`. It is also a working pre-filter: `evaluate` reports a quality-scaled stat score and implements both `BaseFilter` rejection criteria — an absolute `hard_min_variance` floor (reason `vincents_motion_blur_threshold`, forwarded from config and active by default at 120.0) and a population-based `outlier_z` removal (reason `vincents_motion_blur_outlier`, fit over the population via `fit`/`need_fitting`). See `docs/pre-filter/vincents_motion_blur.md`.

### Filter Pipeline Order

The default order in `FilterConfig.filter_order` is:

1. **vincent_empty_mask** — cheapest (mask pixel count)
2. **vincent_border_pixel** — cheap (frame contact)
3. **blur_laplacian** — boundary-band sharpness
4. **blur_tenengrad** — boundary-band gradient sharpness
5. **vincents_artefacts** — mask artifact score

Early filters reject quickly, avoiding unnecessary computation. The population outlier statistics are fit **once on the full dataset** before the filtering loop (`filter_pipeline.fit_observations`, only for filters with `outlier_z` set); scores are then compared against that robust median/MAD distribution during the pass.

**Legacy filters** (`border`, `area`, `occlusion`, `confidence`, `completeness`, and the old whole-image `blur`) are kept for custom `--filter_order` runs but are NOT part of the default set and are not tested / likely not working as proper pre-filters.

---

## Stage 2: Quality Scoring

Every observation that passes the pre-filter receives a **quality score** `Q ∈ [0,1]`.

### Weighted Scorer

The scorer combines exactly **4 components**:

```python
Q = w_blur·S_blur + w_area·S_area + w_vincents_artefacts·S_vincents_artefacts
  + w_centerness·S_centerness
```

All weights are configured in `QualityWeights` and sum to 1.

| Component | Default Weight | Description |
|-----------|---------------|-------------|
| blur | 0.30 | Boundary-band sharpness (reads the `laplacian` pre-filter stat) |
| area | 0.20 | Object size relative to image |
| vincents_artefacts | 0.20 | Mask artifact fraction |
| centerness | 0.30 | Mask centredness |

### Per-Component Scores

- **BorderBlurQuality:** `min(laplacian / max_variance, 1.0)` where `max_variance` is the fixed global anchor `quality_anchors.blur_max_variance` (default 10000). If the `laplacian` pre-filter stat is absent (e.g. a standalone scorer), it computes the boundary-band Laplacian variance directly from the image and mask (stroke width 9).
- **AreaQuality:** `min(area_ratio / 0.20, 1.0)` where `area_ratio` is computed from the mask and 0.20 is `quality_anchors.area_max_fraction`. Scores increase linearly up to 20% image coverage.
- **VincentsArtifactsQuality:** `min(1 - vincent_artifact_fraction / max_fraction, 0)` clamped, anchored by `quality_anchors.artifacts_max_fraction` (default 0.05).
- **CenternessQuality:** how centred the object's **center point** (mask centroid) is in the frame. The centroid at the exact frame centre scores the perfect 1.0; shifting the center point in the interior costs only a light quadratic decrease, but once the center point enters the `BORDER_ZONE_PX = 20` px band along any image border the score falls off exponentially (objects grazing the frame edge get crushed).

### Confidence (post-hoc, diagnostic)

`confidence` is exported to `quality.csv` for diagnostics but is **not** a scorer component. It is computed as `blur · area · vincents_artefacts · centerness` — the product across all quality dimensions, so any weak dimension drags it down hard.

### Quality Floor (Stage 4)

After quality scoring, an adaptive floor (see `docs/scoring.md`) excludes the worst tail of the accepted pool from the **embedding selection pool**. Only observations with `quality >= floor` compete in the selector, so the selected set always meets a minimum quality. See `compute_quality_floor` in `run.py`.

### ObservationMetrics Dataclass

All computed values are stored on `observation.metrics` (see
`data_io/metrics.py`):

```python
# preprocessing (legacy / shape)
laplacian, tenengrad, area_ratio, border_ratio,
edge_top_ratio, edge_bottom_ratio, edge_left_ratio, edge_right_ratio, edge_ratio,
hand_overlap, solidity, extent, convexity, completeness,

# vincent pre-filters (raw stats + soft weights)
vincent_pixel_count, vincent_touches_border,
vincent_area_fraction, vincent_artifact_fraction, vincent_boundary_blur_variance,
vincents_area, vincents_artefacts, vincents_motion_blur,

# quality
blur, area, centerness, confidence,

# final score
quality
```

`border_free` (and the `score` alias) are `quality.csv` columns, not metric
fields: `border_free = 1 - border_ratio` and `score = quality`.

Quality scores and metrics are exported to `quality.csv`.

---

## Stage 2.5: Auto-Threshold Tuning

When `auto_thresholds: True` (default), the pipeline pre-computes thresholds from the dataset before the pre-filter runs. This tunes the **legacy** filters only.

### Strategy

1. Run every filter in eval-only mode on all observations.
2. Collect metric values across the full dataset.
3. Compute a percentile of each metric distribution.
4. Clamp the percentile value within hard safety limits.

| Threshold | Percentile | Direction | Safety Limits |
|-----------|-----------|-----------|---------------|
| `area_minimum_ratio` | 1st | low end | [0.01, 0.05] |
| `border_maximum_ratio` | 95th | high end | [0.001, 0.05] |
| `border_edge_maximum_ratio` | 95th | high end | [0.05, 0.5] |
| `occlusion_maximum_overlap` | 95th | high end | [0.001, 0.30] |
| `completeness_minimum_score` | 1st | low end | [0.50, 0.80] |

The **default blur/artifact pre-filters do not need tuning**: they use static relaxed floors plus population-relative outlier rejection (see `preprocessing/variants.py`), so auto-tuning is skipped for them.

Safety limits ensure a minimum quality bar even for garbage datasets, and prevent overly aggressive rejection on clean datasets.

Disabled by passing `--no-auto-thresholds` or setting `auto_thresholds: False`.

### Threshold Computation Algorithm

```python
stats = compute_metric_stats(observations)
p_border = np.percentile(stats["border_ratio"], 95)
border_max = np.clip(p_border, 0.001, 0.05)
```

See `utils/threshold_tuner.py` and `docs/thresholds.md` for details.

---

## Stage 3: Embedding Extraction

Each accepted observation is encoded into a feature vector for downstream selection.

### Learned Embeddings (GPU)

| Model | Auto-detected Name | Type | Output Dim |
|-------|-------------------|------|-----------|
| DINOv3 | `facebook/dinov3-*` | Vision Transformer | 768 (base) / 1024 (large) |
| DINOv2 | `dinov2_*` | Vision Transformer | 384 (small) / 768 (base) |
| SigLIP2 | `google/siglip2-*` | Multi-modal | 768 (base) |
| SigLIP | `google/siglip-*` | Multi-modal | 768 (base) |
| MoonViT | `moonshotai/MoonViT-*` | Vision Transformer | 1024 |
| CLIP | `ViT-*` or `openai/clip-*` | Multi-modal | 512 (base) |
| EVA-CLIP | `eva-clip` | Multi-modal | 1024 |

The embedding type is **inferred automatically** from the model name. Override with `--embedding`:

```bash
python run.py --embedding dinov2 --embedding_model dinov2_vitb14_reg
```

### Shape Descriptors (CPU, no GPU needed)

For environments without a GPU, classical shape descriptors can be used instead:

| Descriptor | Dim | Invariance |
|-----------|-----|-----------|
| Hu moments | 7 | translation, rotation, scale |
| Zernike moments | 65 (degree 10) | rotation |
| Fourier descriptors | 32 | translation, rotation, scale, start point |
| Shape Context | 60 | translation |

Enable with `--use_shape_descriptors`.

### Cropping

Before encoding, each frame becomes a **grown-mask cut-out on a static maximum-contrast background**, resized to 224×224:

- **Contrast background:** decided *once* over the whole pool from the **original** mask's border pixels of every observation (the object's own edge colour). A mostly bright border set maps to black (0), a mostly dark border set maps to white (255), so the object always sits on the most contrasty backdrop possible.
- **Cut-out:** the mask is grown by 5 px; the cut-out is the image masked by that *grown* mask, so the object plus a thin local-context margin keeps its original pixels. The cut-out is centred on a square canvas whose remaining area is filled with the static background colour.
- **Alpha channel (RGBA):** encoders that can ingest a 4-channel input (`EmbeddingModel.accepts_rgba = True`) receive an alpha channel encoding the region type — `1.0` over the original mask, `0.8` over the grown cut-out margin, `0.66` over the static background. The built-in models all normalise to 3 channels and keep `accepts_rgba = False`; RGB values are identical either way.
- **Bbox crop:** crops the image to the object bounding box (used by `compute_contrast_background`).
- **Masked crop:** applies the mask to the bbox crop (black background).

The crop helpers live in `embeddings/crop.py` (`grow_mask`, `compute_contrast_background`, `contrast_input`, `contrast_mask`). `EmbeddingModel.set_background()` sets the colour computed at run time (`run.py` / the webapp snapshot generator compute it over the pool before encoding); `background=None` keeps the legacy zero-padded square crop.

---

## Stage 4: Subset Selection

From the pool of accepted observations with quality scores and embeddings, select **N** views optimizing a chosen criterion.

### FarthestPointSampling (FPS)

**Objective:** maximize the minimum embedding distance within the selected set.

**Algorithm:**
1. Pick a random starting observation.
2. Repeatedly pick the observation farthest from the already-selected set.

**Use case:** maximal diversity, ignores quality.

### GreedyQualityDiversity (GQD) — DEFAULT

**Objective:** `Score(i) = α·quality(i) + β·min_cosine_distance(i, selected_set)`

**Algorithm:**
1. Start with the highest-quality observation.
2. Greedily pick the observation maximizing the weighted sum of quality and diversity.

**Parameters:** `selector_alpha` (quality weight, default 0.60), `selector_beta` (diversity weight, default 0.40).

**Use case:** balanced quality-diversity trade-off.

### FacilityLocation

**Objective:** maximize `Σ_j max_{i∈S} sim(j,i)` — total similarity from all pool observations to their nearest selected representative.

**Algorithm:**
1. Start with the most central observation (highest total similarity to all others).
2. Greedily pick the observation that maximizes the total coverage.

**Use case:** dataset summarization / representative subset.

### DPPSelector

**Objective:** maximize `det(L_S)` where `L = diag(q) · K · diag(q)` — the determinant of the quality-weighted similarity kernel over the selected set.

**Algorithm:** Greedy MAP inference (iteratively pick the element that maximizes log-determinant gain).

**Parameters:** `dpp_sigma` (similarity kernel bandwidth, default 0.5).

**Use case:** most principled diversity-quality trade-off, but O(N³) per step.

### NextBestView (NBV)

**Objective:** `quality(i) + 0.5 × mean_distance(i, selected_set)`

Designed for datasets with camera pose information (e.g., NeRF, Gaussian Splatting). Without poses, uses embedding distance as a proxy.

**Algorithm:**
1. Start with highest quality.
2. Greedily pick the remaining observation with the best quality+diversity score.

### TopKMeansXNN (Top kMeans Embedding Selection in xNN quality Neighborhood)

**Objective:** one pick per k-means cluster — the best-quality pool sample inside the cluster centroid's `{centroid + xNN}` neighbourhood.

**Algorithm:**
1. Run k-means with `k = num_views`; seed centres by farthest-point sampling (`kmeans_init=farthest`) or the top-quality samples (`kmeans_init=best_quality`).
2. For each cluster, restrict the centroid's x-nearest-neighbours to samples closer to *this* centroid than any other (cluster members only), fall back to the medoid.
3. Pick the highest-quality candidate from that constrained set.

**Parameters:** `kmeans_init` (`farthest`/`best_quality`), `kmeans_xnn_k` (3/5/10).

---

## Stage 5: Outputs

```
outputs/
├── report.json              # Full pipeline report + selection metrics
├── quality.csv              # Per-observation quality metrics + "selected" flag
├── embeddings.npy           # Embedding matrix (selection pool), --save_embeddings
├── selected_indices.npy     # Row indices into embeddings.npy (pool order)
├── selection_pool_ids.npy   # Frame ids aligned with embeddings.npy rows
├── rejected.json            # [{id, reason}, ...]
├── rejected_metrics.csv     # Per-rejected-observation metrics, --save_rejected
├── visualization.png        # Overview grid of selected views, --save_visualization
├── plots/                   # Diagnostic plots, --save_plots (see docs/plotting.md)
├── bad_examples/            # Worst rejected frames (plotting, --save_plots)
├── embedded_samples/        # What each embedding actually saw, --save_plots
├── selected_samples/        # Selected tuples, re-organized by data type:
│   └── <obj_id>/            #   named after the last component of data_root
│       ├── rgb/             #   selected object images
│       ├── mask/            #   selected object masks
│       ├── depth/           #   only when <data_root>/depth exists
│       └── hand_mask/       #   only when a hand mask is available
├── accepted_samples/        # Accepted-but-unselected tuples (--debug only), same layout
└── rejected_samples/        # Rejected tuples grouped by rejection reason:
    └── <reason>/            #   e.g. vincent_border_pixel, blur_laplacian,
        ├── threshold-based/ #   <reason>_threshold variants (below the
        │   └── <obj_id>/    #   relaxed absolute floor; also the pure hard
        │       ├── rgb/     #   reasons like vincent_empty_mask)
        │       ├── mask/
        │       ├── depth/   #   only when <data_root>/depth exists
        │       └── hand_mask/
        └── outlier-based/   #   <reason>_outlier variants (extreme bad
            └── <obj_id>/    #   outliers relative to the population)
                ├── rgb/
                ├── mask/
                ├── depth/
                └── hand_mask/
```

Defaults: `save_rejected`, `save_embeddings` and `save_visualization` are
`True`; `save_plots` and `debug` are `False`. `depth/` frames are copied
through only for frames that have a matching file in `<data_root>/depth`
(`.png`, `.npy`, `.jpg`, `.jpeg`, `.tiff`); the loader itself only reads
`images/`, `masks/` and `object_hands/`.

### report.json includes (excerpt):

```json
{
  "total": 423,
  "accepted": 359,
  "rejected": 64,
  "selected": 10,
  "num_views": 10,
  "embedding": "dinov3",
  "embedding_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
  "selector": "quality_diversity",
  "quality_floor": 0.42,
  "selection_pool_count": 359,
  "data_root": "/path/to/bottle",
  "accepted_ids": [1, 2, 3, ...],
  "selection_pool_ids": [1, 2, 3, ...],
  "selected_ids": [31, 169, 317, ...],
  "rejected_ids": [4, 7, 9, ...],
  "selection_metrics": {
    "selector": "quality_diversity",
    "num_views": 10,
    "selected_count": 10,
    "intra_set": {
      "mean_pairwise_cosine_distance": 0.49,
      "min_pairwise_cosine_distance": 0.31,
      "max_pairwise_cosine_distance": 0.71,
      "mean_similarity": 0.51
    },
    "quality": {
      "selected_mean": 0.80,
      "selected_min": 0.61,
      "selected_max": 0.94,
      "pool_mean": 0.79,
      "pool_min": 0.42,
      "quality_floor": 0.42,
      "selection_pool_count": 359
    },
    "coverage": {
      "mean_min_cosine_dist_to_selected": 0.19,
      "median_min_cosine_dist_to_selected": 0.14,
      "pool_covered_within_05": 349,
      "pool_covered_within_03": 327,
      "total_unselected": 349
    },
    "selection_log": [
      {"step": 0, "id": 31, "quality": 0.84, "min_cosine_dist_to_set": null, "score": null},
      {"step": 1, "id": 169, "quality": 0.78, "min_cosine_dist_to_set": 0.71, "score": 0.74}
    ]
  }
}
```

`selection_log` is only populated for the `quality_diversity` selector; the
other selectors leave it empty.

---

## Configuration Reference

All pipeline parameters are defined in `config.py`.

| Category | Field | Default | Description |
|----------|-------|---------|-------------|
| **Filters** | `blur_laplacian.enabled` | `True` | Boundary-band Laplacian pre-filter |
| | `blur_laplacian.stroke_width` | 9 | Boundary-band stroke width |
| | `blur_laplacian.max_variance` | 20000 | Score anchor for Laplacian variance |
| | `blur_laplacian.hard_min_variance` | 4000 | Absolute garbage floor on the raw stat |
| | `blur_laplacian.outlier_z` | 3.0 | Robust population-outlier z |
| | `blur_tenengrad.enabled` | `True` | Boundary-band Tenengrad pre-filter |
| | `blur_tenengrad.stroke_width` | 9 | Boundary-band stroke width |
| | `blur_tenengrad.max_tenengrad` | 150 | Score anchor for Tenengrad magnitude |
| | `blur_tenengrad.hard_min_tenengrad` | 33 | Absolute garbage floor on the raw stat |
| | `blur_tenengrad.outlier_z` | 3.0 | Robust population-outlier z |
| | `vincents_artefacts.kernel_size` | 3 | Morphology kernel for artifact detection |
| | `vincents_artefacts.max_fraction` | 0.05 | Artifact fraction at which the score hits 0 |
| | `vincents_artefacts.hard_max_fraction` | 0.15 | Absolute garbage ceiling on the raw stat |
| | `vincents_artefacts.outlier_z` | 3.0 | Robust population-outlier z |
| | `vincent_empty_mask.enabled` | `True` | Empty-mask hard filter |
| | `vincent_border_pixel.enabled` | `True` | Border-pixel hard filter |
| | `vincents_area.softness` | 0.3 | Area weight falloff (robust-MADs) |
| | `vincents_area.hard_min_area_fraction` | 0.0 | Absolute garbage floor on area fraction (0 disables) |
| | `vincents_motion_blur.softness` | 0.3 | Boundary-blur weight falloff (robust-MADs) |
| | `vincents_motion_blur.stroke_width` | 9 | Boundary-band stroke width |
| | `vincents_motion_blur.hard_min_variance` | 120 | Absolute floor on band variance (active — forwarded by `build_soft_filters`; reason `vincents_motion_blur_threshold`) |
| | `filter_order` | `[vincent_empty_mask, vincent_border_pixel, blur_laplacian, blur_tenengrad, vincents_artefacts]` | Pre-filter execution order |
| | `area` / `border` / `occlusion` / `confidence` / `completeness` | — | Legacy filters, custom `--filter_order` only (not tested / likely not working) |
| **Quality** | `quality_weights.blur` | 0.30 | Boundary-blur quality weight |
| | `quality_weights.area` | 0.20 | Area quality weight |
| | `quality_weights.vincents_artefacts` | 0.20 | Mask-artifact quality weight |
| | `quality_weights.centerness` | 0.30 | Centerness quality weight |
| | `quality_anchors.blur_max_variance` | 10000 | Global blur anchor |
| | `quality_anchors.area_max_fraction` | 0.20 | Global area anchor |
| | `quality_anchors.artifacts_max_fraction` | 0.05 | Global artifact anchor |
| **Embedding** | `embedding` | `auto` | Type (inferred from model name) |
| | `embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model HF ID or path |
| | `use_shape_descriptors` | `False` | Use CPU-based shape descriptors |
| | `shape_descriptor` | `hu` | Shape descriptor type |
| **Selection** | `selector` | `quality_diversity` | Selection algorithm |
| | `selector_alpha` | 0.60 | Quality weight (GQD) |
| | `selector_beta` | 0.40 | Diversity weight (GQD) |
| | `dpp_sigma` | 0.5 | Similarity bandwidth (DPP) |
| | `num_views` | 10 | Number of views to select |
| **Global** | `auto_thresholds` | `True` | Enable data-driven threshold tuning |
| | `save_visualization` | `True` | Save overview grid |
| | `save_rejected` | `True` | Save rejected images |
| | `save_embeddings` | `True` | Save embedding matrix |

