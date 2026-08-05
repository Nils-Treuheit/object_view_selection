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
└── object_hands/    # 00000.png, 00001.png, ...   (binary, optional)
```

All files are aligned by numeric stem. The `object_hands/` directory is optional — when absent, the occlusion filter passes all observations.

---

## Stage 1: Pre-Filtering

The pre-filter runs a pipeline of **hard filters** in a configurable order, followed by a **soft pre-filter pass** that computes population-adapted weights. All hard filters must pass for an observation to survive. Each hard filter returns `(score, passed, reason)`:

```python
score   ∈ [0, 1]    # pass/fail score
passed  ∈ {True, False}
reason  ∈ str       # rejection label
```

### BorderFilter

Detects objects truncated by the image edge.

**Metrics:**

- `border_ratio = (mask ∩ border_pixels) / total_mask_pixels`

  The image border is defined as the first/last row and first/last column of the image. Any mask pixel on these edges counts as a border pixel. This catches objects that are flush against the frame.

- `edge_ratio = max(top, bottom, left, right)` where each per-edge value is the length of mask touching that image edge divided by the mask extent in the perpendicular direction:

  ```
  edge_top    = |mask[0, :]| / |columns the mask occupies|
  edge_bottom = |mask[-1, :]| / |columns the mask occupies|
  edge_left   = |mask[:, 0]| / |rows the mask occupies|
  edge_right  = |mask[:, -1]| / |rows the mask occupies|
  ```

  This is the key signal for truncation: an object cut off along an edge pins a large fraction of its width/height to the frame, so the ratio approaches 1. A fully-visible object that merely grazes an edge only touches a few pixels, keeping the ratio near 0. This catches objects that are "mostly cut off" even when their visible part is large (where `border_ratio` alone stays below the threshold).

The per-edge values are also stored as `edge_top_ratio`, `edge_bottom_ratio`, `edge_left_ratio`, `edge_right_ratio`.

**Decision:** reject if `border_ratio > maximum_ratio` OR `edge_ratio > edge_maximum_ratio`

**Config:** `BorderConfig(maximum_ratio=0.05, edge_maximum_ratio=0.25)` — default 5% ring contact and 25% edge contact, but auto-tuning typically sets these much tighter.

### AreaFilter

Filters out objects that are too small relative to the image.

**Metric:** `area_ratio = mask_area / image_area`

**Decision:** reject if `area_ratio < minimum_ratio`

**Config:** `AreaConfig(minimum_ratio=0.01)`

### BlurFilter

Measures image sharpness using two complementary metrics.

**Metrics:**

- **Variance of Laplacian (VaL):** computes the Laplacian of the grayscale image, then takes the variance. Low VaL = blurry.
- **Tenengrad:** computes the mean magnitude of the Sobel gradient. Low Tenengrad = blurry.

**Score:** `0.5 * (VaL / threshold) + 0.5 * (Tenengrad / tenengrad_threshold)`, clipped to [0, 1].

**Decision:** reject if **both** `VaL < laplacian_threshold` AND `Tenengrad < tenengrad_threshold`.

### OcclusionFilter

Detects hand or object occlusions using `object_hand` masks.

**Metric:** `hand_overlap = (mask ∩ object_hand) / mask_pixels`

If no `object_hand` is provided for an observation, the filter passes unconditionally.

**Decision:** reject if `hand_overlap > maximum_overlap`

### ConfidenceFilter

An optional detector-confidence gate. Reads `observation.confidence` (set upstream by a detector like SAM or Mask2Former). If no confidence is available (None), the filter passes unconditionally.

**Decision:** reject if `confidence < minimum_confidence`

Disabled by default (`ConfidenceConfig.enabled = False`).

### CompletenessFilter

Measures how complete the visible object shape is using three geometric cues:

- **Solidity:** `contour_area / convex_hull_area` — how jagged/perforated the mask is
- **Extent:** `contour_area / bounding_box_area` — how spread out the mask is within its bbox
- **Convexity:** `convex_hull_perimeter / contour_perimeter` — how concave the boundary is

**Score:** `0.4 × solidity + 0.3 × extent + 0.3 × convexity`

**Decision:** reject if `score < minimum_score`

### VincentEmptyMaskFilter (hard)

Ported from `nit_view_selection/select_best_views.py`. Rejects observations whose mask contains no foreground pixels at all.

**Metric:** `vincent_pixel_count = number of mask pixels > 0`

**Decision:** reject if `pixel_count <= 0` (reason `vincent_empty_mask`)

### VincentBorderPixelFilter (hard)

Ported from `nit_view_selection/select_best_views.py`. Rejects objects whose mask touches the image frame.

**Metric:** `vincent_touches_border` — true if any foreground pixel lies on the first/last row or column

**Decision:** reject if the mask touches the frame (reason `vincent_border_pixel`)

### Soft Pre-Filter Pass (population-adapted)

Ported from `nit_view_selection/select_best_views.py`. The soft pre-filters never hard-reject. They compute a raw per-observation stat, then a **population pass** converts those stats into selection weights in **(0, 1]** using a robust median/MAD typical scale and a one-sided half-Gaussian falloff on the "bad" side:

```
weight = exp(-0.5 * (z / softness)^2),   z = (stat - median) / (MAD * 1.4826)
```

Raw stats are computed for all observations (accepted + rejected, for diagnostics); weights are fit only on the accepted set.

- **VincentsAreaFilter** — `stat = vincent_area_fraction = mask_area / canvas_area`. Small masks are penalized (`low_bad`, `softness=0.3`). Weight stored as `vincents_area`.
- **VincentsArtifactsFilter** — `stat = vincent_artifact_fraction = |open(mask) XOR close(mask)| / mask_pixels`. Speckled/holed/ragged masks are penalized (`high_bad`, `softness=3.0`, `kernel_size=3`). Weight stored as `vincents_artefacts`.
- **VincentsMotionBlurFilter** — `stat = vincent_boundary_blur_variance` = variance of the Laplacian restricted to the boundary band (`dilate(mask) XOR erode(mask)`, `stroke_width=9`). Blurred boundaries are penalized (`low_bad`, `softness=0.3`). Weight stored as `vincents_motion_blur`.

### Filter Pipeline Order

Hard filters are defined in `FilterConfig.filter_order`. The default order is chosen for efficiency and specificity:

1. **vincent_empty_mask** — cheapest (mask pixel count)
2. **vincent_border_pixel** — cheap (frame contact)
3. **border** — catches truncation (ring contact + per-edge contact)
4. **area** — cheap, rejects tiny masks
5. **confidence** — cheap (disabled by default)
6. **blur** — moderately expensive (requires image)
7. **occlusion** — moderately expensive (requires hand mask)
8. **completeness** — most expensive (requires contour finding)

Early filters reject quickly, avoiding unnecessary computation. After the hard pass, the soft pre-filter pass runs on the accepted set.

---

## Stage 2: Quality Scoring

Every observation that passes the pre-filter receives a **quality score** `Q ∈ [0,1]`.

### Weighted Scorer

```python
Q = w_blur·S_blur + w_area·S_area + w_occlusion·S_occlusion
  + w_completeness·S_completeness
  + w_vincents_area·S_vincents_area + w_vincents_artefacts·S_vincents_artefacts
  + w_vincents_motion_blur·S_vincents_motion_blur
```

All weights are configured in `QualityWeights` and sum to 1.

| Component | Default Weight | Description |
|-----------|---------------|-------------|
| completeness | 0.35 | Mask shape completeness |
| blur | 0.20 | Image sharpness |
| occlusion | 0.20 | Freedom from hand overlap |
| area | 0.15 | Object size relative to image |
| confidence | 0.10 | Detection confidence (computed post-hoc as weakest-link) |
| vincents_area | 0.10 | Population-adapted mask area weight |
| vincents_artefacts | 0.10 | Population-adapted mask artifact weight |
| vincents_motion_blur | 0.10 | Population-adapted boundary blur weight |

### Per-Component Scores

- **BlurQuality:** `min(laplacian / max_lap, 1.0)` where `max_lap` is the fixed global anchor `quality_anchors.blur_max_lap` (default 400). This makes sharpness scores comparable across datasets.
- **AreaQuality:** `min(area_ratio / 0.20, 1.0)`. Scores increase linearly up to 20% image coverage.
- **OcclusionQuality:** `1.0 - hand_overlap`. Perfect occlusion gives 1.0, full occlusion gives 0.0.
- **CompletenessQuality:** returns `observation.metrics.completeness` directly (already in [0,1]).
- **VincentsAreaQuality / VincentsArtifactsQuality / VincentsMotionBlurQuality:** anchored in fixed global max values (see `quality_anchors`) applied to the raw stats `vincent_area_fraction`, `vincent_artifact_fraction`, `vincent_boundary_blur_variance`. The population-adapted weights (`vincents_area`, `vincents_artefacts`, `vincents_motion_blur`) remain pre-filter/diagnostic outputs.
- **Confidence** (post-hoc): computed as `min(blur, area, occlusion, completeness, vincents_area, vincents_artefacts, vincents_motion_blur)` — the weakest-link across all quality dimensions.

### Quality Floor (Stage 4)

After quality scoring, an adaptive floor (see `docs/scoring.md`) excludes the worst tail of the accepted pool from the **embedding selection pool**. Only observations with `quality >= floor` compete in the selector, so the selected set always meets a minimum quality. See `compute_quality_floor` in `run.py`.

### ObservationMetrics Dataclass

All computed values are stored on `observation.metrics`:

```python
laplacian, tenengrad, area_ratio, border_ratio, hand_overlap,
solidity, extent, convexity, completeness,
blur, area, occlusion, confidence,
vincent_pixel_count, vincent_touches_border,
vincent_area_fraction, vincent_artifact_fraction, vincent_boundary_blur_variance,
vincents_area, vincents_artefacts, vincents_motion_blur,
quality
```

Quality scores and metrics are exported to `quality.csv`.

---

## Stage 2.5: Auto-Threshold Tuning

When `auto_thresholds: True` (default), the pipeline pre-computes thresholds from the dataset before the pre-filter runs.

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
| `laplacian_threshold` | 5th | low end | [30, 200] |
| `tenengrad_threshold` | 5th | low end | [10, 60] |
| `occlusion_maximum_overlap` | 95th | high end | [0.001, 0.30] |
| `completeness_minimum_score` | 1st | low end | [0.50, 0.80] |

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
| Zernike moments | 27 | rotation |
| Fourier descriptors | 32 | translation, rotation, scale, start point |
| Shape Context | 60 | translation |

Enable with `--use_shape_descriptors`.

### Cropping

Before encoding, images are cropped to the object bounding box (dilated slightly) and resized to 224×224. Two crop modes are available:

- **Bbox crop:** crops the image to object bounding box
- **Masked crop:** applies the mask to the bbox crop (black background)

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

**Parameters:** `selector_alpha` (quality weight, default 0.45), `selector_beta` (diversity weight, default 0.55).

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
├── report.json             # Full pipeline report + selection metrics
├── quality.csv             # Per-observation quality metrics
├── embeddings.npy          # Embedding matrix (accepted pool)
├── selected_indices.npy    # Indices into embeddings.npy
├── rejected.json           # Rejection reasons
├── visualization.png       # Overview grid of selected views
├── selected_samples/       # Selected tuples, re-organized by data type:
│   └── <obj_id>/           #   named after the last component of data_root
│       ├── rgb/            #   selected object images
│       ├── mask/           #   selected object masks
│       ├── depth/          #   only when <data_root>/depth exists
│       └── hand_mask/      #   only when a hand mask is available
└── rejected_samples/       # Rejected tuples, same layout as selected_samples/
    └── <obj_id>/
        ├── rgb/
        ├── mask/
        ├── depth/
        └── hand_mask/
```

### report.json includes:

```json
{
  "total": 423,
  "accepted": 359,
  "rejected": 64,
  "selected_ids": [31, 169, 317, ...],
  "selection_metrics": {
    "intra_set": {
      "mean_pairwise_cosine_distance": 0.49,
      "min_pairwise_cosine_distance": 0.31,
      "max_pairwise_cosine_distance": 0.71
    },
    "quality": {
      "selected_mean": 0.80,
      "pool_mean": 0.79
    },
    "coverage": {
      "mean_min_cosine_dist_to_selected": 0.19,
      "pool_covered_within_03": 327
    },
    "selection_log": [
      {"step": 0, "id": 31, "quality": 0.84, "min_cosine_dist_to_set": null},
      {"step": 1, "id": 169, "quality": 0.78, "min_cosine_dist_to_set": 0.71, "score": 0.74}
    ]
  }
}
```

---

## Configuration Reference

All pipeline parameters are defined in `config.py`.

| Category | Field | Default | Description |
|----------|-------|---------|-------------|
| **Filters** | `blur.threshold` | 120.0 | Laplacian variance threshold |
| | `blur.tenengrad_threshold` | 35.0 | Tenengrad gradient threshold |
| | `area.minimum_ratio` | 0.01 | Minimum mask area ratio |
| | `border.maximum_ratio` | 0.05 | Maximum border-touching ratio |
| | `border.edge_maximum_ratio` | 0.25 | Maximum fraction of mask pinned to a frame edge |
| | `occlusion.maximum_overlap` | 0.15 | Maximum hand overlap ratio |
| | `completeness.minimum_score` | 0.65 | Minimum completeness score |
| | `vincent_empty_mask.enabled` | `True` | Enable empty-mask hard filter |
| | `vincent_border_pixel.enabled` | `True` | Enable border-pixel hard filter |
| | `vincents_area.softness` | 0.3 | Area weight falloff (robust-MADs) |
| | `vincents_artefacts.softness` | 3.0 | Artifact weight falloff (robust-MADs) |
| | `vincents_artefacts.kernel_size` | 3 | Morphology kernel for artifact detection |
| | `vincents_motion_blur.softness` | 0.3 | Boundary-blur weight falloff (robust-MADs) |
| | `vincents_motion_blur.stroke_width` | 9 | Boundary-band stroke width |
| | `filter_order` | `[vincent_empty_mask, vincent_border_pixel, border, area, confidence, blur, occlusion, completeness]` | Hard-pipeline execution order |
| **Quality** | `quality_weights.blur` | 0.20 | Blur quality weight |
| | `quality_weights.area` | 0.15 | Area quality weight |
| | `quality_weights.occlusion` | 0.20 | Occlusion quality weight |
| | `quality_weights.completeness` | 0.35 | Completeness quality weight |
| | `quality_weights.confidence` | 0.10 | Confidence quality weight |
| | `quality_weights.vincents_area` | 0.10 | Mask-area population weight |
| | `quality_weights.vincents_artefacts` | 0.10 | Mask-artifact population weight |
| | `quality_weights.vincents_motion_blur` | 0.10 | Boundary-blur population weight |
| **Embedding** | `embedding` | `auto` | Type (inferred from model name) |
| | `embedding_model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model HF ID or path |
| | `use_shape_descriptors` | `False` | Use CPU-based shape descriptors |
| | `shape_descriptor` | `hu` | Shape descriptor type |
| **Selection** | `selector` | `quality_diversity` | Selection algorithm |
| | `selector_alpha` | 0.4 | Quality weight (GQD) |
| | `selector_beta` | 0.6 | Diversity weight (GQD) |
| | `dpp_sigma` | 0.5 | Similarity bandwidth (DPP) |
| | `num_views` | 10 | Number of views to select |
| **Global** | `auto_thresholds` | `True` | Enable data-driven threshold tuning |
| | `save_visualization` | `True` | Save overview grid |
| | `save_rejected` | `True` | Save rejected images |
| | `save_embeddings` | `True` | Save embedding matrix |

