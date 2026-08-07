# Thresholds & Pre-Filtering

## Overview

The pipeline uses a mix of **data-driven auto-tuning** and **hard safety limits** for all pre-filter thresholds. Auto-tuning computes percentiles from the dataset itself, then clamps the result within sensible safety bounds so even a garbage dataset enforces a minimum quality bar.

Set `auto_thresholds: false` in `config.py` or pass `--no-auto-thresholds` to fall back entirely to the static config values.

---

## Auto-Tuning Strategy (`utils/threshold_tuner.py`)

Auto-tuning applies to the **legacy** filters only. The default blur/artifact pre-filters are not tuned — they use static relaxed floors plus population-relative outlier rejection.

| Filter | Metric | Percentile | Rationale |
|--------|--------|-----------|-----------|
| **Area** | `area_ratio` | **1st** (low end) | The smallest 1% of objects are rejected. Assumes >=99% of the dataset has usable object size. |
| **Border** | `border_ratio` | **95th** (high end) | The 5% of observations with the most border touching are rejected. Border touching is almost always bad, so this is quite aggressive. |
| **Border** | `edge_ratio` | **95th** (high end) | The 5% of observations with the most edge contact (object pinned to the frame) are rejected. Catches objects mostly cut off along an edge. |
| **Blur (legacy)** | `laplacian` | **5th** (low end) | Boundary-band Laplacian variance; computed but not consumed by the default pipeline. |
| **Blur (legacy)** | `tenengrad` | **5th** (low end) | Boundary-band Tenengrad; computed but not consumed by the default pipeline. |
| **Occlusion** | `hand_overlap` | **95th** (high end) | The 5% most occluded observations are rejected. |
| **Completeness** | `completeness` | **1st** (low end) | The 1% least-complete objects are rejected. Very permissive — only extreme cases. |

### How it works

1. Run every filter in eval-only mode on every observation to collect metric values.
2. Compute the specified percentile across the dataset.
3. Clamp the result to `[safety_min, safety_max]` (see below).
4. Use the clamped value as the filter threshold.

This means: if the dataset is already clean (e.g. no border-touching images), the 95th percentile of `border_ratio` is 0.0, so the threshold is clamped to `safety_min=0.001` — still enforcing a basic check.

---

## Safety Limits (Hard Fallbacks)

These are the absolute bounds enforced regardless of dataset statistics. Defined in `SAFETY_LIMITS` in `utils/threshold_tuner.py`.

| Threshold Key | Min | Max | Rationale |
|--------------|-----|-----|-----------|
| `area_minimum_ratio` | `0.01` (1%) | `0.05` (5%) | At least 1% of image must be object; never require more than 5% |
| `border_maximum_ratio` | `0.001` (0.1%) | `0.05` (5%) | At most 0.1% of mask pixels may border the edge; never allow more than 5% |
| `border_edge_maximum_ratio` | `0.05` (5%) | `0.5` (50%) | At most 5% of the mask extent may be pinned to a frame edge; never allow more than 50% |
| `laplacian_threshold` | `5.0` | `1000.0` | Boundary-band Laplacian variance clamp (legacy, not consumed by the default pipeline) |
| `tenengrad_threshold` | `3.0` | `100.0` | Boundary-band Tenengrad clamp (legacy, not consumed by the default pipeline) |
| `occlusion_maximum_overlap` | `0.001` (0.1%) | `0.30` (30%) | At most 0.1% overlap allowed at minimum; never allow more than 30% |
| `completeness_minimum_score` | `0.50` | `0.80` | At least 0.5 completeness score; never demand more than 0.8 |

---

## Static Defaults (`config.py`)

Used when `auto_thresholds: false` or when a tuned value is not provided.

### Default pre-filters

| Config Field | Default | Description |
|-------------|---------|-------------|
| `LaplacianBlurConfig.stroke_width` | `9` | Boundary-band stroke width |
| `LaplacianBlurConfig.max_variance` | `10000.0` | Score anchor for the Laplacian variance |
| `LaplacianBlurConfig.threshold_min` | `0.01` | Relaxed absolute floor (score) |
| `LaplacianBlurConfig.outlier_z` | `3.0` | Robust population-outlier z |
| `TenengradBlurConfig.stroke_width` | `9` | Boundary-band stroke width |
| `TenengradBlurConfig.max_tenengrad` | `100.0` | Score anchor for the Tenengrad magnitude |
| `TenengradBlurConfig.threshold_min` | `0.10` | Relaxed absolute floor (score) |
| `TenengradBlurConfig.outlier_z` | `3.0` | Robust population-outlier z |
| `VincentsArtifactsConfig.kernel_size` | `3` | Morphology kernel for artifact detection |
| `VincentsArtifactsConfig.max_fraction` | `0.05` | Artifact fraction at which the score hits 0 |
| `VincentsArtifactsConfig.threshold_min` | `0.05` | Relaxed absolute floor (score) |
| `VincentsArtifactsConfig.outlier_z` | `3.0` | Robust population-outlier z |

### Legacy filters (custom `--filter_order` only)

| Config Field | Default | Description |
|-------------|---------|-------------|
| `AreaConfig.minimum_ratio` | `0.01` | Minimum mask-to-image area ratio |
| `BorderConfig.maximum_ratio` | `0.05` | Maximum border-pixel-to-mask ratio |
| `BorderConfig.edge_maximum_ratio` | `0.25` | Maximum fraction of mask extent pinned to a frame edge |
| `OcclusionConfig.maximum_overlap` | `0.15` | Maximum hand-overlap-to-mask ratio |
| `CompletenessConfig.minimum_score` | `0.65` | Minimum completeness score |
| `ConfidenceConfig.minimum_confidence` | `0.5` | Minimum detection confidence |

---

## Filter Pipeline Order

Defined in `FilterConfig.filter_order`:

1. **VincentEmptyMask** — mask has no foreground pixels
2. **VincentBorderPixel** — mask touches the image frame
3. **BlurLaplacian** — boundary-band sharpness
4. **BlurTenengrad** — boundary-band gradient sharpness
5. **VincentsArtifacts** — mask artifact score

Filters early in the pipeline reject cheaply before more expensive checks. The population outlier statistics are fit once over the full dataset (`FilterPipeline.fit_observations`) before the main loop. Two Vincent soft filters remain available as population-adapted (0, 1] weights (`VincentsAreaFilter`, `VincentsMotionBlurFilter`) — these never reject and feed `vincents_area` / `vincents_motion_blur`.

---

## Threshold & Outlier Variants (`preprocessing/variants.py`)

Every pre-filter's `evaluate` returns a `(score, passed, reason)` triple with a 0..1 "goodness" score (higher = better). A uniform wrapper, `FilterVariant`, layers two optional rejection modes on top of any filter — and the same logic runs over the soft filters' population weights — without changing the base filter's own behavior:

| Knob | Mode | Behavior | Reject reason |
|------|------|----------|---------------|
| `threshold_min` | absolute floor | `score < threshold_min` → reject | `<reason>_threshold` |
| `outlier_z` | robust bad-outlier | population fit (median/MAD), `z <= -outlier_z` → reject | `<reason>_outlier` |

Both modes key off the same 0..1 score every filter already returns, so they are uniform across filters. When the inner filter rejects on its own, its reason is kept verbatim (no variant suffix). The population fit is skipped entirely when no filter sets `outlier_z`, so default pipeline behavior is unchanged.

### Default pre-filters (`FilterVariant` via `run.build_filters`)

The default blur/artifact filters set both knobs by default, e.g. `LaplacianBlurConfig(threshold_min=0.01, outlier_z=3.0)`. Setting `outlier_z` makes the pipeline flag `requires_fit` and run one population pass (`FilterPipeline.fit_observations`) over all observations before the main loop; the median/MAD robust stats are then used for the z test. This requires the whole pre-filter pass to be two-phase.

### Soft filters (`reject_soft_variants` via `run.py`)

The Vincent soft filters never hard-reject during `evaluate`; they derive a population weight in `(0, 1]`. When `VincentsAreaConfig` or `VincentsMotionBlurConfig` set `threshold_min` or `outlier_z`, the pipeline moves accepted observations whose weight trips the cutoff into `rejected` right after the soft pass, with annotated reasons (`vincents_area_threshold`, `vincents_motion_blur_outlier`, …) so the per-reason sample folders group them cleanly.

---

## Overriding

### Per-run (CLI)
```bash
# Disable auto-tuning entirely, use static config defaults
python run.py --data_root ... --no-auto-thresholds

# Or edit config.py directly
```

### Permanently (config.py)
```python
auto_thresholds = False
filters.blur_laplacian.threshold_min = 0.02
filters.blur_laplacian.outlier_z = 3.5
```

### Custom percentiles (programmatic)
```python
from utils.threshold_tuner import tune_thresholds
thresholds = tune_thresholds(
    observations,
    percentiles=dict(
        border_ratio=99,    # less aggressive
        laplacian=10,       # boundary-band sharpness (legacy key)
    )
)
```

---

## Quality Weights (post-filter scoring)

Not thresholds but affect final ranking. Defined in `QualityWeights` — exactly 4 components summing to 1:

| Component | Weight | Description |
|-----------|--------|-------------|
| `blur` | `0.30` | Boundary-band sharpness |
| `area` | `0.20` | Object size relative to image |
| `vincents_artefacts` | `0.20` | Mask artifact fraction |
| `centerness` | `0.30` | Mask centredness |

`confidence` is exported for diagnostics (`min(blur, area, vincents_artefacts, centerness)`) but is not a scorer component. The Vincent soft-filter softness values (`VincentsAreaConfig.softness`, `VincentsMotionBlurConfig.softness`) control the falloff steepness in robust-MADs: smaller softness → sharper discrimination around the population typical value.
