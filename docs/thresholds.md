# Thresholds & Pre-Filtering

## Overview

**`config.py` is the ground truth for every pre-filter threshold.** The auto-tuner only adjusts those config values on top when it is active (the default, or `--auto-thresholds`): it computes percentiles from the dataset itself, then clamps the result within sensible safety bounds so even a garbage dataset enforces a minimum quality bar.

Set `auto_thresholds: false` in `config.py` or pass `--no-auto-thresholds` to skip the auto-tuner entirely — then the config values are used **exactly as written**.

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
| `LaplacianBlurConfig.max_variance` | `20000.0` | Score anchor for the Laplacian variance |
| `LaplacianBlurConfig.hard_min_variance` | `4000.0` | Absolute garbage floor on the raw stat |
| `LaplacianBlurConfig.outlier_z` | `3.0` | Robust population-outlier z |
| `TenengradBlurConfig.stroke_width` | `9` | Boundary-band stroke width |
| `TenengradBlurConfig.max_tenengrad` | `150.0` | Score anchor for the Tenengrad magnitude |
| `TenengradBlurConfig.hard_min_tenengrad` | `33.0` | Absolute garbage floor on the raw stat |
| `TenengradBlurConfig.outlier_z` | `3.0` | Robust population-outlier z |
| `VincentsArtifactsConfig.kernel_size` | `3` | Morphology kernel for artifact detection |
| `VincentsArtifactsConfig.max_fraction` | `0.05` | Artifact fraction at which the score hits 0 |
| `VincentsArtifactsConfig.hard_max_fraction` | `0.15` | Absolute garbage ceiling on the raw stat |
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

Filters early in the pipeline reject cheaply before more expensive checks. The population outlier statistics are fit once over the full dataset (`FilterPipeline.fit_observations`) before the main loop. Two Vincent soft filters remain available as population-adapted (0, 1] weights feeding `vincents_area` / `vincents_motion_blur` — `VincentsAreaFilter` never rejects, while `VincentsMotionBlurFilter` also implements both rejection criteria itself (see below and `docs/pre-filter/vincents_motion_blur.md`).

---

## Threshold & Outlier Rejection (`preprocessing/base.py`)

Every non-binary pre-filter's `evaluate` returns a `(score, passed, reason)`
triple with a 0..1 "goodness" score (higher = better). The shared `ScoreFilter`
base implements both `BaseFilter` rejection criteria **on the raw stat**,
once, so subclasses only supply `compute_stat` / `compute_score`:

| Knob | Mode | Behavior | Reject reason |
|------|------|----------|---------------|
| `hard_min` / `hard_max` | absolute garbage threshold | raw `stat` outside the bound → reject regardless of the population | `<reason>_threshold` |
| `outlier_z` | robust bad-outlier | population fit (median/MAD), `z` beyond `outlier_z` on the `direction` tail → reject | `<reason>_outlier` |

The population fit is skipped entirely when no filter sets `outlier_z`
(`need_fitting()` returns `True` only then), so default pipeline behavior is
unchanged. The pipeline flags the two-phase pass via
`FilterPipeline.need_fitting` / `fit_observations`.

### Default pre-filters (`ScoreFilter` via `run.build_filters`)

The default blur/artifact filters implement both criteria themselves, e.g.
`LaplacianBlurConfig(hard_min_variance=4000.0, outlier_z=3.0)`. Setting
`outlier_z` makes the filter report `need_fitting() == True`; the pipeline runs
one population pass (`FilterPipeline.fit_observations`) over all observations
before the main loop, and the robust stats are then used for the z test.

### Legacy filters (`OutlierFilter` via `run.build_filters`)

Legacy filters (`AreaFilter`, `BorderFilter`, …) implement only their own
absolute cutoff. When their config sets `outlier_z`, `run.py` wraps them in
`OutlierFilter` (`preprocessing/variants.py`), which layers the population
extreme-bad-outlier rejection on top without touching the inner filter's logic.

### Soft filters (`apply_soft_filters` via `run.py`)

`VincentsAreaFilter` and `VincentsMotionBlurFilter` also implement both
rejection criteria themselves via `ScoreFilter` — the absolute
`hard_min_area_fraction` / `hard_min_variance` floor on the raw stat and an
`outlier_z` population removal — so they can act as working pre-filters
(`apply_soft_filters` runs their `fit` before the per-observation pass and
moves rejected observations out of `accepted` with the annotated reason).

---

## Overriding

### Per-run (CLI)
```bash
# Disable auto-tuning entirely, use static config defaults
python run.py --data_root ... --no-auto-thresholds

# Run ONLY the named pre-filters (including soft ones) in the given order
python run.py --data_root ... --filter_order blur_laplacian,vincents_area --no-auto-thresholds

# Or edit config.py directly
```

When `--filter_order` is given, **only** the named pre-filters execute —
including the soft `vincents_area` / `vincents_motion_blur` filters, which are
otherwise always run as diagnostics. So `--filter_order vincents_area` runs just
`VincentsAreaFilter`, and `--filter_order blur_laplacian` skips both soft
filters entirely. Omitting `--filter_order` keeps the default behavior: hard
filters in the configured order plus both soft filters as diagnostics.

All thresholds — including custom garbage floors and outlier z-cutoffs — are set
in `config.py`, never on the command line. With `--no-auto-thresholds` the
static config values are used as-is, so a manual pipeline is fully described by
`--filter_order` + the config file.

### Permanently (config.py)
```python
auto_thresholds = False
filters.blur_laplacian.hard_min_variance = 5000.0
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
