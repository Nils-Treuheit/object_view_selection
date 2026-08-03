# Thresholds & Pre-Filtering

## Overview

The pipeline uses a mix of **data-driven auto-tuning** and **hard safety limits** for all pre-filter thresholds. Auto-tuning computes percentiles from the dataset itself, then clamps the result within sensible safety bounds so even a garbage dataset enforces a minimum quality bar.

Set `auto_thresholds: false` in `config.py` or pass `--no-auto-thresholds` to fall back entirely to the static config values.

---

## Auto-Tuning Strategy (`utils/threshold_tuner.py`)

| Filter | Metric | Percentile | Rationale |
|--------|--------|-----------|-----------|
| **Area** | `area_ratio` | **1st** (low end) | The smallest 1% of objects are rejected. Assumes >=99% of the dataset has usable object size. |
| **Border** | `border_ratio` | **95th** (high end) | The 5% of observations with the most border touching are rejected. Border touching is almost always bad, so this is quite aggressive. |
| **Border** | `edge_ratio` | **95th** (high end) | The 5% of observations with the most edge contact (object pinned to the frame) are rejected. Catches objects mostly cut off along an edge. |
| **Blur** | `laplacian` | **5th** (low end) | The blurriest 5% are rejected. The sharper 95% are considered acceptable. |
| **Blur** | `tenengrad` | **5th** (low end) | Same logic as laplacian — blurriest 5% rejected. |
| **Occlusion** | `hand_overlap` | **95th** (high end) | The 5% most occluded observations are rejected. |
| **Completeness** | `completeness` | **1st** (low end) | The 1% least-complete objects are rejected. Very permissive — only extreme cases. |

### How it works

1. Run every filter in eval-only mode on every observation to collect metric values.
2. Compute the specified percentile across the dataset.
3. Clamp the result to `[safety_min, safety_max]` (see below).
4. Use the clamped value as the filter threshold.

This means: if the dataset is already clean (e.g. no border-touching images), the 95th percentile of `border_ratio` is 0.0, so the threshold is clamped to `safety_min=0.001` — still enforcing a basic check. Conversely, if the entire dataset is blurry, the 5th percentile of laplacian might be 5.0, and it gets clamped up to `safety_min=30.0`.

---

## Safety Limits (Hard Fallbacks)

These are the absolute bounds enforced regardless of dataset statistics. Defined in `SAFETY_LIMITS` in `utils/threshold_tuner.py`.

| Threshold Key | Min | Max | Rationale |
|--------------|-----|-----|-----------|
| `area_minimum_ratio` | `0.01` (1%) | `0.05` (5%) | At least 1% of image must be object; never require more than 5% |
| `border_maximum_ratio` | `0.001` (0.1%) | `0.05` (5%) | At most 0.1% of mask pixels may border the edge; never allow more than 5% |
| `border_edge_maximum_ratio` | `0.05` (5%) | `0.5` (50%) | At most 5% of the mask extent may be pinned to a frame edge; never allow more than 50% |
| `laplacian_threshold` | `30.0` | `200.0` | Minimum sharpness floor; never demand more than 200 variance |
| `tenengrad_threshold` | `10.0` | `60.0` | Minimum gradient floor; never demand more than 60 |
| `occlusion_maximum_overlap` | `0.001` (0.1%) | `0.30` (30%) | At most 0.1% overlap allowed at minimum; never allow more than 30% |
| `completeness_minimum_score` | `0.50` | `0.80` | At least 0.5 completeness score; never demand more than 0.8 |

---

## Static Defaults (`config.py`)

Used when `auto_thresholds: false` or when a tuned value is not provided.

| Config Field | Default | Description |
|-------------|---------|-------------|
| `BlurConfig.threshold` | `120.0` | Laplacian variance threshold |
| `BlurConfig.tenengrad_threshold` | `35.0` | Tenengrad gradient threshold |
| `AreaConfig.minimum_ratio` | `0.01` | Minimum mask-to-image area ratio |
| `BorderConfig.maximum_ratio` | `0.05` | Maximum border-pixel-to-mask ratio |
| `BorderConfig.edge_maximum_ratio` | `0.25` | Maximum fraction of mask extent pinned to a frame edge |
| `OcclusionConfig.maximum_overlap` | `0.15` | Maximum hand-overlap-to-mask ratio |
| `CompletenessConfig.minimum_score` | `0.65` | Minimum completeness score |
| `ConfidenceConfig.minimum_confidence` | `0.5` | Minimum detection confidence |

---

## Filter Pipeline Order

Defined in `FilterConfig.filter_order` (hard filters):

1. **VincentEmptyMask** — mask has no foreground pixels
2. **VincentBorderPixel** — mask touches the image frame
3. **Border** — truncation by image edge
4. **Area** — object too small
5. **Confidence** — detection confidence too low (disabled by default)
6. **Blur** — image too blurry
7. **Occlusion** — hand/object overlap too high
8. **Completeness** — object mask incomplete

Filters early in the pipeline reject cheaply before more expensive checks. After the hard pass, the population-adapted **soft pre-filter pass** (`VincentsAreaFilter`, `VincentsArtifactsFilter`, `VincentsMotionBlurFilter`) computes (0, 1] weights from robust median/MAD stats — these never reject, they feed quality scoring.

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
filters.border.maximum_ratio = 0.01
filters.blur.threshold = 50.0
```

### Custom percentiles (programmatic)
```python
from utils.threshold_tuner import tune_thresholds
thresholds = tune_thresholds(
    observations,
    percentiles=dict(
        border_ratio=99,    # less aggressive
        laplacian=10,       # reject blurrier 10% instead of 5%
    )
)
```

---

## Quality Weights (post-filter scoring)

Not thresholds but affect final ranking. Defined in `QualityWeights`:

| Component | Weight | Description |
|-----------|--------|-------------|
| `completeness` | `0.35` | Mask completeness |
| `blur` | `0.20` | Image sharpness |
| `occlusion` | `0.20` | Freedom from hand occlusion |
| `area` | `0.15` | Object size relative to image |
| `confidence` | `0.10` | Detection confidence |
| `vincents_area` | `0.10` | Population-adapted mask area weight |
| `vincents_artefacts` | `0.10` | Population-adapted mask artifact weight |
| `vincents_motion_blur` | `0.10` | Population-adapted boundary blur weight |

The Vincent soft-filter softness values (`VincentsAreaConfig.softness`, `VincentsArtifactsConfig.softness`, `VincentsMotionBlurConfig.softness`) control the falloff steepness in robust-MADs: smaller softness → sharper discrimination around the population typical value.
