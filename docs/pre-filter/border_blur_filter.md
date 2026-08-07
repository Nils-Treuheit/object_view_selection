# BorderLaplacianBlurFilter & BorderTenengradBlurFilter (border_blur_filter)

**Source:** `preprocessing/border_blur_filter.py`<br>
**Kind:** hard filter — raw statistical boundary-s sharpenness pre-filter. Both filters implement
the two `BaseFilter` rejection criteria: a population-wide extreme-bad-outlier pass (`outlier_z`) and an absolute threshold-based garbage floor (`hard_min_`). <br> 
**Not part of the default `filter_order`.**

## Purpose

Measure boundary sharpness using two complementary methods — Laplacian variance 
(`BorderLaplacianBlurFilter`) and mean Tenengrad (`BorderTenengradBlurFilter`) — on
the band straddling the object contour. These detect different blur modes: 
Laplacian responds to overall boundary contrast while Tenengrad responds to 
structured gradients, so the two are complementary for detecting motion blur, 
defocus, and similar degradations.

## Algorithm

### BorderLaplacianBlurFilter

1. **Raw statistical Value** (per observation)<br>
   Reflects the <b>variance of the Laplacian</b> restricted to the boundary band (`compute_boundary_blur_variance`):

    ```python
    band = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
    laplacian_variance = Laplacian(gray, ksize=3)[band].var()
    ```

   Lower = blurrier boundary. Implemented in `compute_stat`.

2. **Goodness score**<br>
   The raw stat scaled against a fixed global anchor so it is comparable across 
   datasets:

    ```python
    score = min(stat / max_variance, 1.0)         # max_variance default 10000
    ```

3. **Threshold-based Filter** (absolute)<br>
   If `hard_min_variance > 0` and the statistical value is below it, the observation 
   is rejected outright with reason `blur_laplacian_threshold`. This catches frames whose 
   object boundary is smeared — e.g., values `< ~100` on typical datasets.

4. **Population-based Filter** (relative)<br>
   Robust median/MAD z-score of the raw stat, fit once over the population (`fit`, only 
   when `outlier_z` is set):

    ```python
    z = (stat - median) / robust_scale
    z <= -outlier_z        -> reject with reason blur_laplacian_outlier
    ```

   Boundary sharpness is a continuous spectrum, so the "low_bad" (blurred) tail is where 
   the noticeably-bad outliers live.

5. **Population pass** (selection weight)<br>
   Same robust median/MAD one-sided half-Gaussian fall-off as `VincentsAreaFilter`, penalising 
   the "bad" (low = blurred) side:

    ```python
    weight = exp(-0.5 * (z / softness)^2)
    z = (median - stat) / robust_scale
    ```

   Softness is `0.3` robust-MADs, matching `VincentsMotionBlurFilter`. 
   Implemented in `fit_weights` (inherited from `VincentSoftFilter`).


### BorderTenengradBlurFilter

1. **Raw statistical Value** (per observation)<br>
   Reflects the <b>mean Sobel magnitude</b> restricted to the boundary band (`compute_boundary_tenengrad`):

    ```python
    band = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
    tenengrad = mean(sqrt(Sobel_x^2 + Sobel_y^2))[band]
    ```

   Lower = blurrier boundary. Implemented in `compute_stat`.

2. **Goodness score**<br>
   The raw stat scaled against a fixed global anchor:

    ```python
    score = min(stat / max_tenengrad, 1.0)       # max_tenengrad default 100
    ```

3. **Threshold-based Filter** (absolute)<br>
   If `hard_min_tenengrad > 0` and the statistical value is below it, the observation 
   is rejected outright with reason `blur_tenengrad_threshold`. This catches frames whose 
   object boundary lacks structured gradients — e.g., values `< ~25` on typical datasets.

4. **Population-based Filter** (relative)<br>
   Same robust median/MAD z-score as Laplacian filter:

    ```python
    z = (stat - median) / robust_scale
    z <= -outlier_z        -> reject with reason blur_tenengrad_outlier
    ```

5. **Population pass** (selection weight)<br>
   Identical half-Gaussian fall-off as `BorderLaplacianBlurFilter` above (`softness=0.3`). 
   Implemented in `fit_weights`.


## Metrics

### BorderLaplacianBlurFilter

| Field | Meaning |
|-------|---------|
| `metrics.laplacian` | raw stat: boundary-band Laplacian variance |
| `metrics.vincent_boundary_blur_variance` | alias to raw stat |

### BorderTenengradBlurFilter

| Field | Meaning |
|-------|---------|
| `metrics.tenengrad` | raw stat: boundary-band mean Sobel magnitude |
| `metrics.bound_tenengrad` | alias to raw stat |


## Score and rejection

### BorderLaplacianBlurFilter

`evaluate` returns `(score, passed, reason)` with `score = quality-scaled Laplacian variance` 
and implements both rejection criteria from `BaseFilter`:

- **Threshold-based — Absolute Garbage Rejection**<br>
    ```
    stat < hard_min_variance  →  (0.0, False, "blur_laplacian_threshold")
    ```
  A smeared boundary below the floor is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection**<br>
    ```
    z <= -outlier_z  →  (score, False, "blur_laplacian_outlier")
    ```
  Requires the `fit(observations)` population pass; robust median/MAD fitted over raw stat distribution.

- **Pass**<br>
    ```
    (score, True, "blur_laplacian")
    ```

- **Disabled**<br>
    ```
    (-1.0, True, "")
    ```

### BorderTenengradBlurFilter

Same structure as `BorderLaplacianBlurFilter` with `tenengrad` instead of `laplacian`:

- **Threshold-based — Absolute Garbage Rejection**<br>
    ```
    stat < hard_min_tenengrad  →  (0.0, False, "blur_tenengrad_threshold")
    ```

- **Population-based — Relative Outlier Rejection**<br>
    ```
    z <= -outlier_z  →  (score, False, "blur_tenengrad_outlier")
    ```

- **Pass**<br>
    ```
    (score, True, "blur_tenengrad")
    ```


## Setup

### Import

```python
from preprocessing.border_blur_filter import (
    BorderLaplacianBlurFilter,
    BorderTenengradBlurFilter,
)
```

### Initialisation

Both filters are instantiated with keyword arguments that default to the class-level constants 
(`BORDER_BLUR_STROKE_WIDTH=9`, `MAX_VARIANCE=10000.0` / `MAX_TENEGRAD=100.0`, etc.):

```python
laplacian_filter = BorderLaplacianBlurFilter(
    stroke_width=9,              # boundary-band width (px)
    max_variance=10000.0,        # score scale anchor
    hard_min_variance=100.0,     # absolute garbage floor
    outlier_z=None,              # disable outlier mode
    enabled=True,
)

tenengrad_filter = BorderTenengradBlurFilter(
    stroke_width=9,
    max_tenengrad=100.0,
    hard_min_tenengrad=25.0,     # absolute garbage floor
    outlier_z=None,              # disable outlier mode
    enabled=True,
)
```

To enable the population-based outlier pass, first instantiate without `outlier_z`, then 
set it and call `fit(observations)` before `evaluate`:

```python
laplacian_filter.outlier_z = 4.5                    # set cutoff
laplacian_filter.fit(all_observations)               # fit median/MAD on population
score, passed, reason = laplacian_filter.evaluate(obs)  # then evaluate each obs
```

### Workflow

Both filters follow the `BaseFilter` lifecycle:

1. **Create** — instantiate with desired defaults (see above).
2. **Fit population statistics** (`fit(observations)`) — only needed when `outlier_z` is 
   set; computes robust median/MAD of the raw stat over the full observation set.
3. **Evaluate per observation** (`evaluate(observation)`) — returns `(score, passed, reason)` 
   with score in `[0, 1]`, pass/fail flag, and reason string for diagnostics.
4. **Access weights / stats** post-evaluation on `observation.metrics`:

   ```python
   obs.metrics.laplacian          # raw Laplacian variance
   obs.metrics.vincent_boundary_blur_variance  # alias (same value)
   obs.metrics.tenengrad           # raw mean Sobel magnitude
   obs.metrics.bound_tenengrad     # alias (same value)
   ```

5. **Place in `filter_order`** — if adding to the default pipeline, position these filters 
   alongside other quality pre-filters in the hard-filter stage before selection weighting.

## Configuration (`BorderLaplacianBlurConfig` / `BorderTenengradBlurConfig`)

### BorderLaplacianBlurFilter

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and score |
| `softness` | `0.3` | Falloff in robust-MADs (weight pass) |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_variance` | `100.0` | Absolute hard-reject floor on raw Laplacian variance (`0` disables) |
| `max_variance` | `10000.0` | Scale anchor for the goodness score |
| `outlier_z` | `None` | Robust outlier cutoff on raw stat (`fit`/`evaluate`) and weight pass |
| `threshold_min` | `None` | Optional floor on the fitted weight (complementary layer) |

### BorderTenengradBlurFilter

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and score |
| `softness` | `0.3` | Falloff in robust-MADs (weight pass) |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_tenengrad` | `25.0` | Absolute hard-reject floor on raw Tenengrad (`0` disables) |
| `max_tenengrad` | `100.0` | Scale anchor for the goodness score |
| `outlier_z` | `None` | Robust outlier cutoff on raw stat (`fit`/`evaluate`) and weight pass |
| `threshold_min` | `None` | Optional floor on the fitted weight (complementary layer) |


## Notes

- Both filters compute complementary blur measures: Laplacian variance captures overall 
  boundary contrast variation while Tenengrad responds to structured gradients.
- The raw stat is always recorded even when a rejection criterion trips, so downstream 
  diagnostics and the weight pass see it.
- If `observation.image is None`, the stat is `0.0`; with default thresholds this would 
  be caught by the absolute garbage floor in Laplacian mode and the Tenengrad mode also.
- Both filters store their raw stat on dual attributes: the primary metric name and an 
  alias consistent with existing Vincent filter conventions.
