# VincentsMotionBlurFilter (`vincents_motion_blur`)

**Source:** `preprocessing/vincents_motion_blur.py`<br>
**Ported from:** `score_motion_blur` in [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** soft filter — derives a population-adapted selection weight in
`(0, 1]`, and simultaneously acts as a working pre-filter with the two
`BaseFilter` rejection criteria implemented in `evaluate`.<br> 
**Not** part of the default `filter_order` -> used as a diagnostic and, optionally, as a rejection layer.

## Purpose

Detects motion-blurred object boundaries — the same boundary-band statistic as
`BorderLaplacianBlurFilter`, but treated as a continuous spectrum that is
down-ranked rather than hard-rejected. Frames whose object boundary is smeared
are dropped outright (absolute floor + population outlier), the rest are
ranked by the derived selection weight.

## Algorithm

1. **Raw statistical Value** (per observation)<br> 
   It reflects the <b>variance of the Laplacian</b> restricted to the boundary band (`compute_boundary_blur_variance`):

   ```python
   band = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
   vincent_boundary_blur_variance = Laplacian(gray, ksize=3)[band].var()
   ```

   This is implemented in function `compute_stat`.

2. **Quality scaled stat**<br> 
   The reported score is the raw stat scaled against a fixed global anchor so it
   is comparable across datasets:

   ```python
   score = min(stat / max_variance, 1.0)       # max_variance default 10000
   ```

3. **Threshold-based Filter** (absolute)<br> 
   If `hard_min_variance > 0` and the statistical value is below it, the
   observation is rejected outright with reason `vincents_motion_blur_threshold`.
   This catches frames whose object boundary is smeared by motion blur — e.g.
   values `< ~150` on a 480×640 moving-object dataset vs. `> ~1400` min on a
   static bottle set.

4. **Population-based Filter** (relative)<br> 
   Robust median/MAD z-score of the raw stat, fit once over the population
   (`fit`, only when `outlier_z` is set):

   ```python
   z = (stat - median) / robust_scale
   z <= -outlier_z        -> reject with reason vincents_motion_blur_outlier
   ```

   Boundary sharpness is a continuous spectrum, so the "low_bad" (blurred) tail
   is where the noticeably-bad outliers live.

5. **Population pass** (selection weight)<br> 
   Same robust median/MAD one-sided half-Gaussian fall-off as `VincentsAreaFilter`, penalising the "bad" (low = blurred) side:

   ```python
   weight = exp(-0.5 * (z / softness)^2)
   z = (median - stat) / robust_scale
   ```

   Softness is deliberately small (`0.3` robust-MADs) to discriminate.<br>
   This is implemented in function `fit_weights` (inherited from `VincentSoftFilter`).

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_boundary_blur_variance` | raw stat: boundary-band Laplacian variance |
| `metrics.vincents_motion_blur` | fitted weight in `(0, 1]` |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = quality scaled stat`
and implements the two rejection criteria from `BaseFilter`:

- **Threshold-based — Absolute Garbage Rejection** <br>
  ```
  stat < hard_min_variance  →  (0.0, False, "vincents_motion_blur_threshold")
  ```
  A smeared boundary below the floor is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection** <br>
  ```
  z <= -outlier_z  →  (score, False, "vincents_motion_blur_outlier")
  ```
  Requires the `fit(observations)` population pass (run automatically by
  `apply_soft_filters` when `need_fitting()` is true); the robust median/MAD
  are taken from the raw stat distribution.

- **Pass** <br>
  ```
  (score, True, "vincents_motion_blur")
  ```

Both rejection criteria are the shared `ScoreFilter` implementation (see
`preprocessing/base.py` + `preprocessing/filter_utils.py`).

## Configuration (`VincentsMotionBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and weight |
| `softness` | `0.3` | Falloff in robust-MADs |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_variance` | `120.0` | Absolute hard-reject floor on the raw variance (`0` disables) |
| `outlier_z` | `None` | Robust population-outlier cutoff on the raw stat (`fit`/`evaluate`) |

`hard_min_variance` is forwarded from config by `build_soft_filters`, so the
configured value is active in the pipeline.

## Notes

- The raw stat is always recorded even when a rejection criterion trips, so
  downstream diagnostics and the weight pass see it.
- If `observation.image is None`, the stat is `0.0` (a configured hard floor
  would reject it).
- `VincentsMotionBlurQuality` scores the same stat against the same fixed
  global anchor (`blur_max_variance = 10000`).
