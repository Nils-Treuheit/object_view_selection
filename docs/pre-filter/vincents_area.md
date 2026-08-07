# VincentsAreaFilter (`vincents_area`)

**Source:** `preprocessing/vincents_area_filter.py`<br>
**Ported from:** `score_mask_area` in [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** soft filter — derives a population-adapted selection weight in
`(0, 1]`, and simultaneously acts as a working pre-filter with the two
`BaseFilter` rejection criteria implemented in `evaluate`.<br> 
**Not** part of the default `filter_order` -> used as a diagnostic and, optionally, as a rejection layer.

## Purpose

Penalises objects that occupy a tiny fraction of the frame — small masks are
harder to recognize and often mean the object is far away or poorly framed.
Frames whose mask area falls below an absolute garbage floor or outlying low
tail are dropped outright; the rest are ranked by the derived selection weight.

## Algorithm

1. **Raw statistical Value** (per observation)<br> 
   The **mask area fraction**:

   ```python
   vincent_area_fraction = mask_pixels / canvas_area
   ```

   This is implemented in function `compute_stat`.

2. **Quality scaled stat**<br> 
   The reported score is the raw stat scaled against a fixed global anchor so
   it is comparable across datasets:

   ```python
   score = min(stat / max_fraction, 1.0)       # max_fraction default 0.20
   ```

3. **Threshold-based Filter** (absolute)<br> 
   If `hard_min_area_fraction > 0` and the statistical value is below it, the
   observation is rejected outright with reason `vincents_area_threshold`.
   This catches frames whose object mask occupies a near-zero fraction of the
   canvas — e.g. masks with `area_fraction < ~0.001` on typical datasets vs.
   `> ~0.05` min on larger-object sets.

4. **Population-based Filter** (relative)<br> 
   Robust median/MAD z-score of the raw stat, fit once over the population
   (`fit`, only when `outlier_z` is set):

   ```python
   z = (stat - median) / robust_scale
   z <= -outlier_z        -> reject with reason vincents_area_outlier
   ```

   Mask area is a continuous spectrum, so the "low_bad" (tiny mask) tail
   is where the noticeably-bad outliers live.

5. **Population pass** (selection weight)<br> 
   Same robust median/MAD one-sided half-Gaussian fall-off as `VincentsMotionBlurFilter`, penalising the "bad" (low) side:

   ```python
   weight = exp(-0.5 * (z / softness)^2)
   z = (median - stat) / robust_scale
   ```

   Softness is deliberately small (`0.3` robust-MADs) to discriminate.<br>
   This is implemented in function `fit_weights` (inherited from `VincentSoftFilter`).

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_area_fraction` | raw stat: mask fraction (pixels / canvas) |
| `metrics.vincents_area` | fitted weight in `(0, 1]` |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = quality scaled stat`
and implements the two rejection criteria from `BaseFilter`:

- **Threshold-based — Absolute Garbage Rejection** <br>
  ```
  stat < hard_min_area_fraction  →  (0.0, False, "vincents_area_threshold")
  ```
  A mask below the floor is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection** <br>
  ```
  z <= -outlier_z  →  (score, False, "vincents_area_outlier")
  ```
  Requires the `fit(observations)` population pass (run automatically by
  `apply_soft_filters` when `requires_fit()` is true); the robust median/MAD
  are taken from the raw stat distribution.

- **Pass** <br>
  ```
  (score, True, "vincents_area")
  ```

The fit weight path (`threshold_min` / `outlier_z` on the `(0, 1]` weight via
`reject_soft_variants`) remains available as a complementary layer and reuses
the same `vincents_area_threshold` / `vincents_area_outlier` reasons.

## Configuration (`VincentsAreaConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and weight |
| `softness` | `0.3` | Falloff in robust-MADs |
| `hard_min_area_fraction` | `0.0` | Absolute hard-reject floor on the raw fraction (`0` disables) |
| `threshold_min` | `None` | Optional floor on the fitted weight (`reject_soft_variants` layer) |
| `outlier_z` | `None` | Robust outlier cutoff — on the raw stat (`fit`/`evaluate`) and on the weight (`reject_soft_variants`) |

`hard_min_area_fraction` is forwarded from config by `build_soft_filters`, so 
the configured value is active in the pipeline.

## Notes

- The raw stat is always recorded even when a rejection criterion trips, so
  downstream diagnostics and the weight pass see it.
- If `canvas_area <= 0`, the stat is `0.0` (a configured hard floor would reject it).
- `VincentsAreaQuality` scores the same raw stat against the same fixed global
  anchor (`area_max_fraction = 0.20`).
