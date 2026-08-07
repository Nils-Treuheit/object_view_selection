# VincentsMotionBlurFilter (`vincents_motion_blur`)

**Source:** `preprocessing/vincents_motion_blur.py`<br>
**Ported from:** `score_motion_blur` in [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** soft filter — by default never rejects (derives a population-adapted
selection weight in `(0, 1]`), but with an optional absolute hard-reject floor
on the raw stat.<br> 
**Not** part of the default `filter_order` -> used as a diagnostic and, optionally, as a rejection layer.

## Purpose

Detects motion-blurred object boundaries — the same boundary-band statistic as
`BorderLaplacianBlurFilter`, but treated as a continuous spectrum that is
down-ranked rather than hard-rejected.

## Algorithm

1. **Raw statistical Value** (per observation)<br> 
   It reflects the <b>variance of the Laplacian</b> restricted to the boundary band (`compute_boundary_blur_variance`):

   ```python
   band = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
   vincent_boundary_blur_variance = Laplacian(gray, ksize=3)[band].var()
   ```
   This is implemented in function `evaluate`.

2. **Hard floor Threshold** (optional)<br> 
   If `hard_min_variance > 0` and the statistical Value is below
   it, the observation is rejected outright with reason `motion_blur`.

3. **Population pass** <br>
   Same robust median/MAD one-sided half-Gaussian fall-off as `VincentsAreaFilter`, penalising the "bad" (low = blurred) side:

   ```python
   weight = exp(-0.5 * (z / softness)^2)
   z = (median - stat) / robust_scale
   ```

   Boundary sharpness is a continuous spectrum, so softness is deliberately
   small (`0.3` robust-MADs) to discriminate.<br>
   This is implemented in function `fit_weights`.

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_boundary_blur_variance` | raw stat: boundary-band Laplacian variance |
| `metrics.vincents_motion_blur` | fitted weight in `(0, 1]` |

## Score and rejection

Currently without a hard floor the filter only returns `(1.0, True, "")`, but two rejection paths exist:
- **Absolute - Threshold-based Garbage Rejection** <br>
  ```
  stat < hard_min_variance  →  (0.0, False, "motion_blur")
  ```
  This catches frames whose object boundary is smeared by motion blur — e.g.
  values `< ~150` on a 480×640 moving-object dataset vs. `> ~1400` min on a
  static bottle set — that the soft weight would otherwise merely down-rank.<br>

- **Relative - Population-based Outlier Rejection** <br>
  `threshold_min / outlier_z` on the fitted weight via `reject_soft_variants`:
  ```
  vincents_motion_blur_threshold / vincents_motion_blur_outlier
  ```

## Configuration (`VincentsMotionBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and weight |
| `softness` | `0.3` | Falloff in robust-MADs |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_variance` | `120.0` | Absolute hard-reject floor on the raw variance (`0` disables) |
| `threshold_min` | `None` | Optional floor on the fitted weight |
| `outlier_z` | `None` | Optional robust outlier cutoff on the weight |

## Notes

- The raw stat is always recorded even when the hard floor rejects, so
  downstream diagnostics and the weight pass see it.
- If `observation.image is None`, the stat is `0.0` (and a configured hard
  floor would reject it).
- `VincentsMotionBlurQuality` scores the same stat against a fixed global
  anchor (`blur_max_variance = 10000`).
