# BorderLaplacianBlurFilter (`blur_laplacian`)

**Source:** `preprocessing/border_blur_filter.py`<br>
**Ported from:** [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** score filter — implements both `BaseFilter` rejection criteria
itself via the shared `ScoreFilter` base.<br>
**(Default)** part of `FilterConfig.filter_order`.

## Purpose

Measures the sharpness of the **object boundary** via variance of the Laplacian
restricted to a boundary band (`dilate XOR erode` of the mask). The same
boundary-band statistic is used by `VincentsMotionBlurFilter` (as a selection
weight) — both share `metrics.vincent_boundary_blur_variance`.

## Algorithm

1. **Raw statistical value (per observation)**<br>
   Boundary-band Laplacian variance, computed via `compute_boundary_blur_variance`:

   ```python
   band       = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
   laplacian  = Laplacian(gray, ksize=3)[band].var()
   ```

   Implemented in `compute_stat` (same helper used by
   `VincentsMotionBlurFilter`).

2. **Quality-scaled score**<br>
   The raw stat divided against a fixed global anchor so it is comparable
   across datasets:

   ```python
   lap_score = min(laplacian / max_variance, 1.0)     # max_variance default 20000
   ```

3. **Threshold-based filter (absolute garbage rejection)**<br>
   If the Laplacian stat falls below `hard_min_variance`, the frame is
   rejected outright with reason `blur_laplacian_threshold`. This catches
   frames whose object-boundary variance is unusably low regardless of the
   population:

   ```python
   laplacian < hard_min_variance  →  (0.0, False, "blur_laplacian_threshold")
   ```

4. **Population-based filter (relative outlier rejection)**<br>
   Robust median / MAD z-score of the Laplacian stat, fit once over the
   population (`fit`, only when `outlier_z` is set):

   ```python
   z = (laplacian - median) / robust_scale
   z <= -outlier_z  →  (score, False, "blur_laplacian_outlier")
   ```

   Boundary sharpness is a continuous spectrum, so the "low_bad" (blurred)
   tail is where the noticeably-bad outliers live.

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.laplacian` | raw stat: boundary-band Laplacian variance |
| `metrics.vincent_boundary_blur_variance` | same raw stat, shared with `VincentsMotionBlurFilter` |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = min(stat/max_variance, 1)`:

- **Threshold-based — Absolute Garbage Rejection** — `stat < hard_min_variance` → `(0.0, False, "blur_laplacian_threshold")`.
- **Population-based — Relative Outlier Rejection** — `z <= -outlier_z` → `(score, False, "blur_laplacian_outlier")`.
- **Pass** — `(score, True, "blur_laplacian")`.

Both rejection criteria are the shared `ScoreFilter` implementation (see
`preprocessing/base.py` + `preprocessing/filter_utils.py`).

## Configuration (`LaplacianBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_variance` | `4000.0` | Absolute hard-reject floor on raw Laplacian variance |
| `max_variance` | `20000.0` | Scale anchor for the goodness score |
| `outlier_z` | `3.0` | Robust population-outlier cutoff on raw stat (`fit`/`evaluate`) |

All knobs are forwarded from config by `build_filters`.

## Notes

- The Laplacian raw stat is always published, even when it trips rejection, so
  downstream diagnostics and scoring see it.
- If `observation.image is None`, the stat is `0.0` (a configured hard floor
  would reject it).
- `BorderBlurQuality` scores the same stat against the same fixed global anchor
  (`blur_max_variance = 10000`).
- See also the complementary [blur_tenengrad.md](blur_tenengrad.md) filter and
  the combined [border_blur_filter.md](border_blur_filter.md) reference.
