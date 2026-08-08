# BorderTenengradBlurFilter (`blur_tenengrad`)

**Source:** `preprocessing/border_blur_filter.py`<br>
**Ported from:** [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** score filter — implements both `BaseFilter` rejection criteria
itself via the shared `ScoreFilter` base.<br>
**(Default)** part of `FilterConfig.filter_order`.

## Purpose

Complementary boundary-band sharpness measure to
[`BorderLaplacianBlurFilter`](blur_laplacian.md). The Laplacian variance
measures overall boundary sharpness; the Tenengrad (mean Sobel magnitude)
responds to *structured gradients*, so the two detect complementary blur
modes.

## Algorithm

1. **Raw statistical value (per observation)**<br>
   Boundary-band mean Sobel magnitude, computed via `compute_boundary_tenengrad`:

   ```python
   band       = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
   gradient   = sqrt(SobelX(gray)^2 + SobelY(gray)^2)
   tenengrad  = gradient[band].mean()
   ```

   Implemented in `compute_stat`.

2. **Quality-scaled score**<br>
   The raw stat divided against a fixed global anchor so it is comparable
   across datasets:

   ```python
   ten_score = min(tenengrad / max_tenengrad, 1.0)    # max_tenengrad default 150
   ```

3. **Threshold-based filter (absolute garbage rejection)**<br>
   If the Tenengrad stat falls below `hard_min_tenengrad`, the frame is
   rejected outright with reason `blur_tenengrad_threshold`:

   ```python
   tenengrad < hard_min_tenengrad  →  (0.0, False, "blur_tenengrad_threshold")
   ```

4. **Population-based filter (relative outlier rejection)**<br>
   Robust median / MAD z-score of the Tenengrad stat, fit once over the
   population (`fit`, only when `outlier_z` is set):

   ```python
   z = (tenengrad - median) / robust_scale
   z <= -outlier_z  →  (score, False, "blur_tenengrad_outlier")
   ```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.tenengrad` | raw stat: mean boundary-band Sobel magnitude |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = min(stat/max_tenengrad, 1)`:

- **Threshold-based — Absolute Garbage Rejection** — `stat < hard_min_tenengrad` → `(0.0, False, "blur_tenengrad_threshold")`.
- **Population-based — Relative Outlier Rejection** — `z <= -outlier_z` → `(score, False, "blur_tenengrad_outlier")`.
- **Pass** — `(score, True, "blur_tenengrad")`.

Both rejection criteria are the shared `ScoreFilter` implementation (see
`preprocessing/base.py` + `preprocessing/filter_utils.py`).

## Configuration (`TenengradBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `hard_min_tenengrad` | `33.0` | Absolute hard-reject floor on raw Tenengrad |
| `max_tenengrad` | `150.0` | Scale anchor for the goodness score |
| `outlier_z` | `3.0` | Robust population-outlier cutoff on raw stat (`fit`/`evaluate`) |

All knobs are forwarded from config by `build_filters`.

## Notes

- The Tenengrad raw stat is always published, even when it trips rejection, so
  downstream diagnostics and scoring see it.
- If `observation.image is None`, the stat is `0.0` (a configured hard floor
  would reject it).
- See also the combined [border_blur_filter.md](border_blur_filter.md)
  reference.
