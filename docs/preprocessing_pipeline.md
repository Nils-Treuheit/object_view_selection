# Pre-Filter Reference

Detailed, per-filter documentation for the pre-filter stage. The pipeline
overview lives in [`../pipeline.md`](../pipeline.md); this directory covers
each filter's algorithm, metrics, configuration, and rejection behaviour in
isolation.

## Terminology

Every pre-filter's `evaluate(observation)` returns `(score, passed, reason)`:

- **Hard filter** — always returns `passed=False` with a bare reason (e.g.
  `vincent_empty_mask`) for the failing class. Score is 0/1.
- **Score filter** — *never* rejects by itself; returns a 0..1 "goodness"
  score and `passed=True`. Rejection is layered on top by `FilterVariant`
  (relaxed absolute floor `threshold_min` + population-relative extreme-bad
  outlier `outlier_z`).
- **Soft filter** — *never* rejects; computes a raw stat per observation, then
  a population pass converts it into a selection weight in `(0, 1]` using
  robust median/MAD statistics. Only used as diagnostics unless rejection
  knobs are added on the fitted weight.

## Default set (5 filters, in `FilterConfig.filter_order`)

| # | Filter | Kind | Rejection reason(s) | Doc |
|---|--------|------|--------------------|-----|
| 1 | `VincentEmptyMaskFilter` | hard | `vincent_empty_mask` | [vincent_empty_mask.md](vincent_empty_mask.md) |
| 2 | `VincentBorderPixelFilter` | hard | `vincent_border_pixel` | [vincent_border_pixel.md](vincent_border_pixel.md) |
| 3 | `BorderLaplacianBlurFilter` | score | `blur_laplacian_threshold` / `blur_laplacian_outlier` | [blur_laplacian.md](blur_laplacian.md) |
| 4 | `BorderTenengradBlurFilter` | score | `blur_tenengrad_threshold` / `blur_tenengrad_outlier` | [blur_tenengrad.md](blur_tenengrad.md) |
| 5 | `VincentsArtifactsFilter` | score | `vincents_artefacts_threshold` / `vincents_artefacts_outlier` | [vincents_artefacts.md](vincents_artefacts.md) |

## Soft filters (diagnostics only, not in the default order)

| Filter | Weight attr | Doc |
|--------|-------------|-----|
| `VincentsAreaFilter` | `vincents_area` | [vincents_area.md](vincents_area.md) |
| `VincentsMotionBlurFilter` | `vincents_motion_blur` | [vincents_motion_blur.md](vincents_motion_blur.md) |

## Shared mechanics

- [variants.md](variants.md) — `FilterVariant`: the uniform threshold/outlier
  rejection layer used by all score filters, plus `reject_soft_variants` for
  soft weights.
- [legacy.md](legacy.md) — the old whole-image filters (`area`, `border`,
  `occlusion`, `confidence`, `completeness`, whole-image `blur`), kept only
  for custom `--filter_order` runs and **not** part of the default set.

## Common helpers (`preprocessing/vincent_utils.py`)

Several filters share ported helpers from [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection):

- `compute_boundary_band` — ring straddling the mask contour: `dilate(mask) XOR erode(mask)`.
- `compute_boundary_blur_variance` / `compute_boundary_tenengrad` — Laplacian
  variance / mean Sobel magnitude restricted to the boundary band.
- `compute_artifact_mask` — pixels where `open(mask)` and `close(mask)` disagree.
- `touches_border_pixels` — true if any foreground pixel lies on the frame edge.
- `robust_center_scale` / `one_sided_weight` / `fit_robust_scores` — the robust
  population pass used by the soft filters.
