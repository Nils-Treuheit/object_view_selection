# Pre-Filter Reference

Detailed, per-filter documentation for the pre-filter stage. The pipeline
overview lives in [`../pipeline.md`](../pipeline.md); this directory covers
each filter's algorithm, metrics, configuration, and rejection behaviour in
isolation.

## Terminology

Every pre-filter's `evaluate(observation)` returns `(score, passed, reason)`:

- **Hard (binary) filter** — always returns `passed=False` with a bare reason
  (e.g. `vincent_empty_mask`) for the failing class. Score is 0/1. Exempt from
  the outlier criterion — there is no "goodness" spectrum to fit.
- **Score filter** — implements both `BaseFilter` rejection criteria itself via
  the shared `ScoreFilter` base: an absolute garbage threshold on the raw stat
  (`<reason>_threshold`) plus a population-relative extreme-bad outlier
  rejection (`<reason>_outlier`, robust median/MAD z). Returns a 0..1
  "goodness" score.
- **Soft filter** — a `VincentSoftFilter(ScoreFilter)`: computes a raw stat
  per observation, then `fit_weights` runs a population pass converting it into
  a selection weight in `(0, 1]` (diagnostics). Because it is also a
  `ScoreFilter` it can act as a working pre-filter when the hard-min / outlier
  knobs are configured.

## Default set (5 filters, in `FilterConfig.filter_order`)

| # | Filter | Kind | Rejection reason(s) | Doc |
|---|--------|------|--------------------|-----|
| 1 | `VincentEmptyMaskFilter` | hard | `vincent_empty_mask` | [vincent_empty_mask.md](vincent_empty_mask.md) |
| 2 | `VincentBorderPixelFilter` | hard | `vincent_border_pixel` | [vincent_border_pixel.md](vincent_border_pixel.md) |
| 3 | `BorderLaplacianBlurFilter` | score | `blur_laplacian_threshold` / `blur_laplacian_outlier` | [blur_laplacian.md](blur_laplacian.md) |
| 4 | `BorderTenengradBlurFilter` | score | `blur_tenengrad_threshold` / `blur_tenengrad_outlier` | [blur_tenengrad.md](blur_tenengrad.md) |
| 5 | `VincentsArtifactsFilter` | score | `vincents_artefacts_threshold` / `vincents_artefacts_outlier` | [vincents_artefacts.md](vincents_artefacts.md) |

## Soft filters (diagnostics + working pre-filters, not in the default order)

| Filter | Weight attr | Doc |
|--------|-------------|-----|
| `VincentsAreaFilter` | `vincents_area` | [vincents_area.md](vincents_area.md) |
| `VincentsMotionBlurFilter` | `vincents_motion_blur` | [vincents_motion_blur.md](vincents_motion_blur.md) |

## Shared mechanics

- [variants.md](variants.md) — the shared `ScoreFilter` rejection criteria
  plus the `OutlierFilter` wrapper layered on legacy filters.
- [legacy.md](legacy.md) — the old whole-image filters (`area`, `border`,
  `occlusion`, `confidence`, `completeness`, whole-image `blur`), kept only
  for custom `--filter_order` runs and **not** part of the default set.

## Common helpers

- `preprocessing/vincent_utils.py` — mask/blur helpers ported from
  [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection):
  `compute_boundary_band` (ring straddling the mask contour:
  `dilate(mask) XOR erode(mask)`), `compute_boundary_blur_variance` /
  `compute_boundary_tenengrad` (Laplacian variance / mean Sobel magnitude
  restricted to the boundary band), `compute_artifact_mask` (pixels where
  `open(mask)` and `close(mask)` disagree), `touches_border_pixels`.
- `preprocessing/filter_utils.py` — the **shared** rejection helpers every
  filter uses: `robust_center_scale`, `one_sided_weight`,
  `fit_robust_scores` (the robust population pass), plus `robust_fit`,
  `fit_stat_robust`, `outlier_rejected` (the fit/z-outlier logic that used to
  be copy-pasted per filter).
