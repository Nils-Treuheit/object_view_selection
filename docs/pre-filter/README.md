# Pre-Filter Reference

Detailed, per-filter documentation for the pre-filter stage. The pipeline
overview lives in [`../pipeline.md`](../pipeline.md); this directory covers
each filter's algorithm, metrics, configuration, and rejection behaviour in
isolation.

## Overview

The **default set** (5 filters, in `FilterConfig.filter_order`) is deliberately
small and conservative. Every non-binary default filter implements the two
`BaseFilter` rejection criteria — an absolute garbage threshold and a
population-based bad-outlier rejection — once, via the shared `ScoreFilter`
base (`preprocessing/base.py` + `preprocessing/filter_utils.py`). The two hard
(binary) filters reject on a structural condition with a bare reason.

All other filters (Vincent soft filters, legacy, future work) are kept as
alternatives for custom `--filter_order` runs and diagnostics.

| Status | Filter | Kind | Rejection reason(s) | Doc |
|--------|--------|------|--------------------|-----|
| **(Default)** | `VincentEmptyMaskFilter` | hard | `vincent_empty_mask` | [vincent_empty_mask.md](vincent_empty_mask.md) |
| **(Default)** | `VincentBorderPixelFilter` | hard | `vincent_border_pixel` | [vincent_border_pixel.md](vincent_border_pixel.md) |
| **(Default)** | `BorderLaplacianBlurFilter` | score | `blur_laplacian_threshold` / `blur_laplacian_outlier` | [blur_laplacian.md](blur_laplacian.md) |
| **(Default)** | `BorderTenengradBlurFilter` | score | `blur_tenengrad_threshold` / `blur_tenengrad_outlier` | [blur_tenengrad.md](blur_tenengrad.md) |
| **(Default)** | `VincentsArtifactsFilter` | score | `vincents_artefacts_threshold` / `vincents_artefacts_outlier` | [vincents_artefacts.md](vincents_artefacts.md) |
| Alternative | `VincentsAreaFilter` | soft + score | `vincents_area_threshold` / `vincents_area_outlier` | [vincents_area.md](vincents_area.md) |
| Alternative | `VincentsMotionBlurFilter` | soft + score | `vincents_motion_blur_threshold` / `vincents_motion_blur_outlier` | [vincents_motion_blur.md](vincents_motion_blur.md) |
| Alternative | `BorderLaplacianBlurFilter` + `BorderTenengradBlurFilter` (combined) | score | — | [border_blur_filter.md](border_blur_filter.md) |
| Alternative | legacy `area` / `border` / whole-image `blur` | hard | bare reasons | [legacy.md](legacy.md) |
| Alternative | `occlusion` / `confidence` / `completeness` | hard | bare reasons | [future_work.md](future_work.md) |

## Shared mechanics

- [variants.md](variants.md) — the two `BaseFilter` rejection criteria via the
  shared `ScoreFilter`, plus the `OutlierFilter` wrapper layered on legacy
  filters.

## Terminology

- **Hard (binary) filter** — always returns `passed=False` with a bare reason
  (e.g. `vincent_empty_mask`) for the failing class. Score is 0/1. Exempt from
  the outlier criterion.
- **Score filter** — implements both `BaseFilter` rejection criteria itself via
  `ScoreFilter`: absolute garbage threshold on the raw stat
  (`<reason>_threshold`) + population outlier (`<reason>_outlier`).
- **Soft filter** — `VincentSoftFilter(ScoreFilter)`: computes a raw stat per
  observation, then `fit_weights` derives a `(0, 1]` selection weight
  (diagnostics); can also act as a working pre-filter.
