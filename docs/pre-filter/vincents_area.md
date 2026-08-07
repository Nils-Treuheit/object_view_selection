# VincentsAreaFilter (`vincents_area`)

**Source:** `preprocessing/vincents_area_filter.py`<br>
**Ported from:** `score_mask_area` in [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** soft filter — never rejects; derives a population-adapted selection
weight in `(0, 1]`. **Not** part of the default `filter_order`; used as a
diagnostic and, optionally, as a rejection layer on the fitted weight.

## Purpose

Penalises objects that occupy a tiny fraction of the frame. Small masks are
harder to recognize and often mean the object is far away or poorly framed.

## Algorithm

1. **Raw stat** (per observation, `evaluate`):

   ```
   vincent_area_fraction = mask_pixels / canvas_area
   ```

2. **Population pass** (`fit_weights` / `fit_robust_scores`): compute the
   robust center and scale of the stat over the accepted population
   (`median`, `MAD * 1.4826`), then a one-sided half-Gaussian falloff on the
   "bad" (low) side:

   ```
   weight = exp(-0.5 * (z / softness)^2),   z = (median - stat) / robust_scale
   ```

   Mask area is a continuous spectrum rather than a tight cluster with rare
   outliers, so the softness is deliberately small (`0.3` robust-MADs) to
   discriminate at all.

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_area_fraction` | raw stat: mask pixels / canvas pixels |
| `metrics.vincents_area` | fitted weight in `(0, 1]` |

## Score and rejection

The filter always returns `(1.0, True, "")` — it never hard-rejects. Optional
`threshold_min` / `outlier_z` knobs on the config turn the fitted weight into
a rejection pass via `reject_soft_variants` (see
[variants.md](variants.md)):

```
reason = "vincents_area_threshold"   (weight < threshold_min)
      | "vincents_area_outlier"      (weight z <= -outlier_z)
```

## Configuration (`VincentsAreaConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and weight |
| `softness` | `0.3` | Falloff in robust-MADs |
| `threshold_min` | `None` | Optional floor on the fitted weight |
| `outlier_z` | `None` | Optional robust outlier cutoff on the weight |

## Notes

- Weights are fit only on the **accepted** set (rejected observations don't
  compete for selection), but raw stats are computed for all observations so
  diagnostic plots can compare.
- `VincentsAreaQuality` scores the same raw stat against a fixed global
  anchor (`area_max_fraction = 0.20`), keeping quality comparable across
  datasets — in contrast to this filter's population-relative weight.
