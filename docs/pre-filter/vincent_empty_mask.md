# VincentEmptyMaskFilter

**Source:** `preprocessing/vincent_empty_mask.py`<br>
**Ported from:** [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** hard filter — rejects its failing class outright.

## Purpose

Drops observations whose mask contains no foreground pixels at all (an empty
or fully-black mask). This is the cheapest possible check and the first filter
in the default pipeline.

## Algorithm

Count foreground pixels in the mask:

```
pixel_count = sum(mask > 0)
```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_pixel_count` | number of mask pixels `> 0` (also written even when the filter passes) |

## Decision

```
passed  = pixel_count > 0
reason  = "vincent_empty_mask"   (when rejected)
score   = 0.0 / 1.0
```

## Configuration (`VincentEmptyMaskConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |

No `threshold_min` / `outlier_z` — the filter hard-rejects already, so no
`FilterVariant` is wrapped around it (`run.py` only wraps score filters).

## Placement

First in `FilterConfig.filter_order`: it only touches the mask, so it rejects
cheaply before any image-based computation.
