# VincentBorderPixelFilter

**Source:** `preprocessing/vincent_border_pixel.py`<br>
**Ported from:** [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** hard filter — rejects its failing class outright.

## Purpose

Rejects objects whose mask touches the image frame — a strong signal that the
object is cut off at the edge of the field of view.

## Algorithm

Checks whether any foreground pixel lies on the first/last row or column of
the mask (see `touches_border_pixels` in `preprocessing/vincent_utils.py`):

```
touches = mask[0, :].any() or mask[-1, :].any()
       or mask[:, 0].any() or mask[:, -1].any()
```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_touches_border` | `1.0` if the mask touches the frame, else `0.0` |

## Decision

```
passed  = not touches
reason  = "vincent_border_pixel"   (when rejected)
score   = 0.0 / 1.0
```

## Configuration (`VincentBorderPixelConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |

No `hard_min` / `outlier_z` — the decision is binary (touches the frame or
not), so there is no raw-stat spectrum to fit and no `OutlierFilter` is
wrapped around it.

## Relationship to the legacy `BorderFilter`

This is a strict, binary version of the truncation idea. The legacy
`BorderFilter` (see [legacy.md](legacy.md)) additionally accepts fractional
contact (`maximum_ratio`, `edge_maximum_ratio`) and returns a graded score;
it is **not** part of the default set.

## Placement

Second in `FilterConfig.filter_order`, right after the empty-mask check and
before any image-based blur/artifact computation.
