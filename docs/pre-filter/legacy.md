# Legacy Filters

**Status: NOT part of the default pre-filter set. Kept only for custom
`--filter_order` runs. Not tested / likely not working as proper pre-filters.**

These are the original whole-image filters from before the Vincent pre-filter
rework. They hard-reject with a single absolute cutoff, which tends to be
either too aggressive (rejecting most of a variable dataset) or too lax on a
clean dataset, and several are not exercised by the test suite. Prefer the
default filters in [README.md](README.md).

They are registered in `run.py` and can be enabled via `--filter_order` (e.g.
`--filter_order vincent_empty_mask,blur_laplacian,area`), but the default
`FilterConfig.filter_order` does not include any of them.

## `AreaFilter` (`preprocessing/area_filter.py`)

Rejects small objects.

| Metric | Field |
|--------|-------|
| area ratio = mask pixels / canvas pixels | `metrics.area_ratio` |

```
score  = min(ratio / minimum_ratio, 1.0)
passed = ratio >= minimum_ratio            (default minimum_ratio = 0.01)
reason = "small_object"
```

Superseded by the soft `VincentsAreaFilter` (see
[vincents_area.md](vincents_area.md)).

## `BorderFilter` (`preprocessing/border_truncation.py`)

Rejects objects cut off at the image frame, using two graded measures:

| Metric | Field |
|--------|-------|
| ring ratio = frame-border mask pixels / mask pixels | `metrics.border_ratio` |
| per-edge contact = mask pinned to each frame edge / mask extent | `metrics.edge_{top,bottom,left,right}_ratio`, `metrics.edge_ratio` |

```
ring_score = 1 - min(ring_ratio / maximum_ratio, 1)
edge_score = 1 - min(edge_ratio / edge_maximum_ratio, 1)
score      = min(ring_score, edge_score)
passed     = ring_ratio <= maximum_ratio and edge_ratio <= edge_maximum_ratio
reason     = "border"          (defaults maximum_ratio = 0.05, edge_maximum_ratio = 0.25)
```

Empty masks are rejected with reason `empty_mask`. The default set instead
uses the strict binary `VincentBorderPixelFilter`
([vincent_border_pixel.md](vincent_border_pixel.md)).

## `OcclusionFilter` (`preprocessing/occlusion_filter.py`)

Rejects objects whose mask overlaps a hand mask beyond `maximum_overlap`
(requires `object_hand` data; passes everything when it's absent).

| Metric | Field |
|--------|-------|
| hand overlap = mask ∩ hand / mask | `metrics.hand_overlap` |

```
score  = 1 - min(ratio / maximum_overlap, 1)
passed = ratio <= maximum_overlap            (default maximum_overlap = 0.15)
reason = "occlusion"
```

## `ConfidenceFilter` (`preprocessing/confidence.py`)

Rejects observations below a detection-confidence threshold (`minimum_confidence`,
default 0.5). Reads `observation.confidence`; passes when it's absent.

```
score  = min(confidence / minimum_confidence, 1)
passed = confidence >= minimum_confidence
reason = "low_confidence"
```

## `CompletenessFilter` (`preprocessing/completeness_filter.py`)

Rejects objects whose visible shape is incomplete, via a weighted blend of
three contour shape metrics:

| Metric | Weight (default) | Field |
|--------|------------------|-------|
| solidity = area / convex-hull area | 0.4 | `metrics.solidity` |
| extent = area / bounding-rect area | 0.3 | `metrics.extent` |
| convexity = hull perimeter / perimeter | 0.3 | `metrics.convexity` |

```
score  = 0.4*solidity + 0.3*extent + 0.3*convexity
passed = score >= minimum_score              (default minimum_score = 0.65)
reason = "incomplete_shape"
```

`metrics.completeness` stores the blended score. Empty masks are rejected with
reason `empty_mask`.

## `BlurFilter` (`preprocessing/blur_filter.py`)

Old whole-image sharpness filter (two thresholds). **Not registered** in
`run.py`'s `available` map — the boundary-band `blur_laplacian` /
`blur_tenengrad` filters replaced it.

| Metric | Field |
|--------|-------|
| whole-image Laplacian variance | `metrics.laplacian` |
| whole-image Tenengrad (mean Sobel magnitude) | `metrics.tenengrad` |

```
lap_score = min(lap / laplacian_threshold, 1)      # default 120
ten_score = min(ten / tenengrad_threshold, 1)      # default 35
score     = 0.5*lap_score + 0.5*ten_score
passed    = lap >= laplacian_threshold and ten >= tenengrad_threshold
reason    = "blur"
```

Whole-image measures are dominated by background texture and diluted by
low-texture interiors — the reason the boundary-band approach replaced it.
