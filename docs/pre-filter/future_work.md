# Possible Future Filters

**Status: NOT part of the default pre-filter set. Kept only for custom
`--filter_order` runs. Not tested / likely not working as proper pre-filters.**

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