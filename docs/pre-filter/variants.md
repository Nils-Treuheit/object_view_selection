# Rejection Layering: `FilterVariant` and `reject_soft_variants`

**Source:** `preprocessing/variants.py`

Every pre-filter returns `(score, passed, reason)` with `score ∈ [0, 1]`
(higher = better). `FilterVariant` wraps an existing filter and adds two
optional rejection modes on top, **without touching the wrapped filter's own
logic**. This lets the score filters (`blur_laplacian`, `blur_tenengrad`,
`vincents_artefacts`) keep a single, conservative rejection scheme, and lets
the soft filters reject on their fitted weights.

## Modes

| Mode | Config knob | Rule | Reason suffix |
|------|-------------|------|---------------|
| Threshold | `threshold_min` | `score < threshold_min` → reject | `<reason>_threshold` |
| Outlier | `outlier_z` | `z = (score - median) / MAD*1.4826 ≤ -outlier_z` → reject | `<reason>_outlier` |

The threshold is an absolute, extremely-low-quality cutoff. The outlier mode
is population-relative: scores are fit **once over the population**, then
extreme bad outliers are dropped. Because both key off the same 0..1 score
every filter already returns, the wrapper is uniform across hard and score
filters.

## `FilterVariant`

```python
FilterVariant(inner, threshold_min=None, outlier_z=None)
```

- `name` — the wrapped filter's class name.
- `requires_fit()` — `True` only when `outlier_z` is set; the pipeline calls
  `fit_observations` once before the main loop when any wrapped filter needs it.
- `fit(observations)` — runs the inner filter on every observation to collect
  scores, then stores the robust center/scale (`median`, `MAD * 1.4826`). A
  degenerate zero scale is replaced by `1.0`.
- `evaluate(observation)`:
  1. Defers to the inner filter; if the inner rejects, its reason is kept
     verbatim.
  2. Otherwise applies the threshold check, then the outlier check, rejecting
     with the annotated reason.

## `reject_soft_variants(soft_filters, accepted, rejected)`

Soft filters (`VincentsAreaFilter`, `VincentsMotionBlurFilter`) never
hard-reject in `evaluate`; they store a population weight in `(0, 1]` on
`obs.metrics.<weight_attr>`. When such a filter has `threshold_min` or
`outlier_z` configured, this post-pass moves accepted observations whose
weight trips the cutoff into `rejected`, with the annotated reason
(`<name>_threshold` / `<name>_outlier`) so the per-reason sample folders group
them cleanly.

Called after `apply_soft_filters` in `run.py`:
`reject_soft_variants(soft_filters, accepted, rejected)`.

## Robust statistics (`preprocessing/vincent_utils.py`)

```python
median, robust_scale = robust_center_scale(values)   # MAD * 1.4826
```

MAD is the median absolute deviation from the median; multiplying by `1.4826`
makes it a standard-deviation-equivalent scale that is robust to outliers
unlike `std`.

## On-disk grouping

Rejections are saved under `rejected_samples/<reason>/`:
`threshold-based/` holds the `_threshold` variants (plus the pure hard
filters, which reject with the bare reason), `outlier-based/` holds the
`_outlier` variants. See `save_rejected_samples_by_reason` in `run.py`.
