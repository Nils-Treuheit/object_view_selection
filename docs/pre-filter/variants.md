# Rejection Layer: `OutlierFilter`

**Source:** `preprocessing/variants.py` (wrapper), `preprocessing/base.py`
(`ScoreFilter`), `preprocessing/filter_utils.py` (shared helpers).

Every pre-filter returns `(score, passed, reason)` with `score ∈ [0, 1]`
(higher = better). Every non-binary filter has to implement the two
`BaseFilter` rejection criteria — an **absolute threshold** that filters out
complete unusable garbage and a **population-based outlier** rejection that
removes noticeably bad outliers — and `ScoreFilter`
(`preprocessing/base.py`) is the single shared implementation of both, once.

## The two rejection criteria (`ScoreFilter`)

| Criterion | Config knob | Rule | Reason suffix |
|-----------|-------------|------|---------------|
| Absolute garbage threshold | `hard_min` / `hard_max` | raw `stat < hard_min` or `stat > hard_max` → reject | `<reason>_threshold` |
| Population bad-outlier | `outlier_z` | `z = (stat - median) / (MAD * 1.4826)` beyond `outlier_z` on the `direction` tail → reject | `<reason>_outlier` |

The threshold is an absolute, extremely-low-quality cutoff on the **raw stat**
in its natural units. The outlier mode is population-relative: the robust
(median, MAD*1.4826) of the raw stat is fit **once over the population**
(`fit`), then extreme bad outliers are dropped. Subclasses only implement
`compute_stat` (raw stat) and `compute_score` (stat → 0..1 goodness).

## `OutlierFilter`

```python
OutlierFilter(inner, outlier_z=None)
```

Wraps a filter that implements **only** its own absolute criterion (e.g. the
legacy `AreaFilter` / `BorderFilter`) and layers the population outlier
rejection on top, **without touching the wrapped filter's own logic**:

- `name` — the wrapped filter's class name.
- `need_fitting()` — `True` only when `outlier_z` is set; the pipeline calls
  `fit_observations` once before the main loop when any wrapped filter needs it.
- `fit(observations)` — runs the inner filter on every observation to collect
  scores, then stores the robust center/scale (`median`, `MAD * 1.4826`). A
  degenerate zero scale is replaced by `1.0`.
- `evaluate(observation)`:
  1. Defers to the inner filter; if the inner rejects, its reason is kept
     verbatim.
  2. Otherwise applies the outlier check on the inner score, rejecting with
     `<reason>_outlier`.

The default pre-filters implement both criteria themselves via `ScoreFilter`
and are therefore **not** wrapped; `OutlierFilter` exists for the legacy
filters (wired through `_maybe_outlier` in `run.py`).

## Shared helpers (`preprocessing/filter_utils.py`)

The robust-fit / z-outlier logic that used to be copy-pasted into every filter
lives in one module:

```python
median, robust_scale = robust_center_scale(values)   # MAD * 1.4826
robust = robust_fit(values)                           # (median, scale), scale floored at 1.0
rejected = outlier_rejected(stat, robust, outlier_z, direction)
```

`fit_stat_robust(observations, compute_stat, enabled)` fits over a population;
`one_sided_weight` / `fit_robust_scores` turn raw soft stats into `(0, 1]`
selection weights. MAD is the median absolute deviation from the median;
multiplying by `1.4826` makes it a standard-deviation-equivalent scale that is
robust to outliers unlike `std`.

## On-disk grouping

Rejections are saved under `rejected_samples/<reason>/`:
`threshold-based/` holds the `_threshold` variants (plus the pure hard
filters, which reject with the bare reason), `outlier-based/` holds the
`_outlier` variants. See `save_rejected_samples_by_reason` in `run.py`.
