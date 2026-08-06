# Task 4 — Threshold + outlier variants of every pre-filter

## Goal
- **Every pre-filter** gets two variants:
  - **Threshold-based**: everything below an extremely-low-quality score cutoff
    is thrown (`score < threshold_min` → reject, reason `<filter>_threshold`).
  - **Outlier-based**: extreme-bad outliers are kicked as well, using a robust
    z-score (`(score - median) / (1.4826 * MAD)`) against the population;
    `z <= -outlier_z` → reject, reason `<filter>_outlier`.
- Must be configurable per filter and enabled on top of the existing filter
  logic without breaking soft/fit-based filters.

## Current state
- `preprocessing/variants.py` — `FilterVariant(inner, threshold_min=None,
  outlier_z=None)`: defers to the inner filter; if the inner rejects, its
  reason is kept verbatim; otherwise threshold/outlier may reject with the
  annotated reason. `requires_fit()` true only for outlier mode; `fit(observations)`
  runs a robust median/MAD pass.
- `preprocessing/filter_pipeline.py` — `requires_fit` / `fit_observations`
  (one population pass before the main loop).
- `run.py` — `_maybe_variant(f, conf)` + `build_filters(cfg)` wraps **all 8
  hard filters**: blur, area, border, occlusion, confidence, completeness,
  vincent_empty_mask, vincent_border_pixel.
- `config.py` — the 8 hard filter configs have `threshold_min` / `outlier_z`
  fields (default `None` = disabled).

## Gaps (NOT DONE)
1. **Soft filters are NOT wired**: `VincentsAreaFilter`, `VincentsArtifactsFilter`,
   `VincentsMotionBlurFilter` never hard-reject (`evaluate` returns `(1.0, True, "")`)
   and their configs (`VincentsAreaConfig`, `VincentsArtifactsConfig`,
   `VincentsMotionBlurConfig`) lack `threshold_min` / `outlier_z`. The user wants
   **each** pre-filter to have both variants, so the soft filters need them too.
   Plan: add the two fields to the three soft configs; after `apply_soft_filters`
   fits the population weights (in `(0,1]`), run a rejection pass on the
   **weights**: `weight < threshold_min` → reason `<filter>_threshold`,
   `z(weight) <= -outlier_z` → reason `<filter>_outlier` (fit on the accepted
   population). Move the rejected obs from `accepted` to `rejected`.
2. **No tests exist** for the variants. Add to
   `tests/correctness_test_units/test_filters.py`:
   - threshold variant rejects below cutoff with `<reason>_threshold`;
   - outlier variant rejects the extreme-bad point and keeps the rest
     (`<reason>_outlier`);
   - soft-filter threshold/outlier rejection pass works on the fit weights;
   - reasons match `<filter>_threshold` / `<filter>_outlier` so Task 3's
     per-reason folders work.
3. Docs: README / `docs/thresholds.md` should list the new knobs + reasons.

## Work items
1. `config.py`: add `threshold_min` / `outlier_z` to the three soft filter
   configs.
2. `preprocessing/variants.py`: add a helper to apply threshold/outlier
   rejection on already-fit weights (e.g. `reject_soft_weights`), reusable by
   `run.py`.
3. `run.py`: after `apply_soft_filters(soft_filters, accepted, rejected)`,
   apply the soft variant pass so accepted obs whose weight trips a configured
   threshold/outlier move to `rejected` with the annotated reason.
4. Tests in `test_filters.py` (and possibly `test_pipeline.py`) covering
   hard + soft variants.
5. Docs: thresholds.md/README knobs + reasons.

## Verification
- `python tests/run_correctness.py` passes.
- Real run with knobs enabled rejects extreme-bad frames and groups them under
  `rejected_samples/<filter>_threshold|outlier/`.
- `--only_pre_filter` with knobs enabled still dumps annotated reasons.
