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

## Gaps (ALL DONE)
1. **Soft filters wired** — `VincentsAreaConfig`, `VincentsArtifactsConfig`,
   `VincentsMotionBlurConfig` now have `threshold_min` / `outlier_z`; `VincentSoftFilter`
   gained the two class attrs; `run.build_soft_filters` uses `_soft_with_variant(f, conf)`
   and after `apply_soft_filters` the pipeline calls
   `reject_soft_variants(soft_filters, accepted, rejected)` (in
   `preprocessing/variants.py`) which rejects accepted obs whose **fit weight**
   trips the cutoff: `weight < threshold_min` → reason `<key>_threshold`,
   `z(weight) <= -outlier_z` → reason `<key>_outlier`. Verified live:
   `vincents_area_threshold` (88), `vincents_area_outlier` (36) folders in
   `rejected_samples/`.
2. **Tests added** — `tests/correctness_test_units/test_filters.py`: hard
   threshold (`<reason>_threshold`), inner-reason-verbatim, hard outlier
   (`<reason>_outlier`, fit required), soft threshold + soft outlier pass on fit
   weights, no-knob no-op, and `FilterPipeline` integration
   (`requires_fit`/`fit_observations`). `test_pipeline.py`: `build_filters`
   wraps configured hard variants and flags `requires_fit`;
   `build_soft_filters` propagates the knobs.
3. Docs done — `docs/thresholds.md` has a "Threshold & Outlier Variants"
   section listing the knobs + reasons for hard and soft filters.

## Work items
1. `config.py`: `threshold_min` / `outlier_z` on the three soft filter configs — DONE.
2. `preprocessing/variants.py`: `reject_soft_variants` on already-fit weights — DONE.
3. `run.py`: soft variant pass after `apply_soft_filters` — DONE.
4. Tests in `test_filters.py` + `test_pipeline.py` — DONE.
5. Docs: thresholds.md/README knobs + reasons — DONE.

## Verification
- `python tests/run_correctness.py` passes — 211/0, 760 assertions.
- `python tests/run_smoke.py --data_root .../09_triprong_old` passes — 51/0.
- Real run with knobs enabled rejects extreme-bad frames and groups them under
  `rejected_samples/<filter>_threshold|outlier/` — verified (blur_outlier
  correctly did NOT fire: no z<=-3 accepted outlier).
- `--only_pre_filter` with knobs enabled still dumps annotated reasons —
  verified (accepted 126, rejected 248; `rejected.json` carries the
  `*_threshold` / `*_outlier` reasons).
