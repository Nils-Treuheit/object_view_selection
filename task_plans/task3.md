# Task 3 — Pre-filter inspection: accepted_samples + rejected_samples by reason + --only_pre_filter

## Goal
- Inspect the pre-filter pipeline in detail.
- With the **debug** flag:
  - `rejected_samples/` gets **subfolders per filter/reason** — why each sample
    was rejected at a glance (`rejected_samples/<reason>/<obj_id>/{rgb,mask,...}`).
  - A separate **`accepted_samples/<obj_id>/...`** folder holds samples that
    passed the pre-filter (accepted) but were **not** selected.
- New **`--only_pre_filter`** flag: stop right after the pre-filter stage —
  no quality scoring, no embedding, no selection, no linked plotting — while
  still dumping `accepted_samples/` and `rejected_samples/<reason>/`.

## Current state
- `run.py`:
  - `save_rejected_samples(rejected, data_root, output_dir)` — flat `<obj_id>`
    layout (legacy, kept for tests).
  - `save_rejected_samples_by_reason(rejected, data_root, output_dir)` — groups
    into `rejected_samples/<sanitized_reason>/<obj_id>/{rgb,mask,...}`.
  - `save_accepted_samples(unselected, data_root, output_dir)` —
    `accepted_samples/<obj_id>/{rgb,mask,...}`.
  - Full path: `save_accepted_samples(unselected, ...)` gated on `cfg.debug`
    (line ~635); `save_rejected_samples_by_reason` gated on `cfg.save_rejected`
    (line ~640).
  - `--only_pre_filter` argparse flag + `PipelineConfig.only_pre_filter`
    (config.py line ~194). In that mode it dumps both folders and returns
    before quality scoring / embedding / selection / plots (lines ~485-520).
- `preprocessing/variants.py` / `filter_pipeline.py` produce annotated reasons
  (`<reason>_threshold`, `<reason>_outlier`) that group cleanly (Task 4).

## Work items
1. Save helpers — DONE.
2. Debug-gated `accepted_samples` in the full path — DONE.
3. By-reason `rejected_samples` — DONE.
4. `--only_pre_filter` — DONE (both the pre-filter dump path and the early
   return after the pre-filter loop).
5. Docs — README/docker docs mention the folders. Verify the output tree text
   matches the by-reason layout.

## Verification
- Full debug run: `rejected_samples/<reason>/...` per reason; `accepted_samples/<obj_id>/...`
  contains accepted-but-unselected frames.
- `--only_pre_filter` run stops before selection and prints
  `Pre-filter only (--only_pre_filter): accepted X, rejected Y`.
