# Task 2 — Debug plots: original + mask overlay vs embedding model input

## Goal
- With the **debug** flag, write plots showing a few samples comparing the
  original image (with mask overlay) against the embedding model's input.
- Per figure: **4 examples** in a **4x4 image matrix** (one row per sample:
  original, original + mask, 224x224 embedding input, input + mask).
- **3 plots**, samples drawn at random.
- Saved in an `embedded_samples` folder (sibling of `plots/` / `bad_examples/`).

## Current state
- `plotting_process/embedded_samples_plots.py` — DONE:
  - `plot_embedded_samples(pool_obs, output_dir, n_figures=3, n_examples=4,
    random_state=0)`.
  - Each figure is `n_examples` rows x 4 cols (original / original+mask /
    `padded_square_crop(..., size=224)` / input+mask).
  - Random sampling with a fixed seed (reproducible).
  - Writes `<output_dir>/embedded_samples/samples_<n:02d>.png`.
- `plotting_process/wrapper.py` `plot_all(...)` — DONE: gated on `debug and
  pool_obs` (line ~230).
- `plotting_process/embedding_plots/base.py` provides `padded_square_crop`
  via `embeddings/crop.py`.

## Work items
1. Core plotter — DONE.
2. `plot_all` wiring (debug-gated) — DONE.
3. Docs — README output tree mentions `embedded_samples/samples_<NN>.png`;
   `docs/plotting.md` tree lists it. Verify consistent.

## Verification
- Real run with `--debug` produced `embedded_samples/samples_01.png` …
  `samples_03.png` (verified on 09_triprong_old).
- Plot grid is 4 rows x 4 cols; files are 3; sampling reproducible.
