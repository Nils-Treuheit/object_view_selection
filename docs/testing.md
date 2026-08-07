## Testing

The project has **223 correctness test functions** (hundreds of individual
`check()` assertions) and **55 smoke test checks**.

### Correctness Tests

Each filter, metric, descriptor, and selector is tested against **synthetic data**
with known correct outputs (no external dataset needed):

```bash
# Run all correctness tests
python tests/run_correctness.py
# or
python test_correctness.py

# Expected output: Results: 223 test functions passed, 0 failed out of 223
```

Test files are in `tests/correctness_test_units/`:

- `test_filters.py` — the classic pre-filter modules with known pass/fail cases
- `test_vincent_filters.py` — the Vincent filters (empty mask, border pixel,
  area/artifacts/motion-blur) and robust population scoring
- `test_quality.py` — the 4-component quality scorer (blur, area, artifacts,
  centerness) with known weights
- `test_pipeline.py` — `run.py` wiring: filter order, hard-filter variants,
  soft-filter passes, scorer composition, weakest-link confidence
- `test_quality_floor.py` — the adaptive quality-floor logic (percentile,
  `minimum_pool` guarantee, `num_views` cap)
- `test_descriptors_invariants.py` — Hu/Zernike rotation/translation/scale invariance
- `test_descriptors_shape.py` — Fourier/Shape Context invariance and discrimination
- `test_selection.py` — all selectors produce correct output sizes and diversity
- `test_selection_algorithms.py` — per-selector behaviour: FPS farthest-picks,
  GQD start/score formula, Facility Location coverage, DPP quality-weighted
  kernel, kMeans-xNN quality leaders / xNN radius / medoid fallback / init modes
- `test_edge_case.py` — empty masks, duplicate embeddings, degenerate inputs
- `test_crops.py` — bbox/masked/square crop correctness
- `test_metrics.py` — ObservationMetrics defaults and setters
- `test_plotting.py` — plot generation, standalone loader, DR methods

### Smoke Tests

Run against a real dataset to verify the pipeline runs end-to-end:

```bash
python tests/run_smoke.py --data_root /path/to/bottle
# or
python test_smoke.py --data_root /path/to/bottle
```

Smoke tests check that every component produces the right output types, shapes,
and value ranges on real data (`test_data_io`, `test_filters`, `test_quality`,
`test_descriptors`, `test_selection`, `test_utils_module`, `test_embeddings`).

All suites must pass with **0 failures** before changes are considered complete.
