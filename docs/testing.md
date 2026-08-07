## Testing

The project has **779 correctness checks** and **55 smoke tests**.

### Correctness Tests

Each filter, metric, descriptor, and selector is tested against **synthetic data** with known correct outputs (no external dataset needed):

```bash
# Run all correctness tests
python tests/run_correctness.py
```

Test files are in `tests/correctness_test_units/`:
- `test_filters.py` — the classic pre-filter modules with known pass/fail cases
- `test_vincent_filters.py` — the Vincent filters (empty mask, border pixel, area/artifacts/motion-blur) and robust population scoring
- `test_quality.py` — the 4-component quality scorer (blur, area, artifacts, centerness) with known weights
- `test_descriptors_invariants.py` — Hu/Zernike rotation/translation/scale invariance
- `test_descriptors_shape.py` — Fourier/Shape Context invariance and discrimination
- `test_selection.py` — all 5 selectors produce correct output sizes and diversity
- `test_edge_case.py` — empty masks, duplicate embeddings, degenerate inputs
- `test_crops.py` — bbox/masked/square crop correctness
- `test_metrics.py` — ObservationMetrics defaults and setters
- `test_plotting.py` — plot generation, standalone loader, DR methods

### Smoke Tests

Run against a real dataset to verify the pipeline runs end-to-end:

```bash
python tests/run_smoke.py --data_root /path/to/bottle
```

Smoke tests check that every component produces the right output types, shapes, and value ranges on real data.