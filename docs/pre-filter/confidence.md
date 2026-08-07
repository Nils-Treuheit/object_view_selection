# ConfidenceFilter (`confidence`)

**Source:** `preprocessing/confidence.py`<br>
**Ported from:** (built-in pre-filter)<br>
**Kind:** hard filter — derives a normalised score in `[0, 1]`, and simultaneously acts as a working pre-filter with the two
`BaseFilter` rejection criteria implemented in `evaluate`.<br> 
**Part of the default `filter_order`**.

## Purpose

Inspects per-observation object confidence scores from ``metrics.confidence`` — values that indicate how
likely an object was truly present and correctly detected. Frames whose confidence is unusably low are
dropped outright (absolute floor + population outlier), passing frames receive a normalised goodness score.

## Algorithm

1. **Raw statistical Value** (per observation)<br>
   The raw stat is the per-observation confidence from ``metrics.confidence``:

   ```python
   confidence = observation.metrics.confidence    # from detector/segmenter output
   ```

   This is read directly in `evaluate` by `getattr(observation.metrics, self.stat_attr, 0.0)`.

2. **Quality scaled stat**<br>
   The reported score is the raw stat scaled against a fixed reference anchor so it
   is comparable across datasets:

   ```python
   score = min(confidence / reference_confidence, 1.0)    # reference_confidence default 0.5
   ```

3. **Threshold-based Filter** (absolute)<br>
   If `hard_min_confidence > 0` and the statistical value is below it, the
   observation is rejected outright with reason `confidence_threshold`.
   This catches frames whose object-confidence is unusably low — e.g. false
   negatives or empty detections where the detector failed entirely.

4. **Population-based Filter** (relative)<br>
   Robust median/MAD z-score of the raw confidence, fit once over the population
   (`fit`, only when `outlier_z` is set):

   ```python
   z = (confidence - median) / robust_scale
   z <= -outlier_z    -> reject with reason confidence_outlier
   ```

   Object confidence is a continuous metric where low values indicate poor detections,
   so the "low_bad" (degraded detection quality) tail is where the noticeably-bad outliers live.

5. **Population pass**<br>
   Unlike `VincentSoftFilter`, this filter does **not** derive `(0, 1]` selection weights.
   The population pass exists solely to fit robust median/MAD for the outlier mode; passing
   frames get the quality-scaled stat score as their output.

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.confidence` | raw stat: per-observation object confidence from detector/segmenter |
| `score` | fitted weight in `[0, 1]` (confidence / reference_confidence) |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = quality scaled stat`
and implements the two rejection criteria from `BaseFilter`:

- **Threshold-based — Absolute Garbage Rejection** <br>
  ```
  confidence < hard_min_confidence  →  (score, False, "confidence_threshold")
  ```
  A confidence below the floor is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection** <br>
  ```
  z <= -outlier_z  →  (score, False, "confidence_outlier")
  ```
  Requires the `fit(observations)` population pass (run automatically by
  `apply_hard_filters` when `requires_fit()` is true); the robust median/MAD
  are taken from the raw confidence distribution.

- **Pass** <br>
  ```
  (score, True, "confidence")
  ```

## Configuration (`ConfidenceFilter`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and apply thresholds |
| `hard_min_confidence` | `0.3` | Absolute hard-reject floor on raw confidence (`0` disables) |
| `reference_confidence` | `0.5` | Reference anchor for quality-scaled score (`confidence / reference_confidence`) |
| `outlier_z` | `None` | Robust outlier cutoff — on the raw confidence (`fit`/`evaluate`) |

## Notes

- The raw stat is always recorded even when a rejection criterion trips, so
  downstream diagnostics see it.
- If ``observation.metrics.confidence`` does not exist or has no value, the
  confidence defaults to `0.0` (a configured hard floor would reject it).
- This is a **hard** filter with no soft-weight derivation path — population statistics are used solely for outlier detection, not for ranking passing observations.
