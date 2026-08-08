# ConfidenceFilter (`confidence`)

**Source:** `preprocessing/future_work/confidence.py`<br>
**Kind:** hard filter — rejects below an absolute confidence cutoff.<br>
**Alternative** (legacy/future work) — NOT part of the default `filter_order`;
custom `--filter_order` only.

## Purpose

Rejects observations whose detection confidence is unusably low (e.g. false
negatives where the detector failed entirely). Reads the per-observation
confidence score directly.

## Algorithm

```python
confidence = observation.confidence      # detector/segmenter output
```

## Decision

```
score  = min(confidence / minimum_confidence, 1)   # default minimum_confidence = 0.5
passed = confidence >= minimum_confidence
reason = "low_confidence"                 (when rejected)
```

The raw confidence is also published on `metrics.confidence`.

## Configuration

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `False` | Run this filter |
| `minimum_confidence` | `0.5` | Absolute minimum detection confidence |

The outlier population criterion is layered on via `OutlierFilter` when
`ConfidenceConfig.outlier_z` is set (see [variants.md](variants.md)).

## Notes

- Passes when `observation.confidence` is absent (`0.0` would trip a configured
  floor).
- NOT tested / likely not working as a proper pre-filter — kept for custom
  `--filter_order` runs. See [future_work.md](future_work.md).
