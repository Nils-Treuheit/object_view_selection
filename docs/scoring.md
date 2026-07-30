# Scoring

All quality scores are normalised to **[0, 1]**, where **1.0 = perfect** (ideal candidate) and **0.0 = worst** (unusable candidate). This makes individual dimensions comparable and the multi-metric aggregation well-behaved.

## Individual Quality Metrics

| Metric | Range | Higher meaning |
|---|---|---|
| `blur` | [0, 1] | Sharp, in-focus image |
| `area` | [0, 1] | Object occupies a large fraction of the frame |
| `occlusion` | [0, 1] | No hand/obstacle overlapping the object |
| `completeness` | [0, 1] | Full shape visible (solid, not truncated or fragmented) |

Each is computed by a dedicated `QualityMetric` implementation in `quality/`:

- **`BlurQuality`** – Laplacian variance divided by `max_lap` (2× the pre-filter threshold), clipped to 1.0.
- **`AreaQuality`** – Mask pixel fraction divided by 0.20, clipped to 1.0.
- **`OcclusionQuality`** – 1.0 minus the fraction of mask pixels overlapped by `object_hand`.
- **`CompletenessQuality`** – Reads the pre-filter completeness score (weighted average of solidity, extent, convexity).

## Confidence

`confidence = min(blur, area, occlusion, completeness)`

Confidence is the **weakest-link** score: it reflects the worst-performing quality dimension for a given view. A view with high confidence is strong in every respect; a low confidence immediately identifies the binding constraint.

Confidence is computed **post-hoc** after all individual scores are known and is stored in `obs.metrics.confidence`.

## Score (Final Quality)

`score = Σ(wᵢ · metricᵢ) / Σ wᵢ`

The final quality score is a **weighted arithmetic mean** of the individual quality metrics. Weights are configured via `QualityWeights` in `config.py`. The result is also in [0, 1].

The score is stored in `obs.quality` and exported as both `quality` and `score` in `quality.csv`.

## Relationship

```
individual metrics  ──weighted sum──▶  score  (aggregate quality)
                                       confidence  (weakest link, lower bound)
```

- **Score** answers: *"How good is this view overall?"*
- **Confidence** answers: *"Is there any dimension that makes this view risky?"*
