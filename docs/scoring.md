# Scoring

All quality scores are normalised to **[0, 1]**, where **1.0 = perfect** (ideal candidate) and **0.0 = worst** (unusable candidate). This makes individual dimensions comparable and the multi-metric aggregation well-behaved.

## Individual Quality Metrics

| Metric | Range | Higher meaning |
|---|---|---|
| `blur` | [0, 1] | Sharp, in-focus image |
| `area` | [0, 1] | Object occupies a large fraction of the frame |
| `occlusion` | [0, 1] | No hand/obstacle overlapping the object |
| `completeness` | [0, 1] | Full shape visible (solid, not truncated or fragmented) |
| `vincents_area` | (0, 1] | Mask area typical of the accepted population |
| `vincents_artefacts` | (0, 1] | Mask free of speckle/holes/ragged edges |
| `vincents_motion_blur` | (0, 1] | Object boundary is sharp |

Each is computed by a dedicated `QualityMetric` implementation in `quality/`:

- **`BlurQuality`** – Laplacian variance divided by a fixed global `max_lap` (`quality_anchors.blur_max_lap`, default 400), clipped to 1.0. The anchor is a dataset-independent constant so sharpness scores are comparable across datasets.
- **`AreaQuality`** – Mask pixel fraction divided by 0.20, clipped to 1.0.
- **`OcclusionQuality`** – 1.0 minus the fraction of mask pixels overlapped by `object_hand`.
- **`CompletenessQuality`** – Reads the pre-filter completeness score (weighted average of solidity, extent, convexity).
- **`VincentsAreaQuality`** – Raw mask-area fraction divided by a fixed global max (`quality_anchors.vincents_area_max_fraction`, default 0.20); same scale as `AreaQuality`.
- **`VincentsArtifactsQuality`** – `clip(1 − artifact_fraction / quality_anchors.vincents_artifacts_max_fraction, 0, 1)`: a mask whose artifact fraction reaches the anchor scores 0.
- **`VincentsMotionBlurQuality`** – Boundary-blur Laplacian variance divided by a fixed global max (`quality_anchors.vincents_motion_blur_max_variance`, default 10000), clipped to 1.0.

All three Vincent quality metrics are anchored in **fixed global max/min values**, independent of the dataset. The population-relative scores (`vincents_area`, `vincents_artefacts`, `vincents_motion_blur` weights) are still computed by the soft pre-filter pass for pre-filtering and diagnostics, but quality scoring uses the raw stats through the global anchors so quality is comparable across datasets.

## Confidence

`confidence = min(blur, area, occlusion, completeness, vincents_area, vincents_artefacts, vincents_motion_blur)`

Confidence is the **weakest-link** score: it reflects the worst-performing quality dimension for a given view. A view with high confidence is strong in every respect; a low confidence immediately identifies the binding constraint.

Confidence is computed **post-hoc** after all individual scores are known and is stored in `obs.metrics.confidence`.

## Score (Final Quality)

`score = Σ(wᵢ · metricᵢ) / Σ wᵢ`

The final quality score is a **weighted arithmetic mean** of the individual quality metrics. Weights are configured via `QualityWeights` in `config.py`. The result is also in [0, 1].

The score is stored in `obs.quality` and exported as both `quality` and `score` in `quality.csv`.

## Quality Floor (Selection Pool)

Before the embedding selection, an **adaptive quality floor** is applied to the accepted pool (`quality_floor.*` in `config.py`):

- `percentile` (default 0.10): the bottom 10% of accepted observations by quality are excluded from the selection pool.
- `absolute_min` (default 0.5): no observation below this absolute quality ever enters the selection pool.
- `minimum_pool` (default 20): the floor is capped so at least this many candidates remain for a diverse sample-set selection.
- The floor is additionally capped so the pool never drops below `num_views` candidates.

Observations below the floor are still reported as accepted (they passed pre-filtering) but are **excluded from the embedding selection set** and marked `below_quality_floor` in `quality.csv`. This guarantees the selected views meet a minimum quality while still allowing enough candidates through for diversity-aware selection.

## Relationship

```
individual metrics  ──weighted sum──▶  score  (aggregate quality)
                                       confidence  (weakest link, lower bound)
```

- **Score** answers: *"How good is this view overall?"*
- **Confidence** answers: *"Is there any dimension that makes this view risky?"*
