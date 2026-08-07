# Scoring

All quality scores are normalised to **[0, 1]**, where **1.0 = perfect** (ideal candidate) and **0.0 = worst** (unusable candidate). This makes individual dimensions comparable and the multi-metric aggregation well-behaved.

## Individual Quality Metrics

| Metric | Range | Higher meaning |
|---|---|---|
| `blur` | [0, 1] | Sharp object boundary (boundary-band Laplacian) |
| `area` | [0, 1] | Object occupies a large fraction of the frame |
| `vincents_artefacts` | [0, 1] | Mask free of speckle/holes/ragged edges |
| `centerness` | [0, 1] | Mask centred in the frame |

Each is computed by a dedicated `QualityMetric` implementation in `quality/`:

- **`BorderBlurQuality`** – boundary-band Laplacian variance divided by a fixed global anchor (`quality_anchors.blur_max_variance`, default 10000), clipped to 1.0. It reads the `laplacian` pre-filter stat when present; if absent (e.g. a standalone scorer), it computes the boundary-band variance directly from the image and mask (stroke width 9). The anchor is a dataset-independent constant so sharpness scores are comparable across datasets.
- **`AreaQuality`** – Mask pixel fraction divided by `quality_anchors.area_max_fraction` (default 0.20), clipped to 1.0.
- **`VincentsArtifactsQuality`** – `clip(1 − artifact_fraction / quality_anchors.artifacts_max_fraction, 0, 1)`: a mask whose artifact fraction reaches the anchor (default 0.05) scores 0.
- **`CenternessQuality`** – how centred the mask is in the frame, computed from the mask centroid vs. the frame centre. A centred object scores 1.0.

All four components are anchored in **fixed global max/min values**, independent of the dataset, so quality is comparable across datasets. The population-relative soft weights (`vincents_area`, `vincents_motion_blur`) are still computed by the soft pre-filter pass for diagnostics but are not scorer components.

## Confidence

`confidence = min(blur, area, vincents_artefacts, centerness)`

Confidence is the **weakest-link** score: it reflects the worst-performing quality dimension for a given view. A view with high confidence is strong in every respect; a low confidence immediately identifies the binding constraint.

Confidence is computed **post-hoc** after all individual scores are known and is stored in `obs.metrics.confidence`. It is exported for diagnostics but not used by the scorer.

## Score (Final Quality)

`score = Σ(wᵢ · metricᵢ) / Σ wᵢ`

The final quality score is a **weighted arithmetic mean** of the 4 quality components. Weights are configured via `QualityWeights` in `config.py` (blur 0.30, area 0.20, vincents_artefacts 0.20, centerness 0.30). The result is also in [0, 1].

The score is stored in `obs.quality` and exported as both `quality` and `score` in `quality.csv`.

## Quality Floor (Selection Pool)

Before the embedding selection, an **adaptive quality floor** is applied to the accepted pool (`quality_floor.*` in `config.py`):

- `percentile` (default 0.10): the bottom 10% of accepted observations by quality are excluded from the selection pool.
- `absolute_min` (default 0.66): no observation below this absolute quality ever enters the selection pool.
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
