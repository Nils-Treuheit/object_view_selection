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
- **`AreaQuality`** – Mask pixel fraction divided by a fixed `0.20` max fraction (hardcoded in `quality/area.py`, equal to the `quality_anchors.area_max_fraction` default), clipped to 1.0. Bigger masks score higher.
- **`VincentsArtifactsQuality`** – `clip(1 − artifact_fraction / max_fraction, 0, 1)`, where `max_fraction` comes from `quality_anchors.artifacts_max_fraction` (default 0.05): a mask whose artifact fraction reaches the anchor scores 0. It reads the `vincent_artifact_fraction` pre-filter stat when present, otherwise computes it directly from the mask (morphological open XOR close over the configured kernel).
- **`CenternessQuality`** – how centred the object's **center point** (mask centroid) is in the frame. The centroid at the exact frame centre scores the perfect 1.0. Shifting the center point inside the interior costs only a **light, quadratic decrease**; once the center point enters the **`BORDER_ZONE_PX = 20` px band** along any image border the score drops off **exponentially**, so objects whose center point grazes the frame edge get crushed.

All four components are anchored in **fixed global max/min values**, independent of the dataset, so quality is comparable across datasets. The population-relative soft weights (`vincents_area`, `vincents_motion_blur`) are still computed by the soft pre-filter pass for diagnostics but are not scorer components.

## Confidence

`confidence = blur · area · vincents_artefacts · centerness`

Confidence is the **product** of all four quality scores: it drops sharply if *any* dimension is weak (a 0 in any component zeroes the confidence) while rewarding views that are strong across the board. It is the multiplicative counterpart to the additive weighted score.

Confidence is computed **post-hoc** after all individual scores are known and is stored in `obs.metrics.confidence`. It is exported for diagnostics but not used by the scorer.

## Score (Final Quality)

`score = Σ(wᵢ · metricᵢ) / Σ wᵢ`

The final quality score is a **weighted arithmetic mean** of the 4 quality components. Weights are configured via `QualityWeights` in `config.py` (blur 0.30, area 0.20, vincents_artefacts 0.20, centerness 0.30). The result is also in [0, 1].

The score is stored in `obs.quality` and exported as both `quality` and `score` in `quality.csv`.

## Quality Floor (Selection Pool)

**OPT-IN, disabled by default** — the adaptive quality floor is *not* part of
the default pipeline. Enable it explicitly with `run.py --quality_floor` (or
`cfg.quality_floor.enabled = True`); the whole accepted pool goes to the
embedding selection otherwise.

When enabled, an **adaptive quality floor** is applied to the accepted pool before the embedding selection (`quality_floor.*` in `config.py`):

- `percentile` (default 0.10): the bottom 10% of accepted observations by quality are excluded from the selection pool.
- `absolute_min` (default 0.66): no observation below this absolute quality ever enters the selection pool.
- `minimum_pool` (default 20): the floor is capped so at least this many candidates remain for a diverse sample-set selection.
- The floor is additionally capped so the pool never drops below `num_views` candidates.

Observations below the floor are still reported as accepted (they passed pre-filtering) but are **excluded from the embedding selection set** and marked `below_quality_floor` in `quality.csv`. This guarantees the selected views meet a minimum quality while still allowing enough candidates through for diversity-aware selection.

## Relationship

```
individual metrics  ──weighted sum──▶  score  (aggregate quality)
                    ──product───────▶  confidence
```

- **Score** answers: *"How good is this view overall?"* (weighted arithmetic mean).
- **Confidence** answers: *"How sure are we about this view?"* (product — any weak dimension drags it down hard).

Both are written to `quality.csv` together with each of the 4 individual scores.
