# VincentsArtifactsFilter (`vincents_artefacts`)

**Source:** `preprocessing/vincents_artefacts.py`<br>
**Ported from:** [`score_mask_artifacts`](nit_view_selection/select_best_views.py)<br>
**Kind:** soft filter — derives a quality score in `[0, 1]`, and simultaneously acts as a working pre-filter with the two
`BaseFilter` rejection criteria implemented in `evaluate`.<br> 
**Not** part of the default `filter_order` -> used as a diagnostic and, optionally, as a rejection layer.

## Purpose

Penalises speckled, holed, or ragged-edge masks. Noisy masks are bad for
downstream embedding extraction, which cuts out the masked region. Masks with
high artifact fractions are dropped outright (absolute ceiling + population outlier), 
the rest are ranked by the derived selection weight.

## Algorithm

1. **Raw statistical Value** (per observation)<br> 
   It reflects the <b>artifact fraction</b>:

    ```python
    artifact_mask = open(mask) XOR close(mask)          # elliptical kernel, kernel_size
    vincent_artifact_fraction = |artifact_mask| / mask_pixels
    ```

    High values indicate noisy, speckled, or ragged-edge masks.
    This is implemented in function `compute_stat`.

2. **Quality scaled stat**<br> 
    The reported score is the raw stat inverted and normalised so 0.0 = max artifacts:

    ```python
    score = clip(1 - stat / max_fraction, 0.0, 1.0)
    ```

3. **Threshold-based Filter** (absolute)<br> 
    If `hard_max_fraction > 0` and the statistical value meets or exceeds it, the
    observation is rejected outright with reason `vincents_artefacts_threshold`.
    This catches masks whose artifact fraction is completely unusable regardless of population.

4. **Population-based Filter** (relative)<br> 
    Robust median/MAD z-score of the raw stat, fit once over the population
    (`fit`, only when `outlier_z` is set):

    ```python
    z = (stat - median) / robust_scale
    z >= outlier_z        -> reject with reason vincents_artefacts_outlier
    ```

    Artifact fraction has a right-skewed distribution — the "high_bad" (artifacts) tail
    is where the noticeably-bad outliers live.

5. **Population pass** (selection weight)<br> 
    Same robust median/MAD one-sided half-Gaussian falloff as `VincentsMotionBlurFilter`, penalising the "bad" (high = artefacted) side:

    ```python
    weight = exp(-0.5 * (z / softness)^2)
    z = (stat - median) / robust_scale
    ```

    For artifact fraction it is symmetric around the population median — low_artifact and high_artifact are equally penalised by `softness`.
    This is implemented in function `fit_weights` (inherited from `VincentSoftFilter`).

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_artifact_fraction` | raw stat: artifact pixels / mask pixels |
| `metrics.vincents_artefacts` | fitted weight in `(0, 1]` |

## Score and rejection

`evaluate` returns `(score, passed, reason)` with `score = quality scaled stat`
and implements the two rejection criteria from `BaseFilter`:

- **Threshold-based — Absolute Garbage Rejection** <br>
   ```
    stat >= hard_max_fraction  →  (0.0, False, "vincents_artefacts_threshold")
    ```
    A mask above the artifact ceiling is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection** <br>
    ```
    z >= outlier_z  →  (score, False, "vincents_artefacts_outlier")
    ```
    Requires the `fit(observations)` population pass (run automatically by
    the pipeline when `need_fitting()` is true); the robust median/MAD
    are taken from the raw stat distribution.

- **Pass** <br>
    ```
    (score, True, "vincents_artefacts")
    ```

Both rejection criteria are the shared `ScoreFilter` implementation (see
`preprocessing/base.py` + `preprocessing/filter_utils.py`).

## Configuration (`VincentsArtifactsConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute the stat and weight |
| `kernel_size` | `3` | Morphology kernel for artifact detection |
| `hard_max_fraction` | `0.15` | Absolute hard-reject ceiling on the raw fraction |
| `max_fraction` | `0.05` | Artifact fraction at which the score hits 0.0 |
| `outlier_z` | `3.0` | Robust population-outlier cutoff on the raw stat (`fit`/`evaluate`) |

`hard_max_fraction` and `outlier_z` are forwarded from config by
`build_filters`, so the configured values are active in the pipeline.

## Notes

- The raw stat is always recorded even when a rejection criterion trips, so
   downstream diagnostics and the weight pass see it.
- If `pixel_count <= 0`, the stat is `0.0` (zero artifacts).
- `VincentsArtifactsQuality` scores the same statistic against a fixed global anchor (`max_fraction = 0.05`).
