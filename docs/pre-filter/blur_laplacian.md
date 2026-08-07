# BorderLaplacianBlurFilter (`blur_laplacian`)

**Source:** `preprocessing.legacy.blur_filter.py`<br>
**Kind:** hard pre-filter — derived from a raw boundary-band sharpness stat,
simultaneously producing the `(0, 1]` goodness score and implementing both
rejection criteria mandated by `BaseFilter`.
Also available as its own class in `preprocessing/border_blur_filter.py`.<br>

Not part of the default `filter_order` used for scoring — it is a working pre-
filter: frames whose object boundary fails either rejection criterion are
dropped outright.

## Purpose

Measures the sharpness of the **object boundary** via variance of the Laplacian
restricted to a boundary band (`dilate XOR erode` of the mask). The same
boundary-band approach is used by `VincentsMotionBlurFilter`, so both share the
same raw stat on `metrics.vincent_boundary_blur_variance`.

Combined with Tenengrad inside `BlurFilter`: both metrics are computed per frame,
each enforcing its own garbage floor and outlier removal. The combined score is
`0.5 × lap_score + 0.5 × ten_score`; the Laplacian threshold path alone can
trigger a rejection.

## Algorithm

1. **Raw statistical value (per observation)**<br>  
   Boundary-band Laplacian variance, computed via `compute_boundary_blur_variance`:

    ```python
    band       = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width
    laplacian  = Laplacian(gray, ksize=3)[band].var()
    ```

   This is implemented in the method `compute_stat` (same helper used by  
   `VincentsMotionBlurFilter`).

2. **Quality-scaled score**<br>  
   The raw stat divided against a fixed global anchor so it is comparable across
   datasets:

    ```python
    lap_score     = min(laplacian / max_variance, 1.0)       # max_variance default 10000
    ```

3. **Threshold-based filter (absolute garbage rejection)**<br>  
   If `hard_min_variance > 0` and the Laplacian stat falls below it, the frame
   is rejected outright with reason `blur_threshold`. This catches frames whose
   object boundary variance is unusably low regardless of the population:

    ```python
    laplacian < hard_min_variance  →  (score, False, "blur_threshold")
    ```

4. **Population-based filter (relative outlier rejection)**<br>  
   Robust median / MAD z-score of the Laplacian stat, fit once over the population
   (`fit`, only when `outlier_z` is set):

    ```python
    z = (laplacian - median) / robust_scale
    z <= -outlier_z        →  reject with reason "blur_outlier"
    ```

   Boundary sharpness is a continuous spectrum, so the "low_bad" (blurred) tail
   is where the noticeably-bad outliers live.

5. **Combined pass**<br>  
   When both Laplacian and Tenengrad components of `BlurFilter.evaluate` pass:

    ```python
    score  = 0.5 × lap_score + 0.5 × ten_score
    return (score, True, "blur")
    ```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.laplacian` | raw stat: boundary-band Laplacian variance |
| `metrics.vincent_boundary_blur_variance` | same raw stat, shared with `VincentsMotionBlurFilter` |

## Score and rejection

Inside `BlurFilter.evaluate` the Laplacian component feeds the combined score
and can independently trigger threshold-based rejection. The full method
signature returns`(score, passed, reason)` with:

- **Threshold-based — Absolute Garbage Rejection**<br>  
    ```
    laplacian < hard_min_variance  →  (combined_score, False, "blur_threshold")
    ```
   A smeared boundary below the floor is unusable regardless of the population.

- **Population-based — Relative Outlier Rejection**<br>  
    ```
    z <= -outlier_z  →  (score, False, "blur_outlier")
    ```
   Requires the `fit(observations)` population pass; the robust median/MAD are
   taken from the Laplacian-stat distribution.

- **Combined Pass**<br>  
    When both Laplacian and Tenengrad components pass:
    ```
    combined_score = 0.5 × lap_score + 0.5 × ten_score
    (combined_score, True, "blur")
    ```

## Configuration (`BlurFilter`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Compute both stats and weights |
| `stroke_width` | `9` | Boundary-band stroke width (px) — shared by Laplacian and Tenengrad |
| `max_variance` | `10000.0` | Laplacian variance at which the Laplacian score hits 1.0 |
| `hard_min_variance` | `120.0` | Absolute floor on raw Laplacian variance (`0` disables) |
| `outlier_z` | `None` | Robust outlier cutoff — on both Laplacian and Tenengrad stats |
| `threshold_min` | `None` | Reserved for compatibility in the combined filter score path |

## Notes

- The Laplacian raw stat is always published, even when it trips rejection, so
  downstream diagnostics and scoring see it.
- If `observation.image is None`, the Laplacian stat is `0.0`.
- `VincentsMotionBlurFilter` computes the identical boundary-band Laplacian
  variance (`compute_boundary_blur_variance`) but as a selection weight rather
  than a goodness score.
