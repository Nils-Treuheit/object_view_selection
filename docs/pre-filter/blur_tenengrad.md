# BorderTenengradBlurFilter (`blur_tenengrad`)

**Source:** `preprocessing/blur_filter.py`<br>
**Kind:** hard pre-filter component within `BlurFilter` — computes the mean
boundary-band Sobel magnitude (Tenengrad), produces a `(0, 1]` goodness score,
and enforces both rejection criteria mandated by `BaseFilter`.
Also available as its own standalone class in `preprocessing/border_blur_filter.py`.

Not part of the default `filter_order` used for scoring — it is a working pre-
filter: frames whose boundary Tenengrad fails either criterion are dropped
outright.

## Purpose

Complementary boundary-band sharpness measure to
[`BorderLaplacianBlurFilter`](blur_laplacian.md). The Laplacian variance measures
overall boundary sharpness; the Tenengrad (mean Sobel magnitude) responds to
*structured gradients*, so the two detect complementary blur modes. Both run
inside `BlurFilter` with independent garbage floors and outlier rejection.

Combined inside `BlurFilter`: each metric enforces its own thresholds and
population-based outlier removal, then the combined score is
`0.5 × lap_score + 0.5 × ten_score`.

## Algorithm

1. **Raw statistical value (per observation)**<br>  
   Boundary-band mean Sobel magnitude, computed via `compute_boundary_tenengrad`:

    ```python
    band          = dilate(mask) XOR erode(mask)       # elliptical kernel, stroke_width
    gradient      = sqrt(SobelX(gray)^2 + SobelY(gray)^2)
    tenengrad     = gradient[band].mean()
    ```

   This is implemented in the internal method `_eval_tenengrad` of `BlurFilter`.

2. **Quality-scaled score**<br>  
   The raw stat divided against a fixed global anchor so it is comparable across
   datasets:

    ```python
    ten_score     = min(tenengrad / max_tenengrad, 1.0)     # max_tenengrad default 100
    ```

3. **Threshold-based filter (absolute garbage rejection)**<br>  
   If `hard_min_tenengrad > 0` and the Tenengrad stat falls below it, the frame
   is rejected outright with reason `blur_threshold`. This catches frames whose
   boundary structured gradients are unusably weak regardless of the population:

    ```python
    tenengrad < hard_min_tenengrad  →  (score, False, "blur_threshold")
    ```

4. **Population-based filter (relative outlier rejection)**<br>  
   Robust median / MAD z-score of the Tenengrad stat, fit once over the
   population (`fit`, only when `outlier_z` is set):

    ```python
    z = (tenengrad - median) / robust_scale
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
| `metrics.tenengrad` | raw stat: mean boundary-band Sobel magnitude |
| `metrics.bound_tenengrad` | same raw stat (legacy alias) |

## Score and rejection

Inside `BlurFilter.evaluate` the Tenengrad component feeds the combined score
and can independently trigger threshold-based rejection. The full method
signature returns `(score, passed, reason)` with:

- **Threshold-based — Absolute Garbage Rejection**<br>  
    ```
    tenengrad < hard_min_tenengrad  →  (combined_score, False, "blur_threshold")
    ```
   Weak structured gradients below the floor are unusable regardless of the population.

- **Population-based — Relative Outlier Rejection**<br>  
    ```
    z <= -outlier_z  →  (score, False, "blur_outlier")
    ```
   Requires the `fit(observations)` population pass; the robust median/MAD are
   taken from the Tenengrad-stat distribution.

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
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `max_tenengrad` | `100.0` | Tenengrad at which the Tenengrad score hits 1.0 |
| `hard_min_tenengrad` | `35.0` | Absolute floor on raw Tenengrad (`0` disables) |
| `outlier_z` | `None` | Robust outlier cutoff — on both Laplacian and Tenengrad stats |

## Notes

- The Tenengrad raw stat is always published, even when it trips rejection, so
  downstream diagnostics and scoring see it.
- If `observation.image is None`, the Tenengrad stat is `0.0`.
- `VincentsMotionBlurFilter` uses the same boundary-band construction but on
  the Laplacian rather than Sobel gradients.
