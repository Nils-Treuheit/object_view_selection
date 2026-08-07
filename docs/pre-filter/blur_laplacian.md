# BorderLaplacianBlurFilter (`blur_laplacian`)

**Source:** `preprocessing/border_blur_filter.py`
**Kind:** score filter — always passes, returns a 0..1 goodness score; rejection
is layered on top by `FilterVariant`.

## Purpose

Measures the sharpness of the **object boundary** rather than the whole image.
Blur is most damaging exactly at the object/background transition, and a
whole-image measure would be dominated by background texture while a
whole-mask measure would be diluted by low-texture object interiors.

## Algorithm

1. Build the boundary band straddling the mask contour:

   ```
   band = dilate(mask) XOR erode(mask)     # elliptical kernel, stroke_width px
   ```

2. Compute the variance of the Laplacian of the grayscale image **restricted
   to the band** (helper `compute_boundary_blur_variance`):

   ```
   variance = Laplacian(gray, ksize=3)[band].var()
   ```

   Higher variance = sharper object/background transition.

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.laplacian` | boundary-band Laplacian variance (same value as below) |
| `metrics.vincent_boundary_blur_variance` | same variance, shared with `VincentsMotionBlurFilter` |

## Score and rejection

```
score  = min(variance / max_variance, 1.0)
passed = True always from the filter itself
reason = "blur_laplacian"                       (inner, informational)
      | "blur_laplacian_threshold"              (score < threshold_min)
      | "blur_laplacian_outlier"                (population-relative z <= -outlier_z)
```

The two rejection modes come from the `FilterVariant` wrapper, not from the
filter itself. See [variants.md](variants.md).

If `observation.image is None`, the filter passes immediately (score 1.0).

## Configuration (`LaplacianBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `max_variance` | `10000.0` | Variance at which the score hits 1.0 |
| `threshold_min` | `0.01` | Very relaxed absolute floor below which = awful quality |
| `outlier_z` | `3.0` | Robust population-outlier cutoff |

## Notes

- The score feeds the quality scorer indirectly: `BorderBlurQuality` reads the
  pre-computed `metrics.laplacian` (falling back to computing it directly when
  absent).
- `VincentsMotionBlurFilter` computes the same statistic but as a soft
  population weight instead of a score.
