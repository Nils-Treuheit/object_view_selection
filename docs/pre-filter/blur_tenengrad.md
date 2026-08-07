# BorderTenengradBlurFilter (`blur_tenengrad`)

**Source:** `preprocessing/border_blur_filter.py`
**Kind:** score filter — always passes, returns a 0..1 goodness score; rejection
is layered on top by `FilterVariant`.

## Purpose

Complementary boundary-band sharpness measure to
[`BorderLaplacianBlurFilter`](blur_laplacian.md). The Laplacian variance
measures overall boundary sharpness; the Tenengrad (mean Sobel-magnitude)
responds to *structured* gradients, so the two detect complementary blur modes.

## Algorithm

1. Build the boundary band as `dilate(mask) XOR erode(mask)` (elliptical
   kernel, `stroke_width` px).
2. Compute the mean Sobel magnitude restricted to the band (helper
   `compute_boundary_tenengrad`):

   ```
   gradient = sqrt(SobelX(gray)^2 + SobelY(gray)^2)
   tenengrad = gradient[band].mean()
   ```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.tenengrad` | mean boundary-band Sobel magnitude |

## Score and rejection

```
score  = min(tenengrad / max_tenengrad, 1.0)
passed = True always from the filter itself
reason = "blur_tenengrad"
      | "blur_tenengrad_threshold"
      | "blur_tenengrad_outlier"
```

The `_threshold` / `_outlier` rejections come from the `FilterVariant`
wrapper. See [variants.md](variants.md).

If `observation.image is None`, the filter passes immediately (score 1.0).

## Configuration (`TenengradBlurConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |
| `stroke_width` | `9` | Boundary-band stroke width (px) |
| `max_tenengrad` | `100.0` | Tenengrad at which the score hits 1.0 |
| `threshold_min` | `0.10` | Very relaxed absolute floor below which = awful quality |
| `outlier_z` | `3.0` | Robust population-outlier cutoff |

## Notes

- Runs right after `blur_laplacian` in the default order.
- The score uses a fixed global anchor (`max_tenengrad`), keeping it
  comparable across datasets; only the rejection is population-relative.
