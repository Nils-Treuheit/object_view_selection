# VincentsArtifactsFilter (`vincents_artefacts`)

**Source:** `preprocessing/vincents_artefacts.py`<br>
**Ported from:** [`nit_view_selection/select_best_views.py`](https://github.com/ovgu-nit/nit_object_onboarding/tree/vincis-select-best-view-sandbox/nit_view_selection)<br>
**Kind:** score filter — always passes, returns a 0..1 goodness score; rejection
is layered on top by `FilterVariant`.

## Purpose

Penalises speckled, holed, or ragged-edge masks. Noisy masks are bad for
downstream embedding extraction, which cuts out the masked region.

## Algorithm

1. Compute the artifact mask — pixels where morphological open and close
   disagree (helper `compute_artifact_mask`, elliptical kernel `kernel_size`):

   ```
   artifact_mask = open(mask) XOR close(mask)
   ```

   Opening drops small foreground specks/protrusions; closing fills small
   background holes/gaps. Where the two disagree is unstable, noisy mask
   boundary.

2. Normalize by mask size:

   ```
   vincent_artifact_fraction = |artifact_mask| / mask_pixels
   ```

## Metrics

| Field | Meaning |
|-------|---------|
| `metrics.vincent_artifact_fraction` | artifact pixels / mask pixels |

## Score and rejection

```
score  = clip(1 - fraction / max_fraction, 0, 1)
passed = True always from the filter itself
reason = "vincents_artefacts"                    (inner, informational)
      | "vincents_artefacts_threshold"           (score < threshold_min)
      | "vincents_artefacts_outlier"             (population-relative z <= -outlier_z)
```

A mask whose artifact fraction reaches `max_fraction` scores 0.0. The two
rejection modes come from the `FilterVariant` wrapper. See
[variants.md](variants.md).

Empty masks get `fraction = 0.0` (they are handled earlier by
`VincentEmptyMaskFilter` anyway).

## Configuration (`VincentsArtifactsConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Run this filter |
| `kernel_size` | `3` | Morphology kernel for artifact detection |
| `max_fraction` | `0.05` | Artifact fraction at which the score hits 0.0 |
| `threshold_min` | `0.05` | Very relaxed absolute floor below which = awful quality |
| `outlier_z` | `3.0` | Robust population-outlier cutoff |

## Notes

- Same statistic feeds the `VincentsArtifactsQuality` scorer component.
- Despite the class name, this filter does **not** hard-reject; it is a score
  filter like the two boundary-blur filters.
