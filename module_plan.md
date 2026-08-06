# Object View Selection Pipeline

A modular and configurable pipeline for selecting the best **N image/mask pairs** that maximize **object identifiability**, rather than simply selecting the highest-quality masks.

The pipeline is designed to be easily extensible, allowing different filtering strategies, embedding models, and subset selection algorithms to be swapped without changing the overall architecture.

---

# Dataset Structure

The input dataset consists of a single object directory:

```text
bottle/
│
├── images/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
│
├── masks/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
│
└── object_hands/
    ├── 00000.png
    ├── 00001.png
    └── ...
```

All files are aligned by filename.

Example:

| Image | Mask | Object-Hand |
|--------|------|-------------|
| 00042.png | 00042.png | 00042.png |

---

# Overall Pipeline

```text
Load Dataset
      │
      ▼
Pre-filter Observations
      │
      ▼
Compute Quality Scores
      │
      ▼
Extract Object Embeddings
      │
      ▼
Subset Selection
      │
      ▼
Save Results
```

---

# Project Structure

```text
object_view_selection/
│
├── run.py                          # Pipeline entry point
├── config.py                       # Global configuration
│
├── io/
│   ├── dataset.py                  # Dataset loader
│   └── observation.py              # Observation dataclass
│
├── preprocessing/
│   ├── blur.py
│   ├── truncation.py
│   ├── area.py
│   ├── confidence.py
│   ├── occlusion.py
│   ├── completeness.py
│   └── filter.py
│
├── quality/
│   ├── metrics.py
│   └── quality_score.py
│
├── embeddings/
│   ├── base.py
│   ├── crop.py
│   ├── dinov2.py
│   ├── siglip.py
│   ├── clip.py
│   └── eva_clip.py
│
├── descriptors/
│   ├── hu.py
│   ├── zernike.py
│   ├── fourier.py
│   └── shape_context.py
│
├── selection/
│   ├── selector.py
│   ├── fps.py
│   ├── greedy_quality_diversity.py
│   ├── facility_location.py
│   ├── dpp.py
│   └── next_best_view.py
│
├── utils/
│   ├── geometry.py
│   ├── visualization.py
│   └── math.py
│
└── outputs/
```

---

# Module Overview

## IO

Responsible for loading aligned observations.

### dataset.py

Loads aligned image/mask/object-hand triples.

Returns:

```python
Observation(
    id=42,
    image=...,
    mask=...,
    object_hand=...
)
```

---

### observation.py

Contains the main data structure.

Example:

```python
Observation
├── id
├── image
├── mask
├── object_hand
├── quality
├── embedding
└── metadata
```

---

# Preprocessing

Responsible for rejecting observations that are unlikely to contribute useful information.

Pipeline:

```text
Observation
      │
      ▼
Blur Filter
      │
      ▼
Area Filter
      │
      ▼
Border Filter
      │
      ▼
Occlusion Filter
      │
      ▼
Confidence Filter
      │
      ▼
Accepted / Rejected
```

---

## blur.py

Computes image sharpness.

Possible metrics:

- Variance of Laplacian
- Tenengrad
- FFT energy

Output:

```python
blur_score ∈ [0,1]
```

---

## truncation.py

Detects objects touching the image border.

Metric:

```text
BorderIntersection =
(mask ∩ border) / mask
```

Reject if

```text
BorderIntersection > threshold
```

---

## area.py

Computes visible object size.

Metric:

```text
AreaRatio =
mask area / image area
```

Reject if

```text
AreaRatio < threshold
```

---

## confidence.py

Optional detector confidence.

Supports:

- SAM
- Mask2Former
- detector confidence

---

## occlusion.py

Estimates visible object fraction.

Possible implementations:

- hand overlap
- mask overlap
- contour irregularity
- depth discontinuity
- segmentation confidence

---

## completeness.py

Measures shape completeness.

Possible metrics:

- Solidity
- Convexity
- Extent
- Contour completeness

---

## filter.py

Chains all enabled filters together.

Configurable order:

```text
Blur
↓

Area
↓

Border
↓

Occlusion
↓

Confidence
```

---

# Quality

Responsible for ranking observations.

---

## metrics.py

Computes normalized quality metrics:

- Blur
- Area
- Confidence
- Occlusion
- Completeness

Each metric is normalized to

```text
[0,1]
```

---

## quality_score.py

Computes

```text
Q =
w_blur * blur
+ w_area * area
+ w_confidence * confidence
+ w_occlusion * occlusion
+ w_completeness * completeness
```

Higher values indicate better observations.

---

# Embeddings

Responsible for extracting object descriptors.

Pipeline

```text
Image
   │
Mask
   │
   ▼
Masked Crop
   │
   ▼
Embedding Model
   │
   ▼
Feature Vector
```

---

## crop.py

Supports

- bounding-box crop
- masked crop
- padded crop
- square crop
- resized crop

---

## base.py

Abstract embedding interface.

```python
EmbeddingModel

encode(image, mask)
```

---

## dinov2.py

DINOv2 implementation.

---

## siglip.py

SigLIP implementation.

---

## clip.py

CLIP implementation.

---

## eva_clip.py

EVA-CLIP implementation.

---

# Shape Descriptors

Alternative to learned embeddings.

Modules:

- Hu Moments
- Zernike Moments
- Fourier Descriptors
- Shape Context

Useful when object identity is silhouette-driven.

---

# Selection

Responsible for selecting the final subset.

Common interface:

```python
select(
    embeddings,
    quality_scores,
    n
)
```

---

## fps.py

Farthest Point Sampling.

Objective:

```text
maximize

min distance
```

Produces diverse viewpoints.

---

## greedy_quality_diversity.py

Recommended implementation.

Greedy optimization

```text
Score

=
α Quality

+

β Diversity
```

where diversity is

```text
minimum embedding distance
```

to the already selected set.

---

## facility_location.py

Maximizes representativeness.

Objective:

```text
Σ max similarity
```

Useful for dataset summarization.

---

## dpp.py

Determinantal Point Process.

Kernel:

```text
Lij = qi qj Kij
```

Balances

- quality
- diversity
- information gain

Most principled but computationally expensive.

---

## next_best_view.py

For datasets with camera poses.

Possible metrics:

- occupancy entropy
- NeRF uncertainty
- Gaussian Splat uncertainty
- voxel information gain

---

# Configuration

The entire pipeline is controlled through a configuration object.

Example:

```python
PipelineConfig(

    filters=FilterConfig(
        blur=True,
        truncation=True,
        area=True,
        occlusion=True,
        confidence=False
    ),

    embedding="dinov2",

    selector="quality_diversity",

    num_views=10,

    quality_weights=dict(
        blur=0.30,
        area=0.20,
        confidence=0.10,
        occlusion=0.20,
        completeness=0.20,
    )
)
```

---

# Pipeline Flow

```text
Dataset
   │
   ▼
Load Observations
   │
   ▼
Pre-filter
   │
   ▼
Compute Quality
   │
   ▼
Extract Embeddings
   │
   ▼
Subset Selection
   │
   ▼
Save Outputs
```

---

# Output Structure

```text
outputs/
│
├── selected_samples/
│   └── <obj_id>/
│       ├── rgb/
│       │   ├── 00013.png
│       │   ├── 00047.png
│       │   └── ...
│       ├── mask/
│       ├── depth/           # only when <data_root>/depth exists
│       └── hand_mask/       # only when a hand mask is available
│
├── accepted_samples/        # accepted-but-unselected tuples (--debug), same layout
│   └── <obj_id>/
│       ├── rgb/
│       ├── mask/
│       └── hand_mask/
│
├── rejected_samples/        # grouped by rejection reason:
│   └── <reason>/            #   e.g. blur, incomplete_shape, occlusion,
│       └── <obj_id>/        #   small_object, vincent_border_pixel, plus the
│           ├── rgb/         #   variant reasons <reason>_threshold / _outlier
│           ├── mask/
│           ├── depth/
│           └── hand_mask/
│
├── report.json
│
├── quality.csv
│
├── embeddings.npy
│
└── visualization.png
```

---

# Recommended Pipeline

```text
Load Dataset
      │
      ▼
Blur Filter
      │
      ▼
Area Filter
      │
      ▼
Border Filter
      │
      ▼
Occlusion Filter
      │
      ▼
Confidence Filter
      │
      ▼
Quality Score
      │
      ▼
Embedding Extraction
      │
      ▼
Greedy Quality + Diversity
      │
      ▼
Selected Views
```

---

# Advantages of the Modular Design

- Fully interchangeable embedding models (DINOv2, SigLIP, CLIP, EVA-CLIP)
- Independent preprocessing modules
- Configurable quality scoring
- Multiple subset selection algorithms under a common interface
- Easily extensible for new descriptors or filters
- Supports robotics, object recognition, dataset curation, and 3D reconstruction
- Clear separation of concerns, making the codebase easier to test, maintain, and benchmark
