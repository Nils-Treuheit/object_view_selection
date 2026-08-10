# Plan: Improve Diversity Embedding for kMeans (descriptor-based diversity + rotational embeddings)

> **Status: PLAN ONLY — no implementation.** This document explores two ways to make
> object views land in a more useful semantic space before kMeans clustering:
> (1) descriptor-based diversity blended into the quality-diversity selector, and
> (2) rotational embeddings that make the learned (DINOv3/SigLIP2) features rotation-aware.
> Nothing here is implemented; this is a design proposal to review before coding.

---

## 1. Motivation

The `top_kmeans_xnn` selector clusters object views with kMeans in the embedding
space produced by a frozen vision backbone. Two weaknesses show up in practice:

1. **Learned embeddings are scale/shape agnostic at the cluster level.**
   DINOv3/SigLIP2 are trained for semantic understanding (object identity, parts),
   not for *pose/viewpoint* discrimination. Two views of the same object taken from
   nearby cameras often land closer together in the embedding than two *different*
   views that would be far more informative for a reconstruction/onboarding task.
   kMeans then collapses near-duplicate views into one cluster.

2. **The learned embeddings are not rotation-equivariant.**
   Rotating the object in the image can push a view to a distant region of the
   embedding space even though it adds no new information, while a genuinely novel
   angle stays embedded close to a already-seen one.

Both issues dilute the "diversity" signal that selection algorithms rely on, so the
selected views are less diverse (in a geometric/silhouette sense) than the embedding
space suggests.

---

## 2. Existing state

### 2.1 What is already implemented

- `selection/greedy_quality_diversity.py` — GQD selector supporting:
  - `diversity_mode` = `"min" | "max" | "prototype"` (how distance to the selected set is aggregated),
  - `use_descriptors` / `descriptor_weight` — blends a **descriptor divergence** matrix
    with the embedding cosine distance in the diversity term.
- `descriptors/silhouette.py` — `silhouette_descriptor(mask, size=64)` (binarize → bbox crop →
  square pad → resize → L2-normalise) and `silhouette_divergence(a, b)` (cosine distance).
- `run.py` — `--selector_use_descriptors`, `--selector_descriptor_weight`,
  `--selector_descriptor` (silhouette | hu | zernike | fourier | shape_context),
  `--selector_diversity_mode`; descriptor scores computed in `run_pipeline` and passed to `selector.select(...)`.
- `selection/kmeans_xnn.py` — kMeans over embeddings with `kmeans_init`
  (`best_quality` | `farthest`) and `kmeans_xnn_k` neighbourhood.
- `config.py` — `PipelineConfig` fields: `selector_diversity_mode`, `selector_use_descriptors`,
  `selector_descriptor_weight`, `selector_descriptor`, plus the kMeans flags.

### 2.2 Where the plan would plug in

| Stage | File | Hook point |
|-------|------|------------|
| Descriptor computation | `run.py` → `extract_shape_descriptor` | already returns per-view descriptor vectors |
| Diversity in selection | `selection/greedy_quality_diversity.py` | `silhouette_scores` arg already wired |
| kMeans input space | `selection/kmeans_xnn.py` | currently receives only `embeddings` |
| Embedding extraction | `run.py` → `extract_embeddings` | produces `obs.embedding` from the backbone |

---

## 3. Option A — Descriptor-based diversity for GQD (partially implemented)

### 3.1 What is already done

The GQD selector can already blend a per-view **silhouette divergence** matrix into its
diversity term:

```
score[i] = alpha · quality[i]
         + beta · [ (1 - w) · min_cos(emb_i, selected)
                    + w · min_cos(desc_i, selected) ]
```

With `w = 1.0` the embedding space is ignored for diversity and the selector picks purely
by silhouette dissimilarity (useful for reconstruction/onboarding where view geometry matters).
With `0 < w < 1` both spaces contribute.

### 3.2 What is NOT yet implemented (candidate follow-ups)

- **Descriptor-weighted kMeans.** `kmeans_xnn.py` still clusters on raw embeddings only.
  Proposal: allow a blended feature
  `f_i = concat((1-w)·normalise(emb_i), w·normalise(desc_i))` before kMeans (optionally
  whitened). This makes clusters separate on both semantics and geometry.
- **More descriptor families** (beyond silhouette):
  - **Zernike moments** (`descriptors`-style module): rotation-invariant image moments —
    good for global shape; loses pose information on purpose.
  - **Fourier boundary descriptors**: order-normalised Fourier coefficients of the contour
    (works from the mask; same input as silhouette).
  - **Hu moments**: classic invariant moments; cheap but less discriminative.
  - **HOG over the mask/object-hand**: captures pose/limb layout.
- **Per-cluster prototype descriptor**: after kMeans, report each cluster's mean
  silhouette/descriptor vector in the explorer so the user sees *why* views grouped.

### 3.3 Verification plan

- Unit tests asserting blended-diversity changes cluster membership vs pure embedding kMeans
  on a synthetic dataset (e.g. 4 rotating silhouettes + 4 near-duplicate semantic views).
- Metric: silhouette coefficient of the chosen views *in each space* (embedding, descriptor),
  plus a "novel view" heuristic — mean descriptor divergence within the selected set.

---

## 4. Option B — Rotational embeddings (bigger change, not implemented)

### 4.1 Idea

Make the backbone features rotation-aware by combining a base embedding with an explicit
**rotation estimate**, so kMeans/GQD distances reflect *viewpoint* differences, not just
semantic identity.

### 4.2 Candidate approaches

1. **Image-level rotational augmentation embedding.**
   Extract `e = backbone(image)` for the image and for K rotated copies
   (0°, 90°, 180°, 270°), then build a rotation-aligned descriptor:
   ```
   e_rot = concat( R(0)·e0 , R(90)·e1 , ... )   (R = rotation of feature map / re-sort of tokens)
   ```
   Discriminative in *relative* orientation. Cost: K forward passes.

2. **Relative-rotation head (RotNet-style).**
   Train a small MLP on the backbone features to predict the rotation angle between
   two views of the same object; use `|Δθ|` as an extra diversity signal added to the
   GQD/kMeans distance. Requires a per-object paired-view training set (available in
   the dataset: all views are already aligned by filename).

3. **Canonical-pose alignment (lift to a canonical frame).**
   Use the silhouette/`object_hands` segmentation to rotate the mask into a canonical
   orientation (PCA of the mask or the hands skeleton), then embed the aligned image.
   Two views of the same pose collapse to the same embedding; genuinely different poses
   stay separated. This is the most "physics-aware" option and works on the mask without
   any training.

### 4.3 Trade-offs

| Approach | Training needed | Extra forward passes | Pose discriminative | Risks |
|----------|----------------|----------------------|---------------------|-------|
| Rotational augmentation | none | K (e.g. 4) | relative only | 4× embedding cost; absolute orientation ambiguous |
| RotNet head | small (per object or dataset) | 0 (inference) | absolute/relative | needs labels; may overfit to the training set |
| Canonical-pose alignment | none | 0 | absolute (via mask) | depends on mask/hands quality; PCA sign ambiguity |

### 4.4 Verification plan

- Ablation on a held-out object: for each of 3 views with known angular separation,
  measure embedding cosine distance before/after the rotation-aware step; expect
  monotonic increase of distance with angular separation.
- Re-run the `top_kmeans_xnn` selection with and without the rotation-aware embedding;
  compare the selected set's mean silhouette-divergence (from Option A) — the rotation-aware
  set should be strictly more diverse.

---

## 5. Recommended sequencing

1. **Land Option A for kMeans** (blended descriptor features in `kmeans_xnn.py`) — small,
   well-contained, reuses the already-wired silhouette descriptors.
2. **Add the explorer view** of per-cluster prototype descriptors (visibility of the change).
3. **Prototype Option B3 (canonical-pose alignment)** on the `object_hands`/mask data — no
   training, most likely to produce a *useful* rotation-aware space for viewpoint-diverse
   selection. If its silhouette-divergence gain is large, then evaluate B1/B2 for
   multi-view reconstruction tasks.

---

## 6. Open questions for review

- Should descriptor blending in kMeans be a separate `--kmeans_use_descriptors` flag or reuse
  `--selector_use_descriptors`? (Suggest a separate flag; the two stages have different goals.)
- Should the rotational embedding be a *new embedding type* (`--embedding rot_siglip2`) or an
  *augmentation layer* applied after any backbone? (Suggest the latter — composition is cleaner.)
- Does the exploratory dataset have enough views per object (with known relative pose) to train
  a RotNet head? If not, drop Option B2.
