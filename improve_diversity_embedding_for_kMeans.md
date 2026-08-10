# Plan: Improve Diversity for kMeans Selection (silhouette-relative-scaled diversity)

> **Status: PLAN ONLY — no implementation.** The approach is decided: **relative
> scaling with a silhouette-descriptor divergence score** applied to the
> `top_kmeans_xnn` selector. We process kMeans clusters in order of **descending
> average quality**, start each group's pick from its **highest-quality xNN
> candidate**, then **scale the embedding-cosine diversity term by the (relative)
> silhouette divergence** to the already-picked views before scoring the next
> group. Nothing here is implemented yet.

---

## 1. Decision

For the kMeans selection improvement we **do NOT** take the rotational-embedding
route and we **do NOT** additively blend descriptors into the global GQD score.
Instead we keep the frozen learned embeddings as the clustering space and inject
the silhouette-descriptor divergence **multiplicatively** as a *relative scaling
factor* on the diversity term, cluster by cluster:

```
scaled_diversity(i) = relative_divergence(i)  ·  embedding_cosine_distance(i, selected)
```

- `relative_divergence(i)` ∈ (0, 1] — the silhouette divergence of candidate `i`
  to the already-picked views, normalized across its own xNN candidate group
  (so the *most* divergent candidate in the group keeps the full diversity score,
  and a candidate whose silhouette matches an already-picked view is down-weighted).
- `embedding_cosine_distance(i, selected)` — the embedding-space diversity term
  (distance to the nearest selected view), exactly as the current selector uses it.

The rationale for multiplying rather than adding: the learned embeddings are
"semantically" similar for near-duplicate poses, so an embedding distance can be
*large or small for the wrong reasons*. The silhouette divergence provides a
relative correction factor that only matters when it is small — it can suppress
a candidate whose geometry repeats an already-chosen view, but it never inflates
diversity on its own. Addition would let the descriptor dominate; multiplication
keeps the embedding space in charge and uses the descriptor to *reweight* it.

---

## 2. Motivation

The `top_kmeans_xnn` selector clusters object views with kMeans in the embedding
space of a frozen vision backbone (DINOv3/SigLIP2), then picks the best-quality
sample in the `xNN` neighbourhood of each centroid. Two weaknesses show up in
practice:

1. **Learned embeddings underweight viewpoint/geometry.** They are trained for
   semantic identity, not pose. Near-duplicate views cluster together while
   genuinely novel angles may sit close in the embedding. The xNN neighbourhood
   can therefore contain views that are *geometrically* redundant even though
   their embedding distance looks "diverse".
2. **kMeans order is quality-blind for diversity.** Clusters are ranked by
   average quality, but the within-group pick only looks at quality + plain
   embedding distance. It never checks whether the next pick's *shape* repeats an
   already-selected view.

Silhouette descriptors (`descriptors/silhouette.py`) are cheap (binarize → bbox
crop → resize → L2-normalize), rotation-information-preserving, and computed from
the same masks the pipeline already produces — so they give us a per-view
*geometric* signal for free. This plan uses them as a **relative scale** on the
embedding diversity term of the kMeans-xNN selection.

---

## 3. Existing state (what this builds on)

Already implemented and wired:

| Piece | File | Status |
|-------|------|--------|
| `silhouette_descriptor(mask, size=64)` (binarize → bbox crop → square pad → resize → L2-norm 4096-d) | `descriptors/silhouette.py` | implemented |
| `silhouette_divergence(a, b)` = cosine distance; `0.0` on empty/norm-0 | `descriptors/silhouette.py` | implemented |
| `extract_shape_descriptor(obs, "silhouette")` returns the descriptor vector | `run.py` | implemented |
| `TopKMeansXNN(init=..., k=..., xnn_k=...)` — kMeans + per-cluster xNN + best-quality pick + fill-up | `selection/kmeans_xnn.py` | implemented (unchanged by this plan) |
| Descriptor scores computed in `run_pipeline` for the GQD path | `run.py` | implemented (GQD only, see §6) |

The plan **extends `TopKMeansXNN`** with an optional diversity-scaling mode. The
existing behaviour must remain the default so current runs/tests keep passing.

---

## 4. The algorithm in detail

### 4.1 Inputs

- `embeddings` — (N, D) pool embeddings (learned, unchanged).
- `quality` — (N,) per-view quality scores.
- `silhouettes` — (N, d) per-view silhouette descriptor vectors.
- `k` — number of kMeans clusters (default `num_views`).
- `xnn_k` — xNN neighbourhood radius (default 10).
- `alpha`, `beta` — quality vs diversity weights (reuse `selector_alpha`/`selector_beta`).

### 4.2 Step-by-step

1. **Cluster.** Run kMeans on `embeddings` → `k` clusters `C_1..C_k` with centroids `μ_j`.
2. **Rank clusters.** Compute each cluster's average quality `Q_j = mean(quality in C_j)`;
   sort clusters in **descending** order of `Q_j` → processing order.
3. **First group (highest average quality).** Build the xNN candidate set of cluster `C_1`:
   `G_1 = {μ_1} ∪ xNN(μ_1)` (the x pool samples nearest to `μ_1` in embedding cosine
   distance). **Pick the highest-quality candidate in `G_1`** as the first selection. `P = {p_1}`.
4. **For each subsequent cluster `C_j` in rank order** (i.e. the "next highest xNN group"):
   a. Build `G_j = {μ_j} ∪ xNN(μ_j)`.
   b. For every candidate `i ∈ G_j` (excluding any already-selected pool index):
      - **Divergence:** `div(i) = mean over p ∈ P of silhouette_divergence(s_i, s_p)`
        (candidate's geometric distance to the set of already-picked views).
      - **Relative scale:** `r(i) = div(i) / max_{i' ∈ G_j} div(i')`  (relative scaling;
        the most-divergent candidate in the group keeps scale `1.0`, values ∈ (0, 1]).
      - **Embedding diversity:** `d_emb(i) = min over p ∈ P of cosine_dist(e_i, e_p)`
        (nearest already-picked view in the embedding space).
      - **Scaled diversity:** `D(i) = r(i) · d_emb(i)`.
      - **Score:** `score(i) = alpha · quality(i) + beta · D(i)`.
   c. **Pick `argmax_i score(i)`**, add to `P`.
5. **Fill-up** (only if `num_views > k`): after all `k` clusters contributed one pick,
   draw the remaining best-quality samples per cluster (highest average quality first),
   skipping already-selected indices — the existing fill-up logic unchanged.

If `num_views <= k`, stop after processing the top `num_views` clusters.

### 4.3 Pseudocode

```python
from descriptors.silhouette import silhouette_divergence
from sklearn.metrics import pairwise_distances

labels, centroids = kmeans(embeddings, k, init=init)          # step 1

Q = [quality[labels == j].mean() for j in range(k)]
cluster_order = np.argsort(Q)[::-1]                            # step 2 (desc avg quality)

def xnn_group(j):
    d = pairwise_distances(centroids[j:j+1], embeddings, metric="cosine").ravel()
    nn = np.argsort(d)[:xnn_k]                                 # nearest x pool samples
    return np.unique(np.concatenate([nn, [np.argmin(d)]]))     # {centroid-sample} ∪ xNN

emb_dist = pairwise_distances(embeddings, metric="cosine")
sil_dist = pairwise_distances(silhouettes, metric="cosine")    # precompute divergence matrix

P = []
for pos, j in enumerate(cluster_order):
    G = xnn_group(j)
    G = [i for i in G if i not in P]
    if not G:
        continue
    if pos == 0:
        pick = G[np.argmax(quality[G])]                        # step 3: highest-quality xNN candidate
    else:
        div = sil_dist[G][:, P].mean(axis=1)                   # step 4b: divergence to picked views
        r = div / (div.max() + EPS)                            # step 4b: relative scale
        d_emb = emb_dist[G][:, P].min(axis=1)                  # step 4b: embedding diversity
        D = r * d_emb                                          # step 4b: scaled diversity
        score = alpha * quality[G] + beta * D
        pick = G[np.argmax(score)]
    P.append(int(pick))

if len(P) < num_views:
    P += fill_up(quality, labels, cluster_order, exclude=P)    # step 5 (existing logic)
```

### 4.4 Worked micro-example

Three clusters, processing order by average quality: `C_1 (Q=0.9)`, `C_2 (Q=0.5)`,
`C_3 (Q=0.2)`. `alpha = 0.6`, `beta = 0.4`.

- `C_1` xNN group `{a, b, c}`, qualities `{0.95, 0.80, 0.70}` →
  **first pick `a`** (highest quality). `P = {a}`.
- `C_2` xNN group `{d, e}`, qualities `{0.60, 0.55}`.

  | candidate | `quality` | `div` (silh. to a) | `r = div/max` | `d_emb` (cos to a) | `D = r·d_emb` | `score` |
  |---|---|---|---|---|---|---|
  | `d` | 0.60 | 0.80 | 1.00 | 0.10 | 0.10 | 0.6·0.60 + 0.4·0.10 = **0.40** |
  | `e` | 0.55 | 0.20 | 0.25 | 0.50 | 0.125 | 0.6·0.55 + 0.4·0.125 = **0.38** |

  → **pick `d`**: `e` is farther in the embedding (0.50 vs 0.10) but its
  silhouette repeats the already-picked shape, so the relative scale `0.25`
  collapses its diversity advantage. Without scaling, `e` (score 0.53) would win —
  exactly the failure the plan fixes.

- `C_3` then evaluates against `P = {a, d}` using
  `div(i) = mean(silh-divergence to a, to d)`, scaled within its own group.

### 4.5 Parameter and aggregation choices (tunable, defaults proposed)

| Decision | Proposed default | Alternative |
|----------|------------------|-------------|
| Diversity aggregation over `P` | `min` embedding distance (`d_emb`) | `mean`, `prototype` |
| Divergence aggregation over `P` | `mean` silhouette divergence (`div`) | `min`, `max` |
| Relative normalization | `div / max(div in group)` | `div / (div_first + eps)`, raw `div` (no norm), percentile |
| Weighting | `alpha·quality + beta·scaled_div` (reuse `selector_alpha/beta`) | pure diversity in later clusters, `beta` ramp-up |
| Descriptor | silhouette | hu, zernike, fourier, shape_context (via `extract_shape_descriptor`) |

### 4.6 Config / CLI additions (proposed — not implemented)

Extend the existing selector rather than adding a new one:

| Flag / field | Default | Meaning |
|--------------|---------|---------|
| `--kmeans_diversity_scale` / `cfg.kmeans_diversity_scale` | `none` | `none` (current behaviour) or `silhouette` (enable relative scaling) |
| `--kmeans_descriptor` / `cfg.kmeans_descriptor` | `silhouette` | descriptor family for the divergence term |
| `--kmeans_diversity_alpha` / `cfg.kmeans_diversity_alpha` | `0.60` | quality weight (defaults to `selector_alpha`) |
| `--kmeans_diversity_beta` / `cfg.kmeans_diversity_beta` | `0.40` | diversity weight (defaults to `selector_beta`) |

In `selection/kmeans_xnn.py`, `TopKMeansXNN.select(..., silhouette_scores=None)`
already accepts the descriptor matrix (signature was extended); the scaling logic
is a new optional branch guarded by `diversity_scale == "silhouette"`.

### 4.7 Edge cases

- **Empty cluster / no candidates** — skip (continue).
- **Candidate already selected** — a pool sample can belong to several xNN groups;
  exclude already-picked indices from `G_j` (pseudocode above).
- **All-zero divergences** (duplicate/empty silhouettes) — `max(div)` ≈ 0; guard with
  `+ EPS`, so `r ≈ 1` and the mode degrades to the current pure-embedding behaviour
  rather than crashing or zeroing every score.
- **`num_views < k`** — process only the top `num_views` clusters.
- **`num_views > k`** — one pick per cluster, then existing fill-up.
- **Descriptor extraction failure (empty mask)** — `silhouette_descriptor` returns a
  zero-norm vector → `silhouette_divergence = 0.0` → candidate gets `r ≈ small`;
  this is safe but should be logged.

---

## 5. Relationship to the already-implemented GQD descriptor blending

The GQD selector (already implemented) *additively* blends descriptors into the
diversity term of the **global** quality-diversity pass:

```
GQD:  diversity = (1 - w)·emb_cos + w·sil_div
```

This plan is a **different mechanism for the kMeans selector**:

| | GQD blending (implemented) | kMeans relative scaling (this plan) |
|---|---|---|
| Selector | `quality_diversity` | `top_kmeans_xnn` |
| Combine descriptors | additive blend, weight `w` | multiplicative relative scale `r ∈ (0,1]` |
| Ordering | one global greedy pass | cluster-by-cluster, descending average quality |
| Start | highest-quality sample | highest-quality xNN candidate of the top cluster |
| Embedding stays in charge? | only if `w` small | yes — descriptors only *reweight* |

Both can coexist: GQD governs the global pass; the kMeans mode governs
cluster-wise selection.

---

## 6. Deferred: rotational embeddings

Not part of this plan. The earlier candidate "rotational embeddings"
(rotational augmentation, RotNet head, canonical-pose alignment) is parked — the
silhouette-relative scaling is simpler, training-free and reuses existing
descriptors. Revisit only if the scaled-diversity gains plateau and pose-angle
discrimination is still insufficient.

---

## 7. Verification plan (before merging, after implementation)

1. **Unit tests** (`tests/correctness_test_units/test_selection_algorithms.py`):
   - Synthetic data where two xNN candidates have near-equal embedding distance to
     the first pick but very different silhouette divergences → assert the *scaled*
     mode picks the silhouette-novel one and the *unscaled* mode picks the other.
   - Assert `diversity_scale="none"` reproduces the current `TopKMeansXNN` output
     exactly (backward compatibility).
   - Assert first pick = highest-quality candidate of the highest-average-quality
     cluster's xNN group; clusters processed in descending average quality.
   - Assert already-selected samples are never re-picked; `num_views < k` and
     `num_views > k` fill-up behave.
2. **Descriptor checks** — the relative scale is `≤ 1` for every candidate
   (so scaled diversity never exceeds embedding diversity).
3. **End-to-end** on a real dataset:
   - Selected-set **mean silhouette divergence** must be `≥` the unscaled baseline
     (the core claim).
   - Embedding coverage (mean min-distance of the pool to the selected set) must not
     regress.
4. **Full correctness suite** must stay green (855 assertions today).

---

## 8. Open questions

- **Relative normalization reference**: normalize by the group max (proposed) or by
  the first pick's own divergence (a fixed reference, keeps the scale stable across
  groups)? Group-max is data-driven but lets one outlier candidate lift everyone's
  scale; fixed-reference is stable but needs an `eps` floor.
- **Within-group candidate set**: should `G_j` include the *centroid sample* (the
  cluster's actual member nearest to `μ_j`) as today, or all cluster members?
- **Divergence aggregation over `P`**: `mean` (proposed) or `min` (the strongest
  repeat — most conservative)?
- **Weight schedule**: constant `beta`, or ramp `beta` up in later (lower-quality)
  clusters so diversity matters more once the high-quality clusters are consumed?
- **`k` vs `num_views`**: when `k < num_views`, should the fill-up stage also apply
  relative scaling, or keep the plain best-quality fill-up?
