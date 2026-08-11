# Plan: Improve Diversity for kMeans Selection (silhouette-relative-scaled diversity)

> **Status: PLAN ONLY — no implementation.** The approach is decided: **relative
> scaling with a silhouette-descriptor divergence score** applied to the
> `top_kmeans_xnn` selector. We start the selection in the **highest-average-quality
> cluster**, sample its **highest-quality xNN candidate**, then for every
> subsequent cluster (in descending average quality) compute the **silhouette
> divergence** of all its xNN candidates against the already-picked views and use
> it as a **relative scale** on the embedding-cosine diversity term before the
> quality-weighted pick. Nothing here is implemented yet.

---

## 1. Decision (confirmed)

For the kMeans selection improvement we **do NOT** take the rotational-embedding
route and we **do NOT** additively blend descriptors into the global GQD score.
We keep the frozen learned embeddings as the clustering space and inject the
silhouette-descriptor divergence **multiplicatively** as a *relative scaling
factor* on the diversity term, applied **cluster by cluster** in order of
descending average quality:

```
scaled_diversity(i) = relative_divergence(i)  ·  embedding_cosine_distance(i, selected)
```

- `relative_divergence(i)` ∈ (0, 1] — the silhouette divergence of candidate `i`
  to the already-picked views, normalized across its own xNN candidate group
  (the *most* divergent candidate in the group keeps the full diversity score; a
  candidate whose silhouette matches an already-picked view is down-weighted).
- `embedding_cosine_distance(i, selected)` — the embedding-space diversity term
  (distance to the nearest selected view), exactly as the current selector uses it.

Rationale for multiplying rather than adding: the learned embeddings are
"semantically" similar for near-duplicate poses, so an embedding distance can be
large or small for the wrong reasons. The silhouette divergence provides a
**relative correction factor** that only matters when it is small — it can
*suppress* a candidate whose geometry repeats an already-chosen view, but it
never inflates diversity on its own. Addition would let the descriptor dominate;
multiplication keeps the embedding space in charge and uses the descriptor to
*reweight* it.

---

## 2. The algorithm (as specified, in detail)

```
Step 1   Cluster.            Run kMeans on the pool embeddings → clusters C_1..C_k
                            (unchanged clustering: embeddings, init, xNN radius).

Step 2   Rank clusters.      Average quality Q_j = mean(quality in C_j).
                            Sort clusters by Q_j DESCENDING → processing order.

Step 3   First pick.         Take the HIGHEST-average-quality cluster.
                            Build its xNN group G = {centroid sample} ∪ xNN(μ).
                            Pick the HIGHEST-QUALITY candidate in G.
                            (No divergence / scaling yet — nothing is picked yet.)

Step 4   Loop next groups.   For the NEXT-highest xNN group (cluster) in rank order:
                             4a. Build G = {centroid sample} ∪ xNN(μ).
                             4b. Compute divergence scores: for every candidate in G,
                                 silhouette divergence to EVERY already-picked view.
                             4c. Scale the diversity score:
                                 relative_divergence = divergence / max(divergence in G)
                                 scaled_diversity   = relative_divergence · embedding_cos_distance(i, selected)
                             4d. Score = alpha · quality(i) + beta · scaled_diversity(i).
                             4e. Pick argmax(score) → add to selected set.
                             4f. The adjusted (scaled) space feeds the next group's pick.

Step 5   Fill-up (only if n > k).  Existing best-quality fill-up, unchanged.
```

The "adjusted space" in Step 4f is the diversity surface recomputed after each
pick: because the selected set grows, both `embedding_cosine_distance` (nearest
selected view) and `relative_divergence` (silhouette divergence to the new pick)
change for every remaining candidate.

---

## 3. Motivation

The `top_kmeans_xnn` selector clusters object views with kMeans in the embedding
space of a frozen vision backbone, then picks the best-quality sample in the
`xNN` neighbourhood of each centroid. Two weaknesses show up in practice:

1. **Learned embeddings underweight viewpoint/geometry.** They are trained for
   semantic identity, not pose. Near-duplicate views cluster together while
   genuinely novel angles may sit close in the embedding. The xNN neighbourhood
   can therefore contain views that are *geometrically* redundant even though
   their embedding distance looks "diverse".
2. **kMeans pass 1 is diversity-blind.** Today pass 1 visits clusters in **index
   order** (`for c in range(k)`) and picks the best-quality xNN candidate per
   cluster; it never checks whether a candidate's *shape* repeats an
   already-selected view, and it does not prefer the highest-quality cluster
   first.

Silhouette descriptors (`descriptors/silhouette.py`) are cheap (binarize → bbox
crop → resize → L2-normalize), rotation-information-preserving, and computed from
the same masks the pipeline already produces — so they give a per-view
*geometric* signal for free. This plan uses them as a **relative scale** on the
embedding diversity term of the kMeans-xNN selection.

---

## 4. Existing state (what this builds on)

Already implemented and wired:

| Piece | File | Status |
|-------|------|--------|
| `silhouette_descriptor(mask, size=64)` (binarize → bbox crop → square pad → resize → L2-norm 4096-d) | `descriptors/silhouette.py` | implemented |
| `silhouette_divergence(a, b)` = cosine distance; `0.0` on empty/norm-0 | `descriptors/silhouette.py` | implemented |
| `extract_shape_descriptor(obs, "silhouette")` returns the descriptor vector | `run.py` | implemented |
| `TopKMeansXNN(init=..., k=..., xnn_k=...)` — kMeans + per-cluster xNN + best-quality pick + fill-up | `selection/kmeans_xnn.py` | implemented |
| `TopKMeansXNN.select(..., silhouette_scores=None)` accepts a descriptor matrix | `selection/kmeans_xnn.py` | signature already extended |
| Descriptor extraction + scoring for the GQD path | `run.py` | implemented (GQD only, see §7) |
| Config fields `selector_alpha/beta` (0.60/0.40), `kmeans_init`, `kmeans_xnn_k` | `config.py` | implemented |

Relevant existing code (pass 1, currently index-ordered and diversity-free):

```python
# selection/kmeans_xnn.py  (TopKMeansXNN.select, pass 1)
used = set(); picks = []
for c in range(k):                       # <-- index order, NOT quality order
    candidates = _candidates_for_center(dist_centers[:, c], labels, c, self.xnn_k)
    remaining = [p for p in candidates if p not in used]
    pick = remaining[int(np.argmax(quality_scores[remaining]))]   # pure quality
    used.add(pick); picks.append(pick)
```

The plan **replaces pass 1** with the rank-ordered, diversity-scaled version
below, behind an opt-in flag. Existing behaviour remains the default so current
runs/tests keep passing.

---

## 5. Implementation detail

### 5.1 Changes to `selection/kmeans_xnn.py`

`TopKMeansXNN.__init__` gains one option:

```python
def __init__(self, init="best_quality", k=None, xnn_k=10,
             diversity_scale="none"):        # "none" | "silhouette"
```

`select(embeddings, quality_scores, n, silhouette_scores=None)`:

- `diversity_scale="none"` → **exactly** today's pass 1 + fill-up (backward
  compatible; the current test suite must stay green).
- `diversity_scale="silhouette"` → the new pass 1 below. Requires
  `silhouette_scores`; if absent, fall back to `"none"` behaviour with a log.

New pass 1 (replaces the `for c in range(k)` loop):

```python
cluster_order = _cluster_average_quality(quality_scores, labels, k)   # desc avg Q

# precompute full pairwise matrices once
emb_dist = pairwise_distances(embeddings, metric="cosine")
sil_dist = pairwise_distances(silhouettes, metric="cosine")

used = set(); picks = []
for pos, c in enumerate(cluster_order):
    candidates = _candidates_for_center(dist_centers[:, c], labels, c, self.xnn_k)
    remaining = [p for p in candidates if p not in used]
    if not remaining:
        continue
    if pos == 0:
        # Step 3: highest-average-quality cluster -> highest-quality xNN candidate
        pick = remaining[int(np.argmax(quality_scores[remaining]))]
    else:
        # Step 4b: divergence of every remaining candidate to all picked views
        div = sil_dist[remaining][:, picks].mean(axis=1)
        # Step 4c: relative scale within this group, then scale embedding diversity
        r = div / (div.max() + EPS)
        d_emb = emb_dist[remaining][:, picks].min(axis=1)
        scaled_diversity = r * d_emb
        # Step 4d/e: quality-weighted score, pick argmax
        score = alpha * quality_scores[remaining] + beta * scaled_diversity
        pick = remaining[int(np.argmax(score))]
    used.add(pick); picks.append(pick)
```

Where `alpha`/`beta` reuse `selector_alpha`/`selector_beta` (0.60 / 0.40) —
possibly overridable per the config section below. Fill-up (`_fill_remaining`,
pass 2) is untouched.

### 5.2 Inputs

- `embeddings` — (N, D) pool embeddings (learned, unchanged).
- `quality` — (N,) per-view quality scores.
- `silhouettes` — (N, d) per-view silhouette descriptor vectors (already produced
  for the GQD path via `extract_shape_descriptor(obs, "silhouette")`).
- `k` — number of kMeans clusters (default `num_views`).
- `xnn_k` — xNN neighbourhood radius (default 10).
- `alpha`, `beta` — quality vs diversity weights (default `selector_alpha`/`selector_beta`).

### 5.3 Worked micro-example

Three clusters, processing order by average quality: `C_1 (Q=0.9)`, `C_2 (Q=0.5)`,
`C_3 (Q=0.2)`. `alpha = 0.6`, `beta = 0.4`.

- `C_1` xNN group `{a, b, c}`, qualities `{0.95, 0.80, 0.70}` →
  **first pick `a`** (highest quality, Step 3). `P = {a}`.
- `C_2` xNN group `{d, e}`, qualities `{0.60, 0.55}` (Step 4).

  | candidate | `quality` | `div` (silh. to a) | `r = div/max` | `d_emb` (cos to a) | `scaled_div = r·d_emb` | `score` |
  |---|---|---|---|---|---|---|
  | `d` | 0.60 | 0.80 | 1.00 | 0.10 | 0.10 | 0.6·0.60 + 0.4·0.10 = **0.40** |
  | `e` | 0.55 | 0.20 | 0.25 | 0.50 | 0.125 | 0.6·0.55 + 0.4·0.125 = **0.38** |

  → **pick `d`**: `e` is farther in the embedding (0.50 vs 0.10) but its
  silhouette repeats the already-picked shape, so the relative scale `0.25`
  collapses its diversity advantage. Without scaling, `e` (score 0.53) would win —
  exactly the failure the plan fixes.

- `C_3` then evaluates against `P = {a, d}` using
  `div(i) = mean(silh-divergence to a, to d)`, scaled within its own group.

### 5.4 Parameter and aggregation choices (tunable, defaults proposed)

| Decision | Proposed default | Alternative |
|----------|------------------|-------------|
| Diversity aggregation over `P` | `min` embedding distance (`d_emb`) | `mean`, `prototype` |
| Divergence aggregation over `P` | `mean` silhouette divergence (`div`) | `min`, `max` |
| Relative normalization | `div / max(div in group)` | `div / (div_first + eps)`, raw `div` (no norm), percentile |
| Weighting | `alpha·quality + beta·scaled_div` (reuse `selector_alpha/beta`) | pure diversity in later clusters, `beta` ramp-up |
| Descriptor | silhouette | hu, zernike, fourier, shape_context (via `extract_shape_descriptor`) |

### 5.5 Config / CLI additions (proposed — not implemented)

Extend the existing selector rather than adding a new one:

| Flag / field | Default | Meaning |
|--------------|---------|---------|
| `--kmeans_diversity_scale` / `cfg.kmeans_diversity_scale` | `none` | `none` (current behaviour) or `silhouette` (enable relative scaling) |
| `--kmeans_descriptor` / `cfg.kmeans_descriptor` | `silhouette` | descriptor family for the divergence term |
| `--kmeans_diversity_alpha` / `cfg.kmeans_diversity_alpha` | `0.60` | quality weight (defaults to `selector_alpha`) |
| `--kmeans_diversity_beta` / `cfg.kmeans_diversity_beta` | `0.40` | diversity weight (defaults to `selector_beta`) |

### 5.6 Edge cases

- **Empty cluster / no candidates** — skip (continue).
- **Candidate already selected** — a pool sample can belong to several xNN groups;
  exclude already-picked indices from `remaining` (pseudocode above).
- **All-zero divergences** (duplicate/empty silhouettes) — `max(div)` ≈ 0; guard
  with `+ EPS`, so `r ≈ 1` and the mode degrades to the current pure-embedding
  behaviour rather than crashing or zeroing every score.
- **`num_views < k`** — process only the top `num_views` clusters (break after
  `pos == num_views`).
- **`num_views > k`** — one pick per cluster, then existing fill-up.
- **Descriptor extraction failure (empty mask)** — `silhouette_descriptor`
  returns a zero-norm vector → `silhouette_divergence = 0.0` → candidate gets
  `r ≈ small`; this is safe but should be logged.
- **`silhouette_scores` missing with `diversity_scale="silhouette"`** — fall back
  to `"none"` with a warning.

---

## 6. Relationship to the already-implemented GQD descriptor blending

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

## 7. Deferred: rotational embeddings

Not part of this plan. The earlier candidate "rotational embeddings"
(rotational augmentation, RotNet head, canonical-pose alignment) is parked — the
silhouette-relative scaling is simpler, training-free and reuses existing
descriptors. Revisit only if the scaled-diversity gains plateau and pose-angle
discrimination is still insufficient.

---

## 8. Verification plan (before merging, after implementation)

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

## 9. Open questions

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
