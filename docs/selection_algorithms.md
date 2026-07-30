# Selection Algorithms

When the pipeline has filtered, scored, and embedded the observations, the final stage chooses a subset of `n` views. Five algorithms are available, each with a different strategy for balancing **quality** (preferring high-score views) and **diversity** (covering the embedding space broadly).

Every algorithm implements the same interface:

```python
class SubsetSelector(ABC):
    def select(self, embeddings, quality_scores=None, n=10) -> np.ndarray:
        # returns array of n indices into embeddings / quality_scores
```

They receive:
- `embeddings` — `(N, D)` array of embedding vectors (cosine distance is the default similarity metric)
- `quality_scores` — `(N,)` array in [0, 1] from the quality scorer
- `n` — number of views to select

All can also be run without quality scores (they fall back to uniform weights), producing a purely diversity-driven selection.

---

## 1. Farthest Point Sampling (FPS)

**File:** `selection/fps.py`

### Core Idea

Pick the point that is farthest from everything already picked. Pure diversity — quality is ignored entirely.

### Algorithm

```
1. Pick a starting index at random.
2. Repeat until n are selected:
   a. For every remaining point, compute its minimum cosine distance
      to the current selected set.
   b. Pick the point with the largest such minimum distance.
```

### Pseudocode

```python
dist = pairwise_distances(embeddings, metric="cosine")
idx = [randint(N)]                          # random start

while len(idx) < n:
    min_dists = dist[:, idx].min(axis=1)    # distance to nearest selected
    next_idx = min_dists.argmax()           # farthest from all selected
    idx.append(next_idx)
```

### Properties

| Property | Value |
|----------|-------|
| Quality-aware | No |
| Deterministic | No (random start) |
| Diversity | Maximises minimum pairwise distance between selected points |
| Coverage | Distributes points across the full extent of the embedding space |
| Complexity | O(n · N) distance lookups |

### When to Use

- When you want maximum viewpoint diversity regardless of quality.
- As a baseline for comparison with quality-aware methods.
- When all observations already have acceptable quality and you only need coverage.

---

## 2. Greedy Quality-Diversity (GQD)

**File:** `selection/greedy_quality_diversity.py`

### Core Idea

Score each candidate as a weighted combination of its own **quality** and its **diversity contribution** — defined as the minimum cosine distance to any already-selected point. This is a standard maximum-marginal-relevance (MMR) objective.

### Algorithm

```
1. Start with the single highest-quality observation.
2. Repeat until n are selected:
   a. For every remaining point i:
      score[i] = α · quality[i]  +  β · min(cosine_dist(i, selected))
   b. Pick the point with the highest score.
```

### Pseudocode

```python
dist = pairwise_distances(embeddings, metric="cosine")
idx = [quality_scores.argmax()]             # start with best quality

while len(idx) < n:
    best_score = -∞
    for i not in idx:
        diversity = dist[i, idx].min()       # min dist to ALL selected
        score = α · quality[i]  +  β · diversity
        if score > best_score: ...
    idx.append(best_i)
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.4 | Weight on quality. Higher = prefer high-score views |
| `beta` | 0.6 | Weight on diversity. Higher = spread views apart |

`alpha` and `beta` need not sum to 1 (the scores are compared, not normalised), but the defaults treat them as balanced with a slight diversity bias.

### Properties

| Property | Value |
|----------|-------|
| Quality-aware | Yes |
| Deterministic | Yes (quality argmax tiebreak is fully determined) |
| Diversity | Minimum distance to the selected set — pushes into empty regions |
| Greedy | Makes the best single-step choice at each iteration (not globally optimal) |
| Complexity | O(n · N) distance lookups |

### When to Use

- Default choice for most pipelines.
- When you want a tunable balance between quality and diversity.
- `alpha > beta` when quality is the primary concern; `beta > alpha` when coverage matters more.

---

## 3. Facility Location

**File:** `selection/facility_location.py`

### Core Idea

Select a set of "facilities" such that every point in the pool is as similar as possible to its nearest selected facility — i.e., the selected set should *cover* the whole embedding space. This is the classic k-medoids / facility-location objective from submodular optimisation.

### Algorithm

```
1. Convert cosine distance to similarity: sim = 1 - dist, clipped to [0, 1].
2. Start with the point that has the highest total similarity to all others
   (the most "central" point).
3. Repeat until n are selected:
   a. For every remaining point i, compute the coverage objective if i were added:
      obj = sum_j max_{k in selected ∪ {i}} sim(j, k)
   b. Pick the point that increases the objective the most.
```

### Pseudocode

```python
sim = 1.0 - pairwise_distances(embeddings, metric="cosine")
sim = clip(sim, 0, 1)

idx = [sim.sum(axis=1).argmax()]            # most central start

while len(idx) < n:
    best_obj = -∞
    for i not in idx:
        obj = sim[:, idx ∪ {i}].max(axis=1).sum()   # coverage sum
        if obj > best_obj: ...
    idx.append(best_i)
```

### Properties

| Property | Value |
|----------|-------|
| Quality-aware | No |
| Deterministic | Yes |
| Diversity | Indirect — pushes toward covering distinct regions |
| Coverage | Maximises similarity coverage of the whole pool |
| Complexity | O(n · N²) similarity lookups (more expensive than GQD/FPS) |

### When to Use

- When you want the selected set to be "representative" of the whole pool.
- When coverage — not just pairwise separation — matters.
- For smaller datasets, since it is computationally heavier.

---

## 4. Determinantal Point Process (DPP)

**File:** `selection/dpp.py`

### Core Idea

Model the selected subset as a sample from a determinantal point process. The probability of a set is proportional to the determinant of its kernel matrix — which naturally penalises redundancy (two very similar points produce a near-zero determinant).

### Algorithm

```
1. Build the quality-weighted similarity kernel:
   L[i,j] = quality[i] · quality[j] · exp(-dist[i,j] / sigma)
2. Greedily add the point that increases log-det(L_{selected}) the most.
```

### Pseudocode

```python
sim = exp(-pairwise_distances(embeddings, metric="cosine") / sigma)
q = quality_scores
L = outer(q, q) * sim                       # quality-weighted kernel

selected = []
for _ in range(n):
    for i in remaining:
        candidate = selected + [i]
        gain = slogdet(L[candidate, candidate])[1]   # log-determinant
        if gain > best_gain: ...
    selected.append(best_i)
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sigma` | 0.5 | Length scale of the RBF similarity kernel. Smaller σ = points must be very close to count as similar; larger σ = wider similarity radius |

### Properties

| Property | Value |
|----------|-------|
| Quality-aware | Yes (via the quality-weighted kernel) |
| Deterministic | Yes (greedy MAP inference) |
| Diversity | Natural — determinant penalises linear dependence between selected rows |
| Ground metric | Cosine distance wrapped in an RBF kernel |
| Complexity | O(n · N · k³) where k grows with n (determinant recomputation) |

### When to Use

- When you want a principled probabilistic model of diversity.
- When quality and similarity should interact multiplicatively (a low-quality point counts as less "redundant").
- For smaller datasets, since the determinant computation becomes expensive.

---

## 5. Next Best View (NBV)

**File:** `selection/next_best_view.py`

### Core Idea

A simple heuristic: start with the best-quality view, then repeatedly add the view that maximises `quality + 0.5 · mean_euclidean_distance_to_selected`. Unlike GQD, the diversity term is the *mean* (not minimum) distance and uses *Euclidean* (not cosine) distance in embedding space.

### Algorithm

```
1. Start with the highest-quality observation.
2. Repeat until n are selected:
   a. For every remaining point i:
      diversity = mean(||embed[i] - embed[j]|| for j in selected)
      score = quality[i]  +  0.5 · diversity
   b. Pick the point with the highest score.
```

### Pseudocode

```python
idx = [quality_scores.argmax()]

while len(idx) < n:
    scores = []
    for i in remaining:
        diversity = mean([||embeddings[i] - embeddings[j]|| for j in idx])
        scores.append(quality[i] + 0.5 * diversity)
    idx.append(remaining[argmax(scores)])
```

### Properties

| Property | Value |
|----------|-------|
| Quality-aware | Yes (additive) |
| Deterministic | Yes |
| Diversity metric | Mean Euclidean distance in embedding space |
| Diversity type | Average separation (not worst-case) |
| Complexity | O(n · N) distance computations |

### When to Use

- When you want a simple baseline similar to GQD but with mean (not min) diversity.
- When Euclidean distance in the raw embedding space is more meaningful than cosine similarity.
- The fixed 0.5 diversity weight cannot be tuned.

---

## Comparison Summary

| Algorithm | Quality-aware | Deterministic | Diversity strategy | Metric | Complexity |
|-----------|:---:|:---:|---|:---:|:---:|
| **FPS** | — | — | Farthest from selected set | Cosine | O(n·N) |
| **GQD** | α · quality | ✓ | Min distance to selected set | Cosine | O(n·N) |
| **Facility Location** | — | ✓ | Coverage (max-similarity sum) | Cosine | O(n·N²) |
| **DPP** | Quality-weighted kernel | ✓ | Determinant (anti-redundancy) | RBF(cosine) | O(n·N·k³) |
| **NBV** | quality + 0.5 · diversity | ✓ | Mean distance to selected set | Euclidean | O(n·N) |

## CLI Selection

```bash
# Greedy Quality-Diversity (default)
python run.py --selector quality_diversity --selector_alpha 0.4 --selector_beta 0.6

# Farthest Point Sampling
python run.py --selector fps

# Facility Location
python run.py --selector facility_location

# Determinantal Point Process
python run.py --selector dpp

# Next Best View
python run.py --selector next_best_view
```
