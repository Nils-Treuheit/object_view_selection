"""
Top kMeans Embedding Selection in xNN quality Neighborhood.

Strategy: run k-means over the embedding pool with ``k = n`` (one cluster per
requested view), then for every cluster pick the *best-quality* pool sample
from the cluster centroid's ``xNN`` neighbourhood instead of blindly taking
the centroid itself.

The neighbourhood of a centroid is ``{centroid} ∪ {its x nearest neighbours}``,
with one constraint: a nearest neighbour may only be considered a candidate for
a centroid if it is closer to *that* centroid than to any other centroid (i.e.
it is a member of the cluster in question). Neighbours that actually live in a
neighbouring cluster are dropped; if the whole neighbourhood gets dropped the
cluster's medoid (the pool sample closest to the centroid) is used as fallback.

Two cluster-init modes are available:

  ``best_quality``   seed the k-means centres at the k highest-quality samples.
  ``farthest``       seed them via farthest-point sampling over the embedding
                     space (deterministic: starts at the highest-quality
                     sample, then repeatedly the point farthest from the
                     already-chosen seeds).

Unlike the centroid-only strategy, quality within the neighbourhood decides the
final pick, so a slightly lower-quality but far-from-centroid sample can win as
long as it stays inside the ``xNN`` radius.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


def _quality_seeds(quality_scores, k):
    """Indices of the ``k`` highest-quality samples (stable, ties by index)."""
    q = np.asarray(quality_scores, dtype=float)
    order = np.argsort(-q, kind="stable")
    return [int(i) for i in order[:k]]


def _fps_seeds(embeddings, quality_scores, k):
    """Deterministic farthest-point seeds over the embedding space.

    Starts at the highest-quality sample (index 0 when quality is uniform),
    then repeatedly adds the point with the largest minimum cosine distance to
    the already-chosen seeds.
    """
    n = len(embeddings)
    dist = pairwise_distances(embeddings, metric="cosine")

    q = np.asarray(quality_scores, dtype=float)
    seeds = [int(q.argmax())]
    while len(seeds) < k:
        min_dists = dist[:, seeds].min(axis=1)
        min_dists[seeds] = -1
        seeds.append(int(min_dists.argmax()))
    return seeds


def _candidates_for_center(dist_center, labels, cluster, x):
    """Constrained ``{centroid} ∪ xNN`` candidate set for one cluster centroid.

    ``dist_center`` is the (N,) cosine distance of every pool sample to the
    centroid; ``labels`` the k-means assignment. The raw neighbourhood is the
    ``x + 1`` pool samples nearest to the centroid (the centroid itself plus its
    ``x`` nearest neighbours). A candidate may not be closer to another centroid
    than to this one, so neighbours whose k-means label differs from ``cluster``
    are dropped. Falls back to the cluster medoid when nothing survives.
    """
    order = np.argsort(dist_center)
    raw = [int(p) for p in order[: x + 1]]
    candidates = [p for p in raw if labels[p] == cluster]
    if not candidates:
        members = np.where(labels == cluster)[0]
        medoid = int(members[dist_center[members].argmin()])
        candidates = [medoid]
    return candidates


class TopKMeansXNN(SubsetSelector):

    def __init__(self, init: str = "farthest", xnn_k: int = 3):
        if init not in ("farthest", "best_quality"):
            raise ValueError(f"Unknown k-means init mode: {init}")
        if xnn_k < 1:
            raise ValueError(f"xnn_k must be >= 1, got {xnn_k}")
        self.init = init
        self.xnn_k = xnn_k

    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
    ) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        n = min(n, len(embeddings))
        if n == 0:
            return np.array([], dtype=int)

        if quality_scores is None:
            quality_scores = np.ones(len(embeddings))
        quality_scores = np.asarray(quality_scores, dtype=float)

        k = n
        if self.init == "best_quality":
            seeds = _quality_seeds(quality_scores, k)
        else:
            seeds = _fps_seeds(embeddings, quality_scores, k)

        km = KMeans(n_clusters=k, init=embeddings[seeds], n_init=1, random_state=0)
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_

        dist_centers = pairwise_distances(embeddings, centers, metric="cosine")

        used = set()
        picks = []
        for c in range(k):
            candidates = _candidates_for_center(dist_centers[:, c], labels, c, self.xnn_k)
            remaining = [p for p in candidates if p not in used]
            if remaining:
                pick = remaining[int(np.argmax(quality_scores[remaining]))]
            else:
                order = np.argsort(-quality_scores, kind="stable")
                pick = next((int(i) for i in order if int(i) not in used), None)
                if pick is None:
                    break
            used.add(pick)
            picks.append(pick)

        return np.array(picks, dtype=int)
