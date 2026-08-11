"""
kMeans-xNN selection with silhouette-relative-scaled diversity.

Implements the plan in ``improve_diversity_embedding_for_kMeans.md``. Clusters
are visited in **descending average quality**. The highest-average-quality
cluster contributes its **highest-quality xNN candidate** first; every
subsequent cluster's pick maximises

    score(i) = alpha * quality(i) + beta * relative_divergence(i) * d_emb(i)

where ``relative_divergence(i)`` is the candidate's mean silhouette divergence
to the already-picked views, normalised across its own xNN group, and
``d_emb(i)`` is the candidate's nearest-neighbour embedding cosine distance to
the already-picked views. The relative scale lives in (0, 1], so the silhouette
descriptor only ever **down-weights** candidates whose geometry repeats an
already-chosen view; it never inflates diversity on its own.

Without ``silhouette_scores`` (or when every divergence is zero) the mode
degrades to the plain ``TopKMeansXNN`` behaviour: one best-quality pick per
cluster, then the existing best-quality fill-up.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector
from .kmeans_xnn import (
    _candidates_for_center,
    _cluster_average_quality,
    _fill_remaining,
    _fps_seeds,
    _quality_seeds,
)

EPS = 1e-9


class TopKMeansXNNSilhouetteSampling(SubsetSelector):
    """kMeans-xNN selection scaled by relative silhouette divergence.

    Mirrors ``TopKMeansXNN`` (same clustering, init modes, xNN candidate
    constraint and fill-up) but ranks clusters by average quality and reweights
    each subsequent pick's embedding diversity by the candidate's *relative*
    silhouette divergence to the already-picked set.

    Parameters
    ----------
    init : {"best_quality", "farthest"}
        k-means seed mode (default ``"best_quality"``).
    k : int | None
        Number of k-means clusters; ``None`` means one cluster per requested
        view (default).
    xnn_k : int
        xNN neighbourhood radius around each centroid (default 10).
    alpha : float
        Quality weight (default 0.60, matches ``selector_alpha``).
    beta : float
        Diversity weight (default 0.40, matches ``selector_beta``).
    """

    def __init__(self, init: str = "best_quality", k: int | None = None,
                 xnn_k: int = 10, alpha: float = 0.60, beta: float = 0.40):
        if init not in ("farthest", "best_quality"):
            raise ValueError(f"Unknown k-means init mode: {init}")
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if xnn_k < 1:
            raise ValueError(f"xnn_k must be >= 1, got {xnn_k}")
        self.init = init
        self.k = k
        self.xnn_k = xnn_k
        self.alpha = alpha
        self.beta = beta

    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
        silhouette_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        n = min(n, len(embeddings))
        if n == 0:
            return np.array([], dtype=int)

        if quality_scores is None:
            quality_scores = np.ones(len(embeddings))
        quality_scores = np.asarray(quality_scores, dtype=float)

        # Nothing to scale by -> plain kMeans-xNN (one quality pick per cluster).
        if silhouette_scores is None:
            from .kmeans_xnn import TopKMeansXNN
            return TopKMeansXNN(init=self.init, k=self.k, xnn_k=self.xnn_k).select(
                embeddings, quality_scores, n
            )
        silhouettes = np.asarray(silhouette_scores, dtype=float)
        if silhouettes.shape[0] != len(embeddings):
            raise ValueError(
                f"silhouette_scores has {silhouettes.shape[0]} rows but "
                f"{len(embeddings)} embeddings"
            )

        # Clustering: same seeding + xNN candidate constraint as TopKMeansXNN.
        k = min(self.k if self.k is not None else n, n)
        if self.init == "best_quality":
            seeds = _quality_seeds(quality_scores, k)
        else:
            seeds = _fps_seeds(embeddings, quality_scores, k)

        km = KMeans(n_clusters=k, init=embeddings[seeds], n_init=1, random_state=0)
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_

        dist_centers = pairwise_distances(embeddings, centers, metric="cosine")
        emb_dist = pairwise_distances(embeddings, metric="cosine")
        sil_dist = pairwise_distances(silhouettes, metric="cosine")

        # Step 2: clusters in descending average quality.
        cluster_order = _cluster_average_quality(quality_scores, labels, k)

        used = set()
        picks = []
        for c in cluster_order:
            candidates = _candidates_for_center(dist_centers[:, c], labels, c, self.xnn_k)
            remaining = [p for p in candidates if p not in used]
            if not remaining:
                continue

            if not picks:
                # Step 3: highest-average-quality cluster -> its highest-quality
                # xNN candidate (no divergence computed yet, nothing is picked).
                pick = remaining[int(np.argmax(quality_scores[remaining]))]
            else:
                # Step 4b: divergence of every candidate to ALL picked views.
                selected = np.array(picks)
                div = sil_dist[remaining][:, selected].mean(axis=1)
                # Step 4c: relative scale within this group; zero divergence
                # (duplicate/empty silhouettes) keeps the scale at 1 so the
                # mode degrades to pure embedding behaviour.
                div_max = div.max()
                if div_max <= EPS:
                    rel = np.ones_like(div)
                else:
                    rel = div / div_max
                d_emb = emb_dist[remaining][:, selected].min(axis=1)
                scaled_diversity = rel * d_emb
                # Step 4d/e: quality-weighted score, pick argmax.
                score = self.alpha * quality_scores[remaining] + self.beta * scaled_diversity
                pick = remaining[int(np.argmax(score))]

            used.add(pick)
            picks.append(pick)

        # Step 5: fill up to n when an explicit smaller k was requested.
        if len(picks) < n:
            picks = _fill_remaining(
                embeddings, quality_scores, labels, used, picks, n, cluster_order
            )

        return np.array(picks, dtype=int)
