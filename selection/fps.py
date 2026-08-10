import numpy as np
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


class FarthestPointSampling(SubsetSelector):

    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
        silhouette_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        n = min(n, len(embeddings))
        if n == 0:
            return np.array([], dtype=int)

        dist = pairwise_distances(embeddings, metric="cosine")

        idx = [np.random.randint(len(embeddings))]
        while len(idx) < n:
            min_dists = dist[:, idx].min(axis=1)
            next_idx = min_dists.argmax()
            idx.append(int(next_idx))

        return np.array(idx)