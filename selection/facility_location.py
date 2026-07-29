import numpy as np
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


class FacilityLocation(SubsetSelector):

    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
    ) -> np.ndarray:
        n = min(n, len(embeddings))
        if n == 0:
            return np.array([], dtype=int)

        sim = 1.0 - pairwise_distances(embeddings, metric="cosine")
        sim = np.clip(sim, 0.0, 1.0)

        idx = [int(sim.sum(axis=1).argmax())]
        while len(idx) < n:
            best_obj = -np.inf
            best_i = -1
            for i in range(len(embeddings)):
                if i in idx:
                    continue
                obj = sim[:, idx].max(axis=1).sum()
                if obj > best_obj:
                    best_obj = obj
                    best_i = i
            idx.append(best_i)

        return np.array(idx)