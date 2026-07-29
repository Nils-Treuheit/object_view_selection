import numpy as np
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


class DPPSelector(SubsetSelector):

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
    ) -> np.ndarray:
        n = min(n, len(embeddings))
        if n == 0:
            return np.array([], dtype=int)

        if quality_scores is None:
            quality_scores = np.ones(len(embeddings))

        sim = np.exp(-pairwise_distances(embeddings, metric="cosine") / self.sigma)
        q = quality_scores
        L = np.outer(q, q) * sim

        selected = []
        remaining = list(range(len(embeddings)))

        for _ in range(n):
            best_gain = -np.inf
            best_i = -1
            for i in remaining:
                if len(selected) == 0:
                    gain = L[i, i]
                else:
                    s = selected + [i]
                    Ls = L[np.ix_(s, s)]
                    try:
                        gain = np.linalg.slogdet(Ls)[1]
                    except np.linalg.LinAlgError:
                        gain = -np.inf
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            selected.append(best_i)
            remaining.remove(best_i)

        return np.array(selected)