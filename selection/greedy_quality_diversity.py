import numpy as np
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


class GreedyQualityDiversity(SubsetSelector):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

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

        dist = pairwise_distances(embeddings, metric="cosine")

        idx = [int(quality_scores.argmax())]
        while len(idx) < n:
            best_score = -np.inf
            best_i = -1
            for i in range(len(embeddings)):
                if i in idx:
                    continue
                diversity = dist[i, idx].min()
                score = self.alpha * quality_scores[i] + self.beta * diversity
                if score > best_score:
                    best_score = score
                    best_i = i
            idx.append(best_i)

        return np.array(idx)