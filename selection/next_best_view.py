import numpy as np

from .selector import SubsetSelector


class NextBestView(SubsetSelector):

    def __init__(self, poses: np.ndarray | None = None):
        self.poses = poses

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

        idx = [int(quality_scores.argmax())]
        while len(idx) < n:
            remaining = [i for i in range(len(embeddings)) if i not in idx]
            scores = []
            for i in remaining:
                diversity = np.mean([
                    np.linalg.norm(embeddings[i] - embeddings[j])
                    for j in idx
                ])
                scores.append(quality_scores[i] + 0.5 * diversity)
            idx.append(remaining[int(np.argmax(scores))])

        return np.array(idx)