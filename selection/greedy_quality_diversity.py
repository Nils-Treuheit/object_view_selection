import numpy as np
from sklearn.metrics import pairwise_distances

from .selector import SubsetSelector


class GreedyQualityDiversity(SubsetSelector):
    """Greedy weighted quality-diversity selection (GQD / GDQ).

    Starts at the highest-quality sample, then repeatedly adds the sample
    maximising ``alpha * quality[i] + beta * diversity[i]``. The diversity
    term measures how far a candidate sits from the already-selected set in
    the semantic embedding space (cosine distance), optionally blended with a
    **silhouette-descriptor divergence** score, so each new pick changes the
    distances that drive the next pick.

    ``diversity_mode`` controls how the distance from a candidate to the
    current selected set is aggregated:

    - ``"min"``        nearest-selected-sample distance (classic GQD)
    - ``"max"``        farthest-selected-sample distance
    - ``"prototype"``  distance to the average sample (prototype) of the set

    ``use_descriptors`` toggles including a descriptor-based divergence (the
    ``silhouette_scores`` matrix passed to ``select``) alongside the
    embedding cosine distance; ``descriptor_weight`` sets its share of the
    combined diversity term. Without descriptors the term is the pure
    embedding cosine distance.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 diversity_mode: str = "min",
                 use_descriptors: bool = False,
                 descriptor_weight: float = 0.5):
        if diversity_mode not in ("min", "max", "prototype"):
            raise ValueError(f"Unknown diversity_mode: {diversity_mode!r} "
                             "(expected 'min', 'max' or 'prototype')")
        self.alpha = alpha
        self.beta = beta
        self.diversity_mode = diversity_mode
        self.use_descriptors = use_descriptors
        self.descriptor_weight = descriptor_weight

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

        if quality_scores is None:
            quality_scores = np.ones(len(embeddings))
        quality_scores = np.asarray(quality_scores, dtype=float)
        embeddings = np.asarray(embeddings, dtype=float)

        emb_dist = pairwise_distances(embeddings, metric="cosine")

        sil_scores = None
        sil_dist = None
        if self.use_descriptors and silhouette_scores is not None:
            sil_scores = np.asarray(silhouette_scores, dtype=float)
            if sil_scores.ndim == 2 and len(sil_scores) == len(embeddings):
                sil_dist = pairwise_distances(sil_scores, metric="cosine")

        def set_diversity(X, dmat, selected):
            """Distance of every pool sample to the selected set, in [0, 1]."""
            if self.diversity_mode == "prototype":
                prototype = X[selected].mean(axis=0).reshape(1, -1)
                return pairwise_distances(X, prototype, metric="cosine").ravel()
            if self.diversity_mode == "min":
                return dmat[:, selected].min(axis=1)
            if self.diversity_mode == "max":
                return dmat[:, selected].max(axis=1)
            raise ValueError(f"Unknown diversity_mode: {self.diversity_mode!r}")

        idx = [int(quality_scores.argmax())]
        while len(idx) < n:
            selected = np.array(idx)
            diversity = set_diversity(embeddings, emb_dist, selected)
            if sil_dist is not None:
                sil_div = set_diversity(sil_scores, sil_dist, selected)
                diversity = ((1.0 - self.descriptor_weight) * diversity
                             + self.descriptor_weight * sil_div)

            best_score = -np.inf
            best_i = -1
            for i in range(len(embeddings)):
                if i in idx:
                    continue
                score = self.alpha * quality_scores[i] + self.beta * diversity[i]
                if score > best_score:
                    best_score = score
                    best_i = i
            idx.append(best_i)

        return np.array(idx)
