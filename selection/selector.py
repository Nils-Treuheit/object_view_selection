from abc import ABC, abstractmethod

import numpy as np


class SubsetSelector(ABC):

    @abstractmethod
    def select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray | None = None,
        n: int = 10,
    ) -> np.ndarray:
        pass