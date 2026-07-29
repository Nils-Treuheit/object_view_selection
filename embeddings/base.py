from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):

    @abstractmethod
    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass