from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):

    # Population-computed static background colour for the model input
    # (0 = black, 255 = white). None keeps the legacy zero-padded crop.
    background = None

    def set_background(self, color: int):
        """Set the static background colour used by ``contrast_input``."""
        self.background = color

    @abstractmethod
    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass