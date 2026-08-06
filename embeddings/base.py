from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):

    # Population-computed static background colour for the model input
    # (0 = black, 255 = white). None keeps the legacy zero-padded crop.
    background = None

    # Set to True only if the encoder's preprocessing can ingest a 4-channel
    # RGBA image (alpha: 1.0 original mask, 0.8 cut-out margin, 0.66 static
    # background). The built-in models all normalise to 3 channels, so they
    # keep this False.
    accepts_rgba = False

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