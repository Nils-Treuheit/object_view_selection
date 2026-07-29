import numpy as np
import torch

from .base import EmbeddingModel
from .crop import padded_square_crop


class SigLIP2Embedding(EmbeddingModel):

    def __init__(self, model_name="google/siglip2-base-patch16-224", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

        self._dim = self.model.config.vision_config.hidden_size

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop = padded_square_crop(image, mask, size=224)
        from PIL import Image
        pil = Image.fromarray(crop)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])

        feat = torch.nn.functional.normalize(outputs.pooler_output, dim=-1)
        return feat.cpu().numpy().flatten()
