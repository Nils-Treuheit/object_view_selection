import numpy as np
import torch

from .base import EmbeddingModel
from .crop import contrast_input


class CLIPEmbedding(EmbeddingModel):

    def __init__(self, model_name="ViT-B/32", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        import clip
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self._dim = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }.get(model_name, 512)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop = contrast_input(image, mask, self.background, size=224, rgba=self.accepts_rgba)
        from PIL import Image
        pil = Image.fromarray(crop)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(tensor)

        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy().flatten()