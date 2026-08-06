import numpy as np
import torch

from .base import EmbeddingModel
from .crop import contrast_input


class DINOv2Embedding(EmbeddingModel):

    def __init__(self, model_name="dinov2_vits14_reg", device=None, layer_norm=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        hub_name = {
            # base DINOv2
            "dinov2_vits14": "dinov2_vits14",
            "dinov2_vitb14": "dinov2_vitb14",
            "dinov2_vitl14": "dinov2_vitl14",
            "dinov2_vitg14": "dinov2_vitg14",
            # DINOv2 with registers (DINOv3)
            "dinov2_vits14_reg": "dinov2_vits14_reg",
            "dinov2_vitb14_reg": "dinov2_vitb14_reg",
            "dinov2_vitl14_reg": "dinov2_vitl14_reg",
            "dinov2_vitg14_reg": "dinov2_vitg14_reg",
            # short aliases
            "vit_small": "dinov2_vits14_reg",
            "vit_base": "dinov2_vitb14_reg",
            "vit_large": "dinov2_vitl14_reg",
            "vit_giant": "dinov2_vitg14_reg",
        }.get(model_name, model_name)

        self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.model.eval().to(self.device)

        self.layer_norm = layer_norm

        self._dim = {
            "dinov2_vits14": 384, "dinov2_vits14_reg": 384,
            "vit_small": 384,
            "dinov2_vitb14": 768, "dinov2_vitb14_reg": 768,
            "vit_base": 768,
            "dinov2_vitl14": 1024, "dinov2_vitl14_reg": 1024,
            "vit_large": 1024,
            "dinov2_vitg14": 1536, "dinov2_vitg14_reg": 1536,
            "vit_giant": 1536,
        }.get(model_name, 384)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop = contrast_input(image, mask, self.background, size=224)
        crop = (crop / 255.0).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)

        if self.layer_norm:
            feat = torch.nn.functional.normalize(feat, dim=-1)

        return feat.cpu().numpy().flatten()