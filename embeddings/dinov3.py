import os
import numpy as np
import torch
from PIL import Image

from .base import EmbeddingModel


_HF_HOME = "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"


class DINOv3Embedding(EmbeddingModel):

    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
                 device=None, cache_dir=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cache_dir = cache_dir or _HF_HOME
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval().to(self.device)

        self._dim = self.model.config.hidden_size

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from embeddings.crop import padded_square_crop
        crop = padded_square_crop(image, mask, size=224)
        crop = crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop = (crop - mean) / std
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        feat = out.pooler_output
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy().flatten()