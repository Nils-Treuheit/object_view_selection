import numpy as np
import torch

from .base import EmbeddingModel
from .crop import contrast_input


class EvaCLIPEmbedding(EmbeddingModel):

    def __init__(self, model_name="EVA02-CLIP-B-16", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="metaclip_400m"
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._dim = self.model.visual.output_dim

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