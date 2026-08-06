import functools
import numpy as np
import torch

from .base import EmbeddingModel
from .crop import contrast_input


_MOONVIT_PATCHED = False


def _patch_transformers():
    """Monkey-patch transformers 5.14.x for MoonViT compatibility."""
    global _MOONVIT_PATCHED
    if _MOONVIT_PATCHED:
        return
    from transformers.modeling_utils import PreTrainedModel

    orig = PreTrainedModel._finalize_model_loading

    @functools.wraps(orig)
    def _patched_finalize(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return orig(self, *args, **kwargs)

    PreTrainedModel._finalize_model_loading = _patched_finalize
    _MOONVIT_PATCHED = True


class MoonViTEmbedding(EmbeddingModel):

    def __init__(self, model_name="moonshotai/MoonViT-SO-400M", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        _patch_transformers()

        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval().to(self.device)

        self._dim = self.model.config.hidden_size

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        crop = contrast_input(image, mask, self.background, size=224)
        from PIL import Image
        pil = Image.fromarray(crop)
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        grid_hws = inputs["image_grid_hws"].to(self.device)
        # MoonViT uses bfloat16 internally
        pixel_values = pixel_values.to(torch.bfloat16)

        with torch.no_grad():
            out = self.model(pixel_values, grid_hws)

        if isinstance(out, (tuple, list)):
            feat = torch.stack([o.to(torch.float32).mean(dim=1).mean(dim=0) if o.ndim == 3 else o.to(torch.float32).mean(dim=0) for o in out])
            feat = feat.mean(dim=0, keepdim=True)
        else:
            feat = out.float()
            feat = feat[:, 0] if feat.ndim > 1 else feat
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.cpu().numpy().flatten()
