"""
Smoke tests for embeddings.
"""

import time
import numpy as np

from tests.smoke_test_utils import check


def test_crops(ds):
    obs = ds.observations[0]

    from embeddings.crop import bbox_crop, masked_crop, padded_square_crop
    crop = bbox_crop(obs.image, obs.mask)
    check(crop.ndim == 3, f"Bbox crop {crop.shape}")
    crop = masked_crop(obs.image, obs.mask)
    check(crop.ndim == 3, f"Masked crop {crop.shape}")
    crop = padded_square_crop(obs.image, obs.mask)
    check(crop.shape == (224, 224, 3), f"Square crop {crop.shape}")


def test_embedding_model(ds):
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not installed")
        return

    obs = ds.observations[0]

    from embeddings.dinov2 import DINOv2Embedding
    print("  Loading DINOv2 vits14 (may download)...", end=" ", flush=True)
    t0 = time.time()
    model = DINOv2Embedding(model_name="dinov2_vits14", device="cpu")
    print(f"done ({time.time()-t0:.0f}s)")
    t0 = time.time()
    emb = model.encode(obs.image, obs.mask)
    dt = time.time() - t0
    check(len(emb) == model.dimension, f"DINOv2 dim={len(emb)}, {dt*1000:.0f}ms")
    check(np.all(np.isfinite(emb)), "DINOv2 finite")


EMBEDDING_TESTS = [
    ("Crop smoke tests", test_crops),
    ("Embedding model smoke test", test_embedding_model),
]
