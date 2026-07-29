#!/usr/bin/env python3
"""
Smoke tests for each submodule of the object view selection pipeline.
Run: python test_smoke.py --data_root /path/to/bottle
"""

import argparse
import time
import sys

import numpy as np

DATA_ROOT = "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/bottle"


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return cond


# ─────────────────────────────────────────────
# 1. DATA IO
# ─────────────────────────────────────────────
def test_data_io(data_root):
    print("\n--- 1. DATA IO ---")
    from data_io.dataset import Dataset
    from data_io.observation import Observation

    ds = Dataset(data_root)
    check(len(ds) > 0, f"Dataset loaded {len(ds)} observations")

    obs = ds.observations[0]
    check(isinstance(obs, Observation), "Observation is proper type")
    check(obs.image_path.exists(), f"Image path exists: {obs.image_path.name}")
    check(obs.mask_path.exists(), f"Mask path exists: {obs.mask_path.name}")

    ds.load_images()
    check(obs.image is not None, f"Image loaded, shape={obs.image.shape}")
    check(obs.mask is not None, f"Mask loaded, shape={obs.mask.shape}")
    check(obs.image.shape[:2] == obs.mask.shape[:2], "Image/mask dimensions match")
    check(obs.mask.dtype == np.uint8, "Mask is uint8")
    check(obs.image.shape[2] == 3, "Image is RGB")
    print(f"  Image range: [{obs.image.min()}..{obs.image.max()}]")
    return ds


# ─────────────────────────────────────────────
# 2. PREPROCESSING FILTERS
# ─────────────────────────────────────────────
def test_filters(ds):
    print("\n--- 2. PREPROCESSING FILTERS ---")
    obs = ds.observations[0]

    from preprocessing.blur_filter import BlurFilter
    bf = BlurFilter(laplacian_threshold=120, tenengrad_threshold=35, enabled=True)
    score, passed, reason = bf.evaluate(obs)
    check(isinstance(score, float), f"Blur score={score:.4f}")
    check(isinstance(passed, bool), f"Blur passed={passed}")
    check(obs.metrics.laplacian > 0, f"  Laplacian={obs.metrics.laplacian:.2f}")

    from preprocessing.area_filter import AreaFilter
    af = AreaFilter(minimum_ratio=0.02, enabled=True)
    score, passed, reason = af.evaluate(obs)
    check(isinstance(score, float), f"Area score={score:.4f}")

    from preprocessing.border_truncation import BorderFilter
    btf = BorderFilter(maximum_ratio=0.01, enabled=True)
    score, passed, reason = btf.evaluate(obs)
    check(isinstance(score, float), f"Border score={score:.4f}")

    from preprocessing.occlusion_filter import OcclusionFilter
    of = OcclusionFilter(maximum_overlap=0.15, enabled=True)
    score, passed, reason = of.evaluate(obs)
    check(isinstance(score, float), f"Occlusion score={score:.4f}")

    from preprocessing.completeness_filter import CompletenessFilter
    cf = CompletenessFilter(minimum_score=0.5, enabled=True)
    score, passed, reason = cf.evaluate(obs)
    check(isinstance(score, float), f"Completeness score={score:.4f}")

    from preprocessing.filter_pipeline import FilterPipeline
    pipeline = FilterPipeline([BlurFilter(), AreaFilter(), BorderFilter(), OcclusionFilter()])
    obs.rejected = False
    obs.rejection_reason = None
    result = pipeline.run(obs)
    check(isinstance(result, bool), f"Pipeline result={result}")

    # Rejection tests
    print("  -- synthetic rejection --")
    bad = ds.observations[0]
    bad.mask = np.zeros_like(bad.mask)
    s, p, r = af.evaluate(bad)
    check(not p and r == "small_object", f"Area rejects empty mask: {r}")

    bad2 = ds.observations[0]
    bad2.mask[:bad2.mask.shape[0]//3, :] = 255
    s, p, r = btf.evaluate(bad2)
    check(not p and r == "border", f"Border rejects truncated: {r}")


# ─────────────────────────────────────────────
# 3. QUALITY SCORING
# ─────────────────────────────────────────────
def test_quality(ds):
    print("\n--- 3. QUALITY SCORING ---")
    obs = ds.observations[0]

    from quality.blur import BlurQuality
    s = BlurQuality().compute(obs)
    check(0 <= s <= 1, f"BlurQuality={s:.4f} in [0,1]")

    from quality.area import AreaQuality
    s = AreaQuality().compute(obs)
    check(0 <= s <= 1, f"AreaQuality={s:.4f} in [0,1]")

    from quality.occlusion import OcclusionQuality
    s = OcclusionQuality().compute(obs)
    check(0 <= s <= 1, f"OcclusionQuality={s:.4f} in [0,1]")

    from quality.completeness import CompletenessQuality
    s = CompletenessQuality().compute(obs)
    check(0 <= s <= 1, f"CompletenessQuality={s:.4f} in [0,1]")

    from quality.quality_scorer import QualityScorer
    scorer = QualityScorer(
        metrics=[BlurQuality(), AreaQuality(), OcclusionQuality(), CompletenessQuality()],
        weights={"blur": 0.3, "area": 0.2, "occlusion": 0.2, "completeness": 0.3},
    )
    q = scorer.score(obs)
    check(0 < q <= 1, f"Overall quality={q:.4f} in (0,1]")
    check(obs.quality == q, "Observation.quality set")


# ─────────────────────────────────────────────
# 4. SHAPE DESCRIPTORS
# ─────────────────────────────────────────────
def test_descriptors(ds):
    print("\n--- 4. SHAPE DESCRIPTORS ---")
    mask = ds.observations[0].mask

    from descriptors.hu import hu_moments
    t0 = time.time()
    hu = hu_moments(mask)
    dt = time.time() - t0
    check(len(hu) == 7, f"Hu dim=7, got {len(hu)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(hu)), "Hu finite")

    from descriptors.zernike import zernike_moments
    t0 = time.time()
    z = zernike_moments(mask, degree=6)
    dt = time.time() - t0
    check(len(z) > 0, f"Zernike dim={len(z)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(z)), "Zernike finite")

    from descriptors.fourier import fourier_descriptors
    t0 = time.time()
    fd = fourier_descriptors(mask, num_descriptors=32)
    dt = time.time() - t0
    check(len(fd) == 32, f"Fourier dim=32, got {len(fd)}, {dt*1000:.1f}ms")
    check(np.all(np.isfinite(fd)), "Fourier finite")

    from descriptors.shape_context import shape_context_descriptor
    t0 = time.time()
    sc = shape_context_descriptor(mask)
    dt = time.time() - t0
    check(len(sc) > 0, f"Shape context dim={len(sc)}, {dt*1000:.1f}ms")
    check(abs(sc.sum() - 1.0) < 0.01, f"Shape context sums to ~1: {sc.sum():.4f}")

    check(np.allclose(hu, hu_moments(mask)), "Hu deterministic")


# ─────────────────────────────────────────────
# 5. SUBSET SELECTION
# ─────────────────────────────────────────────
def test_selection():
    print("\n--- 5. SUBSET SELECTION ---")
    n_obs, n_sel = 50, 8
    rng = np.random.RandomState(42)
    emb = rng.randn(n_obs, 64).astype(np.float32)
    qual = rng.rand(n_obs).astype(np.float32)

    from selection.fps import FarthestPointSampling
    idx = FarthestPointSampling().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "FPS")

    from selection.greedy_quality_diversity import GreedyQualityDiversity
    idx = GreedyQualityDiversity(alpha=0.4, beta=0.6).select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "GQD")

    from selection.facility_location import FacilityLocation
    idx = FacilityLocation().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "FacilityLocation")

    from selection.dpp import DPPSelector
    idx = DPPSelector(sigma=0.5).select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "DPP")

    from selection.next_best_view import NextBestView
    idx = NextBestView().select(emb, qual, n=n_sel)
    check(len(idx) == n_sel and len(set(idx)) == n_sel, "NBV")

    idx0 = FarthestPointSampling().select(emb[:0], qual[:0], n=5)
    check(len(idx0) == 0, "FPS empty")

    idx1 = FarthestPointSampling().select(emb[:3], qual[:3], n=10)
    check(len(idx1) == 3, f"FPS capped: {len(idx1)}/3")


# ─────────────────────────────────────────────
# 6. UTILS
# ─────────────────────────────────────────────
def test_utils(ds):
    print("\n--- 6. UTILS ---")
    obs = ds.observations[0]

    from utils.geometry import contour_area, bounding_box, mask_centroid
    check(contour_area(obs.mask) > 0, "Contour area")
    bb = bounding_box(obs.mask)
    check(len(bb) == 4, f"Bounding box={bb}")
    cy, cx = mask_centroid(obs.mask)
    check(0 <= cy < obs.mask.shape[0], f"Centroid y={cy:.1f}")

    from utils.math import normalize, softmax
    n = normalize(np.array([1., 2., 3., 4., 5.]))
    check(abs(n.min()) < 1e-6 and abs(n.max() - 1) < 1e-6, "Normalize")
    sm = softmax(np.array([1., 2., 3.]))
    check(abs(sm.sum() - 1) < 1e-6, "Softmax")

    from utils.visualization import create_overview_grid
    grid = create_overview_grid([obs.image, obs.image], [obs.mask, obs.mask],
                                titles=["A", "B"], cols=2)
    check(grid.ndim == 3 and grid.shape[2] == 3, f"Grid shape={grid.shape}")


# ─────────────────────────────────────────────
# 7. EMBEDDINGS
# ─────────────────────────────────────────────
def test_embeddings(ds):
    print("\n--- 7. EMBEDDINGS ---")
    obs = ds.observations[0]

    from embeddings.crop import bbox_crop, masked_crop, padded_square_crop
    crop = bbox_crop(obs.image, obs.mask)
    check(crop.ndim == 3, f"Bbox crop {crop.shape}")
    crop = masked_crop(obs.image, obs.mask)
    check(crop.ndim == 3, f"Masked crop {crop.shape}")
    crop = padded_square_crop(obs.image, obs.mask)
    check(crop.shape == (224, 224, 3), f"Square crop {crop.shape}")

    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not installed")
        return

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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    args = parser.parse_args()

    ds = test_data_io(args.data_root)
    test_filters(ds)
    test_quality(ds)
    test_descriptors(ds)
    test_selection()
    test_utils(ds)
    test_embeddings(ds)

    print(f"\n{'='*50}")
    print("  ALL SMOKE TESTS COMPLETE")
    print(f"{'='*50}")
