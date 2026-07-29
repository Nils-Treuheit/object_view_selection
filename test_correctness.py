#!/usr/bin/env python3
"""
Correctness tests — each filter/module tested with known synthetic data.
Run: python test_correctness.py
"""

import sys
import numpy as np

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def make_image(h=200, w=200):
    return (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)


def make_circle_mask(h=200, w=200, radius=80):
    ys, xs = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    return ((xs - cx)**2 + (ys - cy)**2 <= radius**2).astype(np.uint8) * 255


# ─────────────────────────────────────────────
# 1. BLUR FILTER
# ─────────────────────────────────────────────
def test_blur():
    from preprocessing.blur_filter import BlurFilter

    h, w = 100, 100

    sharp = np.zeros((h, w, 3), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    sharp[(xs - w//2)**2 + (ys - h//2)**2 < 40**2] = 255

    import cv2
    blurry = cv2.GaussianBlur(sharp, (31, 31), 10)

    from data_io.observation import Observation
    from pathlib import Path

    obs_sharp = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=sharp, mask=np.ones((h, w), dtype=np.uint8) * 255)
    obs_blurry = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                             image=blurry, mask=np.ones((h, w), dtype=np.uint8) * 255)

    bf = BlurFilter(laplacian_threshold=120, tenengrad_threshold=35)
    s_s, p_s, _ = bf.evaluate(obs_sharp)
    s_b, p_b, _ = bf.evaluate(obs_blurry)

    check(obs_sharp.metrics.laplacian > obs_blurry.metrics.laplacian * 10,
          f"Sharp laplacian ({obs_sharp.metrics.laplacian:.0f}) >> blurry ({obs_blurry.metrics.laplacian:.0f})")
    check(obs_sharp.metrics.tenengrad > obs_blurry.metrics.tenengrad,
          f"Sharp tenengrad ({obs_sharp.metrics.tenengrad:.1f}) > blurry ({obs_blurry.metrics.tenengrad:.1f})")
    check(p_s, f"Sharp passes blur filter (lap={obs_sharp.metrics.laplacian:.0f}, ten={obs_sharp.metrics.tenengrad:.1f})")
    check(not p_b, f"Blurry fails blur filter (lap={obs_blurry.metrics.laplacian:.0f}, ten={obs_blurry.metrics.tenengrad:.1f})")


# ─────────────────────────────────────────────
# 2. AREA FILTER
# ─────────────────────────────────────────────
def test_area():
    from preprocessing.area_filter import AreaFilter
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    mask_large = np.zeros((h, w), dtype=np.uint8)
    mask_large[25:75, 25:75] = 255
    # 2500 / 10000 = 25%

    mask_small = np.zeros((h, w), dtype=np.uint8)
    mask_small[49:51, 49:51] = 255
    # 4 / 10000 = 0.04%

    af = AreaFilter(minimum_ratio=0.02)
    obs_large = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask_large)
    obs_small = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask_small)

    s_l, p_l, _ = af.evaluate(obs_large)
    s_s, p_s, _ = af.evaluate(obs_small)

    check(p_l, "Large mask (25%) passes area filter")
    check(not p_s, "Small mask (0.04%) fails area filter")
    check(abs(obs_large.metrics.area_ratio - 0.25) < 0.01,
          f"Large area ratio = {obs_large.metrics.area_ratio:.4f} (expected ~0.25)")
    check(abs(obs_small.metrics.area_ratio - 0.0004) < 0.0001,
          f"Small area ratio = {obs_small.metrics.area_ratio:.6f} (expected ~0.0004)")


# ─────────────────────────────────────────────
# 3. BORDER / TRUNCATION FILTER
# ─────────────────────────────────────────────
def test_border():
    from preprocessing.border_truncation import BorderFilter
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[25:75, 25:75] = 255

    mask_border = np.zeros((h, w), dtype=np.uint8)
    mask_border[0:75, 0:50] = 255
    # mask_pixels = 75*50 = 3750
    # border pixels: row 0 (75) + col 0 (75) - overlap (1) = 149
    # expected ratio = 149/3750 = 0.03973
    expected_border_ratio = (75 + 50 - 1) / (75 * 50)

    btf = BorderFilter(maximum_ratio=0.01)
    obs_c = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                        image=img, mask=mask_center)
    obs_b = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                        image=img, mask=mask_border)

    s_c, p_c, _ = btf.evaluate(obs_c)
    s_b, p_b, _ = btf.evaluate(obs_b)

    check(p_c, "Center mask passes border filter")
    check(not p_b, "Border mask fails border filter")
    check(obs_c.metrics.border_ratio == 0.0, f"Center border ratio = {obs_c.metrics.border_ratio}")
    check(abs(obs_b.metrics.border_ratio - expected_border_ratio) < 0.001,
          f"Border ratio = {obs_b.metrics.border_ratio:.4f} (expected {expected_border_ratio:.4f})")

    # Single-pixel corner touch
    mask_corner = np.zeros((h, w), dtype=np.uint8)
    mask_corner[0, 0] = 255
    obs_corner = Observation(id=2, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                             image=img, mask=mask_corner)
    s_cr, p_cr, _ = btf.evaluate(obs_corner)
    check(not p_cr, "Single-pixel corner fails border filter")
    check(abs(obs_corner.metrics.border_ratio - 1.0) < 0.01,
          f"Corner border ratio = {obs_corner.metrics.border_ratio:.2f} (expected 1.0)")


# ─────────────────────────────────────────────
# 4. OCCLUSION FILTER
# ─────────────────────────────────────────────
def test_occlusion():
    from preprocessing.occlusion_filter import OcclusionFilter
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # 2500 pixels

    hand_light = np.zeros((h, w), dtype=np.uint8)
    hand_light[25:35, 25:35] = 255  # 100 / 2500 = 4%

    hand_heavy = np.zeros((h, w), dtype=np.uint8)
    hand_heavy[25:75, 25:50] = 255  # 1250 / 2500 = 50%

    of = OcclusionFilter(maximum_overlap=0.15)

    obs_none = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=img, mask=mask, object_hand=None)
    obs_light = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask, object_hand=hand_light)
    obs_heavy = Observation(id=2, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask, object_hand=hand_heavy)

    s_n, p_n, _ = of.evaluate(obs_none)
    s_l, p_l, _ = of.evaluate(obs_light)
    s_h, p_h, _ = of.evaluate(obs_heavy)

    check(p_n, "No hand -> passes occlusion")
    check(p_l, "Light hand (4%) -> passes occlusion")
    check(not p_h, "Heavy hand (50%) -> fails occlusion")
    check(abs(obs_heavy.metrics.hand_overlap - 0.5) < 0.01,
          f"Heavy hand overlap = {obs_heavy.metrics.hand_overlap:.3f} (expected 0.5)")
    check(abs(obs_light.metrics.hand_overlap - 0.04) < 0.01,
          f"Light hand overlap = {obs_light.metrics.hand_overlap:.3f} (expected 0.04)")


# ─────────────────────────────────────────────
# 5. COMPLETENESS FILTER
# ─────────────────────────────────────────────
def test_completeness():
    from preprocessing.completeness_filter import CompletenessFilter
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    circle = make_circle_mask(h, w, radius=40)

    star = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    angle = np.arctan2(ys - cy, xs - cx)
    star_radius = 30 + 15 * np.sin(5 * angle)
    star_points = (r <= star_radius).astype(np.uint8) * 255

    cf = CompletenessFilter(minimum_score=0.6)

    obs_circle = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                             image=img, mask=circle)
    obs_star = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=img, mask=star_points)

    s_c, p_c, _ = cf.evaluate(obs_circle)
    s_s, p_s, _ = cf.evaluate(obs_star)

    check(p_c, "Circle passes completeness")
    check(not p_s, "Star fails completeness")
    check(obs_circle.metrics.solidity > 0.9, f"Circle solidity = {obs_circle.metrics.solidity:.3f}")
    check(obs_circle.metrics.completeness > 0.7, f"Circle completeness = {obs_circle.metrics.completeness:.3f}")
    check(obs_circle.metrics.completeness > obs_star.metrics.completeness + 0.1,
          f"Circle ({obs_circle.metrics.completeness:.3f}) >> star ({obs_star.metrics.completeness:.3f})")


# ─────────────────────────────────────────────
# 6. QUALITY SCORER
# ─────────────────────────────────────────────
def test_quality_scorer():
    from quality.quality_scorer import QualityScorer
    from quality.blur import BlurQuality
    from quality.area import AreaQuality
    from quality.occlusion import OcclusionQuality
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    scorer = QualityScorer(
        metrics=[BlurQuality(), AreaQuality(), OcclusionQuality()],
        weights={"blur": 0.5, "area": 0.3, "occlusion": 0.2},
    )

    obs = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                      image=img, mask=np.ones((h, w), dtype=np.uint8) * 255)
    q = scorer.score(obs)
    check(0.0 <= q <= 1.0, f"Scorer output in [0,1]: {q:.4f}")
    check(obs.quality > 0, f"Quality > 0: {obs.quality:.4f}")

    # Degraded case: blurry, occluded — score well below 1.0
    import cv2
    blurry = cv2.GaussianBlur(img, (31, 31), 10)
    hand_heavy = np.zeros((h, w), dtype=np.uint8)
    hand_heavy[25:75, 25:50] = 255
    obs_bad = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                          image=blurry, mask=np.ones((h, w), dtype=np.uint8) * 255,
                          object_hand=hand_heavy)
    q_bad = scorer.score(obs_bad)
    check(0.0 <= q_bad <= 1.0, f"Degraded quality in [0,1]: {q_bad:.4f}")
    check(q_bad < 0.6, f"Degraded quality ({q_bad:.4f}) < 0.6")
    check(q_bad > 0.05, f"Degraded quality ({q_bad:.4f}) > 0.05")


# ─────────────────────────────────────────────
# 7. HU MOMENTS — known invariants
# ─────────────────────────────────────────────
def test_hu_moments():
    from descriptors.hu import hu_moments

    h, w = 100, 100

    circle = make_circle_mask(h, w, radius=30)

    # Translation invariance
    shifted = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    shifted[(xs - 30)**2 + (ys - 40)**2 <= 30**2] = 255

    hu_base = hu_moments(circle)
    hu_trans = hu_moments(shifted)
    check(np.allclose(hu_base, hu_trans, atol=1e-3),
          f"Hu translation invariant: max_diff={np.max(np.abs(hu_base - hu_trans)):.6f}")

    # Rotation invariance (rotate by 90° — exact integer mapping)
    rotated = np.rot90(circle)
    hu_rot = hu_moments(rotated)
    check(np.allclose(hu_base, hu_rot, atol=0.1),
          f"Hu rotation invariant: max_diff={np.max(np.abs(hu_base - hu_rot)):.6f}")

    # Scale invariance (radius 15 vs 30)
    small = make_circle_mask(h, w, radius=15)
    hu_small = hu_moments(small)
    check(np.allclose(hu_base, hu_small, atol=0.01),
          f"Hu scale invariant: max_diff={np.max(np.abs(hu_base - hu_small)):.6f}")

    check(np.all(np.isfinite(hu_base)), "Hu moments are finite")
    check(len(hu_base) == 7, "7 Hu moments")


# ─────────────────────────────────────────────
# 8. FOURIER DESCRIPTORS — known shape
# ─────────────────────────────────────────────
def test_fourier():
    from descriptors.fourier import fourier_descriptors

    h, w = 100, 100
    circle = make_circle_mask(h, w, radius=30)
    fd = fourier_descriptors(circle, num_descriptors=32)
    check(len(fd) == 32, f"32 Fourier descriptors, got {len(fd)}")
    check(np.all(np.isfinite(fd)), "Fourier finite")
    check(np.all(fd >= 0), "Fourier non-negative")
    check(abs(fd.sum() - 1.0) < 0.01, f"Fourier L1-normalized: sum={fd.sum():.4f}")
    check(np.allclose(fd, fourier_descriptors(circle, num_descriptors=32)), "Fourier deterministic")

    fd_r40 = fourier_descriptors(make_circle_mask(h, w, radius=40), num_descriptors=32)
    fd_r50 = fourier_descriptors(make_circle_mask(h, w, radius=50), num_descriptors=32)
    check(np.allclose(fd_r40, fd_r50, atol=0.05),
          f"Fourier scale invariant: max_diff={np.max(np.abs(fd_r40 - fd_r50)):.4f}")

    # Translation invariance
    shifted = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    shifted[(xs - 30)**2 + (ys - 40)**2 <= 30**2] = 255
    fd_shift = fourier_descriptors(shifted, num_descriptors=32)
    check(np.allclose(fd, fd_shift, atol=0.01),
          f"Fourier translation invariant: max_diff={np.max(np.abs(fd - fd_shift)):.4f}")

    # Different shapes produce different descriptors
    rect = np.zeros((h, w), dtype=np.uint8)
    rect[25:75, 25:75] = 255
    fd_rect = fourier_descriptors(rect, num_descriptors=32)
    check(not np.allclose(fd, fd_rect, atol=0.05),
          f"Circle and rectangle produce different Fourier: diff={np.max(np.abs(fd - fd_rect)):.4f}")


# ─────────────────────────────────────────────
# 9. SHAPE CONTEXT
# ─────────────────────────────────────────────
def test_shape_context():
    from descriptors.shape_context import shape_context_descriptor

    h, w = 100, 100
    circle = make_circle_mask(h, w, radius=30)
    sc = shape_context_descriptor(circle)
    check(len(sc) > 0, f"Shape context dim={len(sc)}")
    check(abs(sc.sum() - 1.0) < 0.01, f"Shape context sums to 1: {sc.sum():.4f}")
    check(np.all(sc >= 0), "Shape context non-negative")
    check(np.allclose(sc, shape_context_descriptor(circle)), "Shape context deterministic")

    # Different shapes produce different descriptors
    rect = np.zeros((h, w), dtype=np.uint8)
    rect[25:75, 25:75] = 255
    sc_rect = shape_context_descriptor(rect)
    check(not np.allclose(sc, sc_rect, atol=0.1),
          f"Circle and rectangle produce different shape context: diff={np.max(np.abs(sc - sc_rect)):.4f}")


# ─────────────────────────────────────────────
# 10. ZERNIKE MOMENTS
# ─────────────────────────────────────────────
def test_zernike():
    from descriptors.zernike import zernike_moments

    h, w = 100, 100
    circle = make_circle_mask(h, w, radius=30)
    z = zernike_moments(circle, degree=6)
    check(len(z) > 0, f"Zernike dim={len(z)}")
    check(np.all(np.isfinite(z)), "Zernike finite")
    check(np.all(z >= 0), "Zernike non-negative (abs)")
    check(np.allclose(z, zernike_moments(circle, degree=6), atol=1e-6), "Zernike deterministic")

    # Rotation invariance: 90° rotation
    rotated = np.rot90(circle, k=1)
    z_rot = zernike_moments(rotated, degree=6)
    check(np.allclose(z, z_rot, atol=0.01),
          f"Zernike rotation invariant: max_diff={np.max(np.abs(z - z_rot)):.4f}")

    # Different shapes produce different descriptors
    rect = np.zeros((h, w), dtype=np.uint8)
    rect[25:75, 25:75] = 255
    z_rect = zernike_moments(rect, degree=6)
    check(not np.allclose(z, z_rect, atol=0.1),
          f"Circle and rectangle produce different Zernike: diff={np.max(np.abs(z - z_rect)):.4f}")


# ─────────────────────────────────────────────
# 11. SUBSET SELECTION — FPS
# ─────────────────────────────────────────────
def test_fps():
    from sklearn.metrics import pairwise_distances
    from selection.fps import FarthestPointSampling

    # Three mutually orthogonal vectors — all pairs have cos_dist=1, picks all 3
    eq = np.eye(3, dtype=np.float32)
    idx = FarthestPointSampling().select(eq, None, n=3)
    check(len(set(idx)) == 3, f"FPS selects all 3 orthogonal vectors: {sorted(idx)}")

    # Point cloud spanning full angular range — farthest by cosine are opposites
    sq = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0.5, 0.5, 0]],
                  dtype=np.float32)
    idx_sq = FarthestPointSampling().select(sq, None, n=2)
    d_cos = pairwise_distances(sq[idx_sq], metric="cosine")[0, 1]
    check(d_cos > 1.5,
          f"FPS picks near-opposite vectors (cos_dist={d_cos:.2f})")

    # Identical vectors — floating point epsilon breaks ties, picks distinct indices
    dup = np.ones((5, 2), dtype=np.float32)
    idx_dup = FarthestPointSampling().select(dup, None, n=3)
    check(len(set(idx_dup)) == 3, "FPS handles duplicate points")


# ─────────────────────────────────────────────
# 12. GREEDY QUALITY + DIVERSITY
# ─────────────────────────────────────────────
def test_gqd():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    # Three clusters: one with high quality, two far away
    emb = np.array([
        [0, 0], [0, 0.1], [0, -0.1],
        [10, 0], [10, 0.1],
        [-10, 0], [-10, 0.1],
    ], dtype=np.float32)
    qual = np.array([10, 9.5, 9, 1, 0.5, 1, 0.5], dtype=np.float32)

    # Pure quality mode: picks best quality points (cluster 0 dominated)
    gqd_q = GreedyQualityDiversity(alpha=0.99, beta=0.01)
    idx_q = gqd_q.select(emb, qual, n=3)
    check(np.all(idx_q[:2] == [0, 1]) or np.all(idx_q[:2] == [0, 2]),
          f"Quality-only GQD picks top-2 from cluster 0: {idx_q}")

    # Diversity mode: picks one from each cluster
    gqd_d = GreedyQualityDiversity(alpha=0.01, beta=0.99)
    idx_d = gqd_d.select(emb, qual, n=3)
    check(0 in idx_d, f"Diversity GQD includes best quality: {idx_d}")
    check(len(set(emb[idx_d, 0].round())) >= 2,
          f"Diversity GQD picks from multiple clusters: {idx_d}")


# ─────────────────────────────────────────────
# 13. DPP
# ─────────────────────────────────────────────
def test_dpp():
    from selection.dpp import DPPSelector

    # Two near-identical points: DPP should avoid picking both
    emb = np.array([[1, 0], [1.001, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    qual = np.ones(5, dtype=np.float32)

    dpp = DPPSelector(sigma=0.5)
    idx = dpp.select(emb, qual, n=2)
    check(len(set(idx)) == 2, "DPP selects 2 distinct points")
    d12 = np.linalg.norm(emb[idx[0]] - emb[idx[1]])
    check(d12 > 0.5, f"DPP avoids near-duplicates (dist={d12:.3f})")

    # With dissimilar points, DPP should pick the highest-quality first
    emb2 = np.eye(5, dtype=np.float32)
    qual2 = np.array([0.1, 0.2, 0.9, 0.3, 0.4], dtype=np.float32)
    idx2 = dpp.select(emb2, qual2, n=2)
    check(idx2[0] == 2, f"DPP picks highest quality first: {idx2} (expected 2)")


# ─────────────────────────────────────────────
# 14. CROP FUNCTIONS
# ─────────────────────────────────────────────
def test_crops():
    from embeddings.crop import bbox_crop, masked_crop, padded_square_crop

    h, w = 100, 100
    img = make_image(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:70, 20:80] = 255

    crop = bbox_crop(img, mask)
    check(crop.shape == (40, 60, 3), f"Bbox crop (40,60) = {crop.shape}")

    donut_mask = np.zeros((h, w), dtype=np.uint8)
    donut_mask[25:75, 25:75] = 255
    donut_mask[40:60, 40:60] = 0

    crop = masked_crop(img, donut_mask)
    check(crop.shape == (50, 50, 3), f"Masked crop (50,50) = {crop.shape}")
    check(np.sum(crop[15:35, 15:35]) == 0,
          f"Center hole is black: sum={np.sum(crop[15:35, 15:35])}")

    off_donut = np.zeros((h, w), dtype=np.uint8)
    off_donut[20:80, 30:90] = 255
    off_donut[45:65, 55:75] = 0
    crop_off = masked_crop(img, off_donut)
    check(crop_off.shape == (60, 60, 3), f"Off-center masked crop (60,60) = {crop_off.shape}")
    check(np.sum(crop_off[25:45, 25:45]) == 0,
          f"Off-center hole region is black: sum={np.sum(crop_off[25:45, 25:45])}")

    crop = padded_square_crop(img, mask)
    check(crop.shape == (224, 224, 3), f"Square crop 224x224 = {crop.shape}")


# ─────────────────────────────────────────────
# 15. FILTER PIPELINE — order + cumulative behavior
# ─────────────────────────────────────────────
def test_filter_pipeline():
    from preprocessing.filter_pipeline import FilterPipeline
    from preprocessing.blur_filter import BlurFilter
    from preprocessing.area_filter import AreaFilter
    from preprocessing.border_truncation import BorderFilter
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    pipeline = FilterPipeline([
        AreaFilter(minimum_ratio=0.02),
        BorderFilter(maximum_ratio=0.01),
    ])

    obs_pass = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=img, mask=np.ones((h, w), dtype=np.uint8) * 255)
    check(pipeline.run(obs_pass), "Perfect mask passes pipeline")

    obs_tiny = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=img, mask=np.zeros((h, w), dtype=np.uint8))
    obs_tiny.mask[49:51, 49:51] = 255
    check(not pipeline.run(obs_tiny), "Tiny mask rejected by pipeline")

    obs_border = Observation(id=2, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                             image=img, mask=np.zeros((h, w), dtype=np.uint8))
    obs_border.mask[0:50, 0:50] = 255
    check(not pipeline.run(obs_border), "Border mask rejected by pipeline")


# ─────────────────────────────────────────────
# 16. METRICS DATACLASS
# ─────────────────────────────────────────────
def test_metrics():
    from data_io.metrics import ObservationMetrics

    m = ObservationMetrics()
    check(m.laplacian == 0.0, "Default laplacian = 0")
    check(m.area_ratio == 0.0, "Default area_ratio = 0")
    m.laplacian = 123.4
    check(m.laplacian == 123.4, "Set laplacian = 123.4")


# ─────────────────────────────────────────────
# 17. EDGE CASES
# ─────────────────────────────────────────────
def test_edge_cases():
    from data_io.observation import Observation
    from pathlib import Path

    h, w = 50, 50
    img = make_image(h, w)

    # Empty mask
    from preprocessing.area_filter import AreaFilter
    af = AreaFilter(minimum_ratio=0.01)
    obs_empty = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=np.zeros((h, w), dtype=np.uint8))
    s_e, p_e, r_e = af.evaluate(obs_empty)
    check(not p_e, "Empty mask fails area filter")

    # Full mask
    obs_full = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=img, mask=np.ones((h, w), dtype=np.uint8) * 255)
    s_f, p_f, _ = af.evaluate(obs_full)
    check(p_f, "Full mask passes area filter")
    check(abs(obs_full.metrics.area_ratio - 1.0) < 0.01, f"Full mask area ratio = {obs_full.metrics.area_ratio:.4f}")

    # Grayscale image
    from preprocessing.blur_filter import BlurFilter
    gray = (make_image(h, w)[:, :, 0] * 0.299 + make_image(h, w)[:, :, 1] * 0.587 + make_image(h, w)[:, :, 2] * 0.114).astype(np.uint8)
    bf = BlurFilter()
    obs_gray = Observation(id=2, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                           image=np.stack([gray] * 3, axis=-1), mask=np.ones((h, w), dtype=np.uint8) * 255)
    bf.evaluate(obs_gray)
    check(obs_gray.metrics.laplacian > 0, f"Grayscale laplacian = {obs_gray.metrics.laplacian:.1f}")

    # Selection with n > dataset size
    from selection.fps import FarthestPointSampling
    emb_small = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    idx = FarthestPointSampling().select(emb_small, None, n=10)
    check(len(idx) == 3, f"FPS capped at dataset size: {len(idx)} (expected 3)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Object View Selection — Correctness Tests")
    print("=" * 55)

    tests = [
        ("Blur filter", test_blur),
        ("Area filter", test_area),
        ("Border filter", test_border),
        ("Occlusion filter", test_occlusion),
        ("Completeness filter", test_completeness),
        ("Quality scorer", test_quality_scorer),
        ("Hu moments", test_hu_moments),
        ("Fourier descriptors", test_fourier),
        ("Shape context", test_shape_context),
        ("Zernike moments", test_zernike),
        ("FPS selection", test_fps),
        ("Greedy quality+diversity", test_gqd),
        ("DPP selection", test_dpp),
        ("Crop functions", test_crops),
        ("Filter pipeline", test_filter_pipeline),
        ("Metrics dataclass", test_metrics),
        ("Edge cases", test_edge_cases),
    ]

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            FAIL += 1
            import traceback
            print(f"  [FAIL] {name} threw: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 55}")
    print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print(f"{'=' * 55}")
    sys.exit(0 if FAIL == 0 else 1)
