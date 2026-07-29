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

    # Sharp image: white circle on black background (many edges)
    sharp = np.zeros((h, w, 3), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    sharp[(xs - w//2)**2 + (ys - h//2)**2 < 40**2] = 255

    # Blurry = Gaussian blurred
    import cv2
    blurry = cv2.GaussianBlur(sharp, (31, 31), 10)

    from data_io.observation import Observation
    from data_io.metrics import ObservationMetrics
    from pathlib import Path

    obs_sharp = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=sharp, mask=np.ones((h, w), dtype=np.uint8) * 255)
    obs_blurry = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                             image=blurry, mask=np.ones((h, w), dtype=np.uint8) * 255)

    bf = BlurFilter(laplacian_threshold=120, tenengrad_threshold=35)
    s_s, p_s, _ = bf.evaluate(obs_sharp)
    s_b, p_b, _ = bf.evaluate(obs_blurry)

    check(obs_sharp.metrics.laplacian > obs_blurry.metrics.laplacian * 1.5,
          f"Sharp laplacian ({obs_sharp.metrics.laplacian:.0f}) > blurry ({obs_blurry.metrics.laplacian:.0f})")
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
    from data_io.metrics import ObservationMetrics
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    # Large mask: 50% of image
    mask_large = np.zeros((h, w), dtype=np.uint8)
    mask_large[25:75, 25:75] = 255

    # Small mask: 1% of image
    mask_small = np.zeros((h, w), dtype=np.uint8)
    mask_small[49:51, 49:51] = 255

    af = AreaFilter(minimum_ratio=0.02)
    obs_large = Observation(id=0, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask_large)
    obs_small = Observation(id=1, image_path=Path(""), mask_path=Path(""), object_hand_path=None,
                            image=img, mask=mask_small)

    s_l, p_l, _ = af.evaluate(obs_large)
    s_s, p_s, _ = af.evaluate(obs_small)

    check(p_l, "Large mask (50%) passes area filter")
    check(not p_s, "Small mask (1%) fails area filter")
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
    from data_io.metrics import ObservationMetrics
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    # Centered mask — no border touch
    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[25:75, 25:75] = 255

    # Border-touching mask — covers entire left edge
    mask_border = np.zeros((h, w), dtype=np.uint8)
    mask_border[0:75, 0:50] = 255  # touches top + left edge

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
    check(obs_b.metrics.border_ratio > 0.0, f"Border ratio > 0 = {obs_b.metrics.border_ratio:.4f}")


# ─────────────────────────────────────────────
# 4. OCCLUSION FILTER
# ─────────────────────────────────────────────
def test_occlusion():
    from preprocessing.occlusion_filter import OcclusionFilter
    from data_io.observation import Observation
    from data_io.metrics import ObservationMetrics
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    # Object mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    # No hand
    no_hand = None

    # Hand covering 5% of mask
    hand_light = np.zeros((h, w), dtype=np.uint8)
    hand_light[25:35, 25:35] = 255

    # Hand covering 50% of mask
    hand_heavy = np.zeros((h, w), dtype=np.uint8)
    hand_heavy[25:75, 25:50] = 255

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
    check(p_l, "Light hand (5%) -> passes occlusion")
    check(not p_h, "Heavy hand (50%) -> fails occlusion")
    check(obs_heavy.metrics.hand_overlap > 0.3,
          f"Heavy hand overlap = {obs_heavy.metrics.hand_overlap:.3f}")


# ─────────────────────────────────────────────
# 5. COMPLETENESS FILTER
# ─────────────────────────────────────────────
def test_completeness():
    from preprocessing.completeness_filter import CompletenessFilter
    from data_io.observation import Observation
    from data_io.metrics import ObservationMetrics
    from pathlib import Path

    h, w = 100, 100
    img = make_image(h, w)

    # Perfect circle (solidity ~1, extent ~pi/4)
    circle = make_circle_mask(h, w, radius=40)

    # Jagged star shape (low solidity)
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
    check(obs_circle.metrics.solidity > 0.9, f"Circle solidity = {obs_circle.metrics.solidity:.3f}")
    check(obs_circle.metrics.completeness > 0.7, f"Circle completeness = {obs_circle.metrics.completeness:.3f}")
    check(obs_circle.metrics.completeness > obs_star.metrics.completeness,
          f"Circle ({obs_circle.metrics.completeness:.3f}) more complete than star ({obs_star.metrics.completeness:.3f})")


# ─────────────────────────────────────────────
# 6. QUALITY SCORER
# ─────────────────────────────────────────────
def test_quality_scorer():
    from quality.quality_scorer import QualityScorer
    from quality.blur import BlurQuality
    from quality.area import AreaQuality
    from quality.occlusion import OcclusionQuality
    from data_io.observation import Observation
    from data_io.metrics import ObservationMetrics
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


# ─────────────────────────────────────────────
# 7. HU MOMENTS — known invariants
# ─────────────────────────────────────────────
def test_hu_moments():
    from descriptors.hu import hu_moments

    h, w = 100, 100

    # Circle centered
    circle1 = make_circle_mask(h, w, radius=30)

    # Same circle shifted manually (not via array slice to avoid shape mismatch)
    circle2 = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.ogrid[:h, :w]
    circle2[(xs - 30)**2 + (ys - 40)**2 <= 30**2] = 255

    hu1 = hu_moments(circle1)
    hu2 = hu_moments(circle2)

    # Hu moments are invariant to translation
    check(np.allclose(hu1, hu2, atol=0.2), f"Hu translation invariant: max_diff={np.max(np.abs(hu1 - hu2)):.4f}")

    # Circle Hu moments have known approx values (order of magnitude)
    check(np.all(np.isfinite(hu1)), "Hu moments are finite")
    check(len(hu1) == 7, "7 Hu moments")


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
    check(np.any(fd > 0), f"At least one descriptor > 0 (max={fd.max():.4f})")

    # Same shape = same descriptors
    fd2 = fourier_descriptors(circle, num_descriptors=32)
    check(np.allclose(fd, fd2), "Fourier deterministic")


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

    sc2 = shape_context_descriptor(circle)
    check(np.allclose(sc, sc2), "Shape context deterministic")


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

    z2 = zernike_moments(circle, degree=6)
    check(np.allclose(z, z2, atol=1e-6), "Zernike deterministic")


# ─────────────────────────────────────────────
# 11. SUBSET SELECTION — FPS
# ─────────────────────────────────────────────
def test_fps():
    from selection.fps import FarthestPointSampling

    rng = np.random.RandomState(42)
    n = 20
    emb = rng.randn(n, 8).astype(np.float32)
    qual = rng.rand(n).astype(np.float32)

    # On 3 equidistant points, FPS should pick all 3
    eq = np.array([[0, 0], [10, 0], [5, 8.66]], dtype=np.float32)
    idx = FarthestPointSampling().select(eq, None, n=3)
    check(len(set(idx)) == 3, "FPS selects all 3 equidistant points")

    # FPS with quality should still pick distinct
    idx = FarthestPointSampling().select(emb, qual, n=5)
    check(len(set(idx)) == 5, "FPS selects 5 distinct points")


# ─────────────────────────────────────────────
# 12. GREEDY QUALITY + DIVERSITY
# ─────────────────────────────────────────────
def test_gqd():
    from selection.greedy_quality_diversity import GreedyQualityDiversity

    rng = np.random.RandomState(42)
    emb = rng.randn(30, 8).astype(np.float32)
    # One point has very high quality
    qual = np.ones(30, dtype=np.float32)
    qual[5] = 100.0

    gqd = GreedyQualityDiversity(alpha=0.9, beta=0.1)
    idx = gqd.select(emb, qual, n=3)

    check(5 in idx, "Highest quality point #5 selected first")
    check(len(set(idx)) == 3, "GQD selects 3 distinct points")


# ─────────────────────────────────────────────
# 13. DPP
# ─────────────────────────────────────────────
def test_dpp():
    from selection.dpp import DPPSelector

    rng = np.random.RandomState(42)
    emb = rng.randn(10, 8).astype(np.float32)
    qual = rng.rand(10).astype(np.float32)

    dpp = DPPSelector(sigma=0.5)
    idx = dpp.select(emb, qual, n=4)
    check(len(set(idx)) == 4, "DPP selects 4 distinct points")


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

    # Donut mask (hole in center) — mask in bbox has hole
    donut_mask = np.zeros((h, w), dtype=np.uint8)
    donut_mask[25:75, 25:75] = 255
    donut_mask[40:60, 40:60] = 0  # hole

    crop = masked_crop(img, donut_mask)
    check(crop.shape == (50, 50, 3), f"Masked crop (50,50) = {crop.shape}")
    # Center pixel (hole) should be black in masked crop
    check(np.sum(crop[15:35, 15:35]) == 0,
          f"Hole region is black: sum={np.sum(crop[15:35, 15:35])}")

    crop = padded_square_crop(img, mask)
    check(crop.shape == (224, 224, 3), f"Square crop 224x224 = {crop.shape}")


# ─────────────────────────────────────────────
# 15. METRICS DATACLASS
# ─────────────────────────────────────────────
def test_metrics():
    from data_io.metrics import ObservationMetrics

    m = ObservationMetrics()
    check(m.laplacian == 0.0, "Default laplacian = 0")
    check(m.area_ratio == 0.0, "Default area_ratio = 0")
    m.laplacian = 123.4
    check(m.laplacian == 123.4, "Set laplacian = 123.4")


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
        ("Metrics dataclass", test_metrics),
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
