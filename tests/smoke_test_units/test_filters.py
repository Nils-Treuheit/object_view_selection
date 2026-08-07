"""
Smoke tests for preprocessing filters.
"""

from tests.smoke_test_utils import check


DATA_ROOT = "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/09_triprong"


def test_filter_basics(ds):
    obs = ds.observations[0]

    from preprocessing.border_blur_filter import BorderLaplacianBlurFilter
    bf = BorderLaplacianBlurFilter(enabled=True)
    score, passed, reason = bf.evaluate(obs)
    check(isinstance(score, float), f"BorderLaplacian score={score:.4f}")
    check(obs.metrics.laplacian > 0, f"  Laplacian={obs.metrics.laplacian:.2f}")

    from preprocessing.border_blur_filter import BorderTenengradBlurFilter
    tf = BorderTenengradBlurFilter(enabled=True)
    score, passed, reason = tf.evaluate(obs)
    check(isinstance(score, float), f"BorderTenengrad score={score:.4f}")
    check(obs.metrics.tenengrad > 0, f"  Tenengrad={obs.metrics.tenengrad:.2f}")

    from preprocessing.vincents_artefacts import VincentsArtifactsFilter
    af = VincentsArtifactsFilter(enabled=True)
    score, passed, reason = af.evaluate(obs)
    check(isinstance(score, float), f"Artifacts score={score:.4f}")

    from preprocessing.area_filter import AreaFilter
    ar = AreaFilter(minimum_ratio=0.02, enabled=True)
    score, passed, reason = ar.evaluate(obs)
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
    pipeline = FilterPipeline([bf, tf, af])
    obs.rejected = False
    obs.rejection_reason = None
    result = pipeline.run(obs)
    check(isinstance(result, bool), f"Pipeline result={result}")


def test_filter_rejection(ds):
    import numpy as np
    from preprocessing.area_filter import AreaFilter

    af = AreaFilter(minimum_ratio=0.02)
    bad = ds.observations[0]
    bad.mask = np.zeros_like(bad.mask)
    s, p, r = af.evaluate(bad)
    check(not p and r == "small_object", f"Area rejects empty mask: {r}")

    from preprocessing.border_truncation import BorderFilter
    btf = BorderFilter(maximum_ratio=0.005)
    bad2 = ds.observations[0]
    bad2.mask = np.zeros_like(bad2.mask)
    bad2.mask[:, :] = 255
    s, p, r = btf.evaluate(bad2)
    check(not p and r == "border", f"Border rejects truncated: {r}")

    from preprocessing.vincent_empty_mask import VincentEmptyMaskFilter
    vf = VincentEmptyMaskFilter()
    bad3 = ds.observations[0]
    bad3.mask = np.zeros_like(bad3.mask)
    s, p, r = vf.evaluate(bad3)
    check(not p and r == "vincent_empty_mask", f"Empty-mask filter rejects: {r}")


FILTER_TESTS = [
    ("Filter evaluation basics", test_filter_basics),
    ("Filter rejection tests", test_filter_rejection),
]
