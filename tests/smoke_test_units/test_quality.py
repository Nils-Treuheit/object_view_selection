"""
Smoke tests for quality scoring.
"""

from tests.smoke_test_utils import check


def test_quality_scoring(ds):
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


QUALITY_TESTS = [
    ("Quality scoring smoke test", test_quality_scoring),
]
