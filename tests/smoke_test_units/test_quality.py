"""
Smoke tests for quality scoring.
"""

from tests.smoke_test_utils import check


def test_quality_scoring(ds):
    obs = ds.observations[0]

    from quality.blur import BorderBlurQuality
    s = BorderBlurQuality().compute(obs)
    check(0 <= s <= 1, f"BorderBlurQuality={s:.4f} in [0,1]")

    from quality.area import AreaQuality
    s = AreaQuality().compute(obs)
    check(0 <= s <= 1, f"AreaQuality={s:.4f} in [0,1]")

    from quality.vincent import VincentsArtifactsQuality
    s = VincentsArtifactsQuality().compute(obs)
    check(0 <= s <= 1, f"VincentsArtifactsQuality={s:.4f} in [0,1]")

    from quality.centerness import CenternessQuality
    s = CenternessQuality().compute(obs)
    check(0 <= s <= 1, f"CenternessQuality={s:.4f} in [0,1]")

    from quality.quality_scorer import QualityScorer
    scorer = QualityScorer(
        metrics=[BorderBlurQuality(), AreaQuality(), VincentsArtifactsQuality(), CenternessQuality()],
        weights={"blur": 0.3, "area": 0.2, "vincents_artefacts": 0.2, "centerness": 0.3},
    )
    q = scorer.score(obs)
    check(0 < q <= 1, f"Overall quality={q:.4f} in (0,1]")
    check(obs.quality == q, "Observation.quality set")


QUALITY_TESTS = [
    ("Quality scoring smoke test", test_quality_scoring),
]
