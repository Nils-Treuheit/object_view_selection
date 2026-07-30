"""
Metrics dataclass correctness tests.

Tests:
- default values
- field assignment
"""

from tests.test_utils import check


def test_metrics_defaults():
    from data_io.metrics import ObservationMetrics

    m = ObservationMetrics()

    check(m.laplacian == 0.0, "Default laplacian = 0")
    check(m.area_ratio == 0.0, "Default area_ratio = 0")


def test_metrics_assignment():
    from data_io.metrics import ObservationMetrics

    m = ObservationMetrics()
    m.laplacian = 123.4

    check(m.laplacian == 123.4, "Set laplacian = 123.4")


METRICS_TESTS = [
    ("Metrics defaults", test_metrics_defaults),
    ("Metrics assignment", test_metrics_assignment),
]
