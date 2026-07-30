# tests/test_correctness.py
"""
Master correctness test runner.

Imports and executes all correctness test modules from
correctness_test_units/.

Run:
    python tests/test_correctness.py

or:
    python tests/run_correctness.py
"""

import sys
import traceback


PASS = 0
FAIL = 0


def run_tests_in_module(mod):
    global PASS, FAIL

    functions = [
        getattr(mod, attr)
        for attr in dir(mod)
        if attr.startswith("test_")
        and callable(getattr(mod, attr))
    ]

    if not functions:
        print("  [WARNING] No tests found")

    for test_fn in functions:
        try:
            test_fn()

            PASS += 1
            print(f"  [PASS] {test_fn.__name__}")

        except AssertionError as e:
            FAIL += 1
            print(f"  [FAIL] {test_fn.__name__}")
            if str(e):
                print(f"          {e}")

        except Exception as e:
            FAIL += 1
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            traceback.print_exc()


def main():

    print("=" * 70)
    print(" Object View Selection - Correctness Test Suite")
    print("=" * 70)

    from tests.correctness_test_units import (
        test_filters,
        test_descriptors_invariants,
        test_descriptors_shape,
        test_selection,
        test_quality,
        test_pipeline,
        test_edge_case,
        test_crops,
        test_metrics,
        test_plotting,
        test_selection_algorithms,
    )

    modules = [
        ("Filters", test_filters),
        ("Descriptor invariants", test_descriptors_invariants),
        ("Descriptor shape discrimination", test_descriptors_shape),
        ("Selection algorithms", test_selection),
        ("Quality scoring", test_quality),
        ("Pipeline integration", test_pipeline),
        ("Edge cases", test_edge_case),
        ("Crop functions", test_crops),
        ("Metrics dataclass", test_metrics),
        ("Plotting", test_plotting),
        ("Selection algorithms (algorithmic)", test_selection_algorithms),
    ]

    for name, module in modules:
        print(f"\n--- {name} ---")
        run_tests_in_module(module)

    print(f"\n{'=' * 70}")
    print(f" Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print("=" * 70)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
