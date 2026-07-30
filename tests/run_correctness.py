#!/usr/bin/env python3
"""
Master correctness test runner.

Imports and executes all correctness test modules from
correctness_test_units/.

Usage:
    python tests/run_correctness.py
"""

import sys
import importlib
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


PASS = 0
FAIL = 0


def run_tests_in_module(module_name):
    global PASS, FAIL

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        FAIL += 1
        print(f"\n  [ERROR] Could not import {module_name}: {e}")
        return

    test_fns = [
        getattr(mod, attr)
        for attr in dir(mod)
        if attr.startswith("test_")
        and callable(getattr(mod, attr))
    ]

    if not test_fns:
        print(f"\n  [WARNING] No test_* functions found in {module_name}")
        return

    for fn in test_fns:
        try:
            fn()
        except AssertionError as e:
            FAIL += 1
            print(f"  [FAIL] {module_name}.{fn.__name__}")
            if str(e):
                print(f"          {e}")
        except Exception as e:
            FAIL += 1
            print(f"  [ERROR] {module_name}.{fn.__name__}: {e}")
            traceback.print_exc()


def main():
    from tests.test_utils import PASS, FAIL, reset_results, get_results

    reset_results()

    print("=" * 70)
    print(" Object View Selection — Correctness Test Suite")
    print("=" * 70)

    modules = [
        "tests.correctness_test_units.test_filters",
        "tests.correctness_test_units.test_descriptors_invariants",
        "tests.correctness_test_units.test_descriptors_shape",
        "tests.correctness_test_units.test_quality",
        "tests.correctness_test_units.test_selection",
        "tests.correctness_test_units.test_pipeline",
        "tests.correctness_test_units.test_edge_case",
        "tests.correctness_test_units.test_crops",
        "tests.correctness_test_units.test_metrics",
        "tests.correctness_test_units.test_plotting",
        "tests.correctness_test_units.test_selection_algorithms",
    ]

    for mod_name in modules:
        label = mod_name.rsplit(".", 1)[-1].replace("test_", "").replace("_", " ").title()
        print(f"\n--- {label} ---")
        run_tests_in_module(mod_name)

    p, f = get_results()

    print(f"\n{'=' * 70}")
    print(f"  Results: {p} passed, {f} failed out of {p + f}")
    print(f"{'=' * 70}")

    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
