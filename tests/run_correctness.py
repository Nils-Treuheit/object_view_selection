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


def run_tests_in_module(module_name):
    """Run every test_* function in a module.

    Returns (function_pass, function_fail) for the module. Per-function
    check()-assertion results are printed on failure.
    """

    from tests.test_utils import get_results

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"\n  [ERROR] Could not import {module_name}: {e}")
        return 0, 1

    test_fns = [
        getattr(mod, attr)
        for attr in dir(mod)
        if attr.startswith("test_")
        and callable(getattr(mod, attr))
    ]

    if not test_fns:
        print(f"\n  [WARNING] No test_* functions found in {module_name}")
        return 0, 0

    fn_pass = 0
    fn_fail = 0
    for fn in test_fns:
        before_pass, before_fail = get_results()
        try:
            fn()
            fn_pass += 1
        except AssertionError as e:
            fn_fail += 1
            print(f"  [FAIL] {module_name}.{fn.__name__}")
            if str(e):
                print(f"          {e}")
        except Exception as e:
            fn_fail += 1
            print(f"  [ERROR] {module_name}.{fn.__name__}: {e}")
            traceback.print_exc()
        after_pass, after_fail = get_results()
        if after_fail - before_fail:
            print(
                f"  [FAIL] {module_name}.{fn.__name__}: "
                f"{after_fail - before_fail} check(s) failed"
            )
    return fn_pass, fn_fail


def main():
    from tests.test_utils import reset_results, get_results

    reset_results()

    print("=" * 70)
    print(" Object View Selection — Correctness Test Suite")
    print("=" * 70)

    modules = [
        "tests.correctness_test_units.test_filters",
        "tests.correctness_test_units.test_vincent_filters",
        "tests.correctness_test_units.test_descriptors_invariants",
        "tests.correctness_test_units.test_descriptors_shape",
        "tests.correctness_test_units.test_quality",
        "tests.correctness_test_units.test_selection",
        "tests.correctness_test_units.test_pipeline",
        "tests.correctness_test_units.test_edge_case",
        "tests.correctness_test_units.test_crops",
        "tests.correctness_test_units.test_metrics",
        "tests.correctness_test_units.test_plotting",
        "tests.correctness_test_units.test_quality_floor",
        "tests.correctness_test_units.test_selection_algorithms",
        "tests.correctness_test_units.test_prefilter_app",
    ]

    total_fn_pass = 0
    total_fn_fail = 0
    for mod_name in modules:
        label = mod_name.rsplit(".", 1)[-1].replace("test_", "").replace("_", " ").title()
        print(f"\n--- {label} ---")
        p, f = run_tests_in_module(mod_name)
        total_fn_pass += p
        total_fn_fail += f

    check_pass, check_fail = get_results()

    print(f"\n{'=' * 70}")
    print(
        f"  Results: {total_fn_pass} test functions passed, "
        f"{total_fn_fail} failed out of {total_fn_pass + total_fn_fail}"
    )
    print(f"  Check assertions: {check_pass} passed, {check_fail} failed")
    print(f"{'=' * 70}")

    return 0 if (total_fn_fail == 0 and check_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
