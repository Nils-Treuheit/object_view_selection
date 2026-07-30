#!/usr/bin/env python3
"""
Master smoke test runner.

Usage:
    python tests/run_smoke.py --data_root /path/to/bottle
"""

import argparse
import sys
import importlib
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_tests_in_module(module_name, ds=None):
    mod = importlib.import_module(module_name)

    test_fns = [
        (name, fn)
        for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
    ]

    if not test_fns:
        print(f"\n  [WARNING] No test_* functions found in {module_name}")
        return

    for name, fn in test_fns:
        try:
            import inspect
            sig = inspect.signature(fn)
            kwargs = {}
            if "ds" in sig.parameters:
                kwargs["ds"] = ds
            fn(**kwargs)
        except Exception as e:
            import tests.smoke_test_utils
            tests.smoke_test_utils.FAIL += 1
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/bottle")
    args = parser.parse_args()

    from tests.smoke_test_utils import reset_results, get_results

    reset_results()

    from data_io.dataset import Dataset
    print(f"Loading dataset from {args.data_root}")
    ds = Dataset(args.data_root)
    ds.load_images()
    print(f"Loaded {len(ds)} observations\n")

    modules = [
        "tests.smoke_test_units.test_data_io",
        "tests.smoke_test_units.test_filters",
        "tests.smoke_test_units.test_quality",
        "tests.smoke_test_units.test_descriptors",
        "tests.smoke_test_units.test_selection",
        "tests.smoke_test_units.test_utils_module",
        "tests.smoke_test_units.test_embeddings",
    ]

    for mod_name in modules:
        label = mod_name.rsplit(".", 1)[-1].replace("test_", "").replace("_", " ").title()
        print(f"\n--- {label} ---")
        run_tests_in_module(mod_name, ds)

    p, f = get_results()

    print(f"\n{'=' * 70}")
    print(f"  Results: {p} passed, {f} failed out of {p + f}")
    print(f"{'=' * 70}")

    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
