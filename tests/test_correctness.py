# tests/test_correctness.py
"""
Master correctness test runner.

Imports and executes all correctness test modules.

Run:
    python tests/test_correctness.py

or:

    pytest tests/test_correctness.py
"""

import sys
import traceback


PASS = 0
FAIL = 0


def run_test_module(name, module):
    global PASS, FAIL

    print("\n" + "=" * 70)
    print(f" {name}")
    print("=" * 70)

    functions = [
        getattr(module, attr)
        for attr in dir(module)
        if attr.startswith("test_")
        and callable(getattr(module, attr))
    ]

    if not functions:
        print("  [WARNING] No tests found")

    for test_fn in functions:
        try:
            test_fn()

            PASS += 1
            print(
                f"  [PASS] {test_fn.__name__}"
            )

        except AssertionError as e:
            FAIL += 1

            print(
                f"  [FAIL] {test_fn.__name__}"
            )

            if str(e):
                print(
                    f"          {e}"
                )

        except Exception as e:
            FAIL += 1

            print(
                f"  [ERROR] {test_fn.__name__}: {e}"
            )

            traceback.print_exc()


def main():

    print("=" * 70)
    print(" Object View Selection - Correctness Test Suite")
    print("=" * 70)


    # Import test modules

    from tests import test_filters
    from tests import test_descriptors_invariants
    from tests import test_descriptors_shape
    from tests import test_selection
    from tests import test_quality
    from tests import test_pipeline
    from tests import test_edge_cases


    modules = [
        (
            "Filters",
            test_filters
        ),
        (
            "Descriptor invariants",
            test_descriptors_invariants
        ),
        (
            "Descriptor shape discrimination",
            test_descriptors_shape
        ),
        (
            "Selection algorithms",
            test_selection
        ),
        (
            "Quality scoring",
            test_quality
        ),
        (
            "Pipeline integration",
            test_pipeline
        ),
        (
            "Edge cases",
            test_edge_cases
        ),
    ]


    for name, module in modules:
        run_test_module(
            name,
            module
        )


    print("\n" + "=" * 70)
    print(
        f" Results: {PASS} passed, {FAIL} failed "
        f"out of {PASS + FAIL}"
    )
    print("=" * 70)


    return 0 if FAIL == 0 else 1



if __name__ == "__main__":

    sys.exit(
        main()
    )