"""
Shared utilities for smoke tests.
"""


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
    return cond


def reset_results():
    global PASS, FAIL
    PASS = 0
    FAIL = 0


def get_results():
    return PASS, FAIL
