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


def first_usable_observation(ds):
    """First observation whose mask has at least one non-empty closed contour."""
    import cv2

    for obs in ds.observations:
        contours, _ = cv2.findContours(
            obs.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours and cv2.contourArea(contours[0]) > 0:
            return obs
    return ds.observations[0]
