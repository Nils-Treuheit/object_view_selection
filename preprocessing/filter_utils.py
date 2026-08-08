"""
Shared rejection helpers for pre-filters.

Every non-binary pre-filter has to implement the two ``BaseFilter`` rejection
criteria:

  * an absolute threshold-based criterion that filters out complete unusable
    garbage (a hard floor/ceiling on the raw stat), and
  * a population-based criterion that removes noticeably (by a large margin)
    bad outliers via a robust median/MAD z-score.

The population-fitting and z-outlier logic used to be copy-pasted into every
filter (``fit`` + a z check in ``evaluate``).  This module consolidates that
repetitive code so all filters share exactly one implementation:

  * ``robust_center_scale`` / ``one_sided_weight`` / ``fit_robust_scores``:
    the robust population-scoring primitives (also used to turn raw soft stats
    into ``(0, 1]`` selection weights).
  * ``robust_fit`` / ``fit_stat_robust``: fit a robust ``(median, scale)`` over
    a raw-stat population for the outlier criterion.
  * ``outlier_rejected``: the one-sided z-score tail check shared by every
    outlier rejection (hard ``evaluate`` and ``OutlierFilter`` alike).
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Robust population scoring
# --------------------------------------------------------------------------- #


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-derived robust scale (MAD * 1.4826 ~ std-equivalent)."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    robust_scale = float(np.median(np.abs(values - median))) * 1.4826
    return median, robust_scale


def one_sided_weight(
    values: np.ndarray,
    median: float,
    robust_scale: float,
    direction: str,
    softness: float,
) -> np.ndarray:
    """One-sided half-Gaussian decay from the robust center.

    Full weight on the "good" side of the median, smooth falloff on the "bad"
    side. `direction` is "low_bad" (values below median are penalized) or
    "high_bad" (values above median are penalized). `softness` is in
    robust-MADs and controls how quickly the falloff bites.
    """
    values = np.asarray(values, dtype=float)
    if direction == "high_bad":
        deviation = np.maximum(values - median, 0.0)
    elif direction == "low_bad":
        deviation = np.maximum(median - values, 0.0)
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    if robust_scale <= 0:
        return np.where(deviation <= 0, 1.0, 0.0)
    z = deviation / robust_scale
    return np.exp(-0.5 * (z / softness) ** 2)


def fit_robust_scores(
    observations,
    stat_attr: str,
    weight_attr: str,
    direction: str,
    softness: float,
) -> None:
    """Population pass: turn per-observation raw stats into robust (0,1] weights.

    Stores the computed weight on ``observation.metrics.<weight_attr>``.
    """
    values = np.array(
        [getattr(obs.metrics, stat_attr, 0.0) for obs in observations],
        dtype=float,
    )
    median, robust_scale = robust_center_scale(values)
    weights = one_sided_weight(values, median, robust_scale, direction, softness)
    for obs, weight in zip(observations, weights):
        setattr(obs.metrics, weight_attr, float(weight))


# --------------------------------------------------------------------------- #
# Shared fit / reject implementation for the outlier criterion
# --------------------------------------------------------------------------- #


def robust_fit(values) -> tuple[float, float] | None:
    """Robust ``(median, scale)`` of ``values``; ``None`` when empty.

    The scale is floored at ``1.0`` so a degenerate population (constant raw
    stat) never produces division-by-zero z-scores.
    """
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return None
    median, scale = robust_center_scale(values)
    if scale <= 0:
        scale = 1.0
    return median, scale


def fit_stat_robust(
    observations,
    compute_stat,
    enabled: bool = True,
) -> tuple[float, float] | None:
    """Robust ``(median, scale)`` of ``compute_stat`` over the population."""
    return robust_fit(
        float(compute_stat(obs))
        for obs in observations
        if enabled
    )


def outlier_rejected(
    stat: float,
    robust: tuple[float, float] | None,
    outlier_z: float | None,
    direction: str,
) -> bool:
    """True when ``stat`` is a bad outlier ``outlier_z`` robust-MADs from center.

    ``direction`` picks the penalized tail: ``"low_bad"`` rejects the low tail
    (``z <= -outlier_z``), ``"high_bad"`` rejects the high tail
    (``z >= outlier_z``).  No rejection when ``outlier_z`` or the fit is unset.
    """
    if outlier_z is None or robust is None:
        return False
    median, scale = robust
    z = (stat - median) / scale
    if direction == "low_bad":
        return z <= -outlier_z
    if direction == "high_bad":
        return z >= outlier_z
    raise ValueError(f"unknown direction: {direction!r}")
