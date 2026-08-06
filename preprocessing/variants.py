"""
Threshold- and outlier-based rejection variants layered on any pre-filter.

Every pre-filter's ``evaluate`` returns a ``(score, passed, reason)`` triple
with a 0..1 "goodness" score (higher = better). ``FilterVariant`` wraps an
existing filter and adds two optional rejection modes on top of it, without
touching the wrapped filter's own logic:

  * threshold  (``threshold_min``): an absolute, extremely-low-quality cutoff.
    ``score < threshold_min`` -> reject with reason ``<reason>_threshold``.
  * outlier    (``outlier_z``): a population-relative extreme-bad-outlier
    removal. Scores are fit once over the population (robust median/MAD z),
    then ``z <= -outlier_z`` -> reject with reason ``<reason>_outlier``.

Because both modes key off the same 0..1 score every filter already returns,
the wrapper is uniform across hard and soft filters. The population fit is
optional and skipped entirely when ``outlier_z`` is unset, so the default
pipeline behaviour is unchanged.
"""

import numpy as np

from .base import BaseFilter
from .vincent_utils import robust_center_scale


class FilterVariant(BaseFilter):
    """Add an absolute threshold floor and/or robust-bad-outlier rejection.

    ``inner`` must be a ``BaseFilter``. The variant defers to the inner filter
    for the base decision; if the inner rejects, its reason is kept verbatim.
    Otherwise the threshold / outlier checks may reject with an annotated
    reason, which groups cleanly into the per-reason sample folders.
    """

    def __init__(
        self,
        inner: BaseFilter,
        threshold_min: float | None = None,
        outlier_z: float | None = None,
    ):
        super().__init__(enabled=getattr(inner, "enabled", True))
        self.inner = inner
        self.threshold_min = threshold_min
        self.outlier_z = outlier_z
        self._robust = None  # (median, robust_scale) from fit()

    @property
    def name(self) -> str:
        return type(self.inner).__name__

    def requires_fit(self) -> bool:
        return self.outlier_z is not None

    def fit(self, observations):
        """Population pass: robust score stats for the outlier mode.

        Runs the inner filter on every observation so their scores are known,
        then stores the median/MAD robust center and scale. Only needed when
        ``outlier_z`` is set.
        """
        if self.outlier_z is None:
            return
        scores = []
        for obs in observations:
            if not self.inner.enabled:
                continue
            score, _passed, _reason = self.inner.evaluate(obs)
            scores.append(float(score))
        if scores:
            median, scale = robust_center_scale(np.array(scores, dtype=float))
            if scale <= 0:
                scale = 1.0
            self._robust = (median, scale)

    def evaluate(self, observation):
        score, passed, reason = self.inner.evaluate(observation)
        if not passed:
            return score, passed, reason

        if self.threshold_min is not None and float(score) < self.threshold_min:
            return score, False, f"{reason}_threshold"

        if self.outlier_z is not None and self._robust is not None:
            median, scale = self._robust
            z = (float(score) - median) / scale
            if z <= -self.outlier_z:
                return score, False, f"{reason}_outlier"

        return score, passed, reason


def reject_soft_variants(soft_filters, accepted, rejected):
    """Apply threshold/outlier rejection on already-fit soft weights.

    Soft filters never hard-reject during ``evaluate``; instead they derive a
    population weight in ``(0, 1]`` (``obs.metrics.<weight_attr>``).  When a
    soft filter is configured with ``threshold_min`` or ``outlier_z`` this
    moves accepted observations whose weight trips the cutoff into
    ``rejected``, with the annotated reason so the per-reason sample folders
    group them cleanly.  ``soft_filters`` is the name-keyed dict from
    ``run.build_soft_filters``.
    """
    if not accepted:
        return
    if rejected is None:
        rejected = []
    for key, soft_filter in soft_filters.items():
        threshold_min = getattr(soft_filter, "threshold_min", None)
        outlier_z = getattr(soft_filter, "outlier_z", None)
        weight_attr = getattr(soft_filter, "weight_attr", None)
        if (threshold_min is None and outlier_z is None) or weight_attr is None:
            continue

        weights = np.array(
            [getattr(o.metrics, weight_attr, 1.0) for o in accepted], dtype=float
        )
        median = scale = None
        if outlier_z is not None and weights.size:
            median, scale = robust_center_scale(weights)
            if scale <= 0:
                scale = 1.0

        keep = []
        drop = []
        for obs, w in zip(accepted, weights):
            if threshold_min is not None and float(w) < threshold_min:
                reason = f"{key}_threshold"
            elif (
                outlier_z is not None
                and median is not None
                and (float(w) - median) / scale <= -outlier_z
            ):
                reason = f"{key}_outlier"
            else:
                reason = None
            if reason is not None:
                obs.rejection_reason = reason
                drop.append(obs)
            else:
                keep.append(obs)

        if drop:
            print(f"  soft variant {key}: rejected {len(drop)} / {len(accepted)}")
            accepted[:] = keep
            rejected.extend(drop)
