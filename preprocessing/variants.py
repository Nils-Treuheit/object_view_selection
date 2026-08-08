"""
Outlier-based rejection variant layered on any pre-filter.

Every pre-filter's ``evaluate`` returns a ``(score, passed, reason)`` triple
with a 0..1 "goodness" score (higher = better).  ``OutlierFilter`` wraps an
existing filter and adds a population-based extreme-bad-outlier rejection on
top of it, without touching the wrapped filter's own absolute threshold logic:

  * outlier (``outlier_z``): population-relative extreme-bad-outlier removal.
    Scores are fit once over the population (robust median/MAD z, see
    ``filter_utils``), then ``z <= -outlier_z`` -> reject with reason
    ``<reason>_outlier``.

Because the mode keys off the same 0..1 score every filter already returns,
the wrapper is uniform across hard and soft filters.  The population fit is
optional and skipped entirely when ``outlier_z`` is unset, so the default
pipeline behaviour is unchanged.

The default pre-filters implement both rejection criteria themselves via
``ScoreFilter`` (``preprocessing/base.py``) and are therefore not wrapped;
``OutlierFilter`` exists for filters that only implement their own absolute
criterion (e.g. the legacy ``AreaFilter`` / ``BorderFilter``).
"""

from .base import BaseFilter
from .filter_utils import outlier_rejected, robust_fit


class OutlierFilter(BaseFilter):
    """Add a robust-bad-outlier rejection on top of an inner ``BaseFilter``.

    ``inner`` must be a ``BaseFilter``.  The variant defers to the inner filter
    for the base decision; if the inner rejects, its reason is kept verbatim.
    Otherwise the outlier check may reject with the annotated
    ``<reason>_outlier`` reason, which groups cleanly into the per-reason
    sample folders.
    """

    def __init__(
        self,
        inner: BaseFilter,
        outlier_z: float | None = None,
    ):
        super().__init__(enabled=getattr(inner, "enabled", True))
        self.inner = inner
        self.outlier_z = outlier_z
        self._robust = None  # (median, robust_scale) from fit()

    @property
    def name(self) -> str:
        return type(self.inner).__name__

    def need_fitting(self) -> bool:
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
        self._robust = robust_fit(scores)

    def evaluate(self, observation):
        score, passed, reason = self.inner.evaluate(observation)
        if not passed:
            return score, passed, reason

        if outlier_rejected(
            float(score), self._robust, self.outlier_z, direction="low_bad"
        ):
            return score, False, f"{reason}_outlier"

        return score, passed, reason
