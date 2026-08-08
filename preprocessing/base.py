from abc import ABC, abstractmethod

from data_io.observation import Observation

from .filter_utils import fit_stat_robust, outlier_rejected


class BaseFilter(ABC):
    """Common interface every pre-filter implements.

    A filter's ``evaluate`` returns a ``(score, passed, reason)`` triple with
    a 0..1 "goodness" score (higher = better).  Each filter has to implement
    an absolute threshold-based rejection criterion to filter out complete
    unusable garbage and — unless the decision is binary (e.g. touches border
    [truncation], no object mask [empty]) — also a population-based rejection
    of noticeably (by a large margin) bad outliers.

    Filters that need the population pass (the outlier mode) override
    ``need_fitting`` to return ``True`` and implement ``fit``; the pipeline
    calls ``fit_observations`` once over the whole dataset before the
    per-observation loop.
    """

    def __init__(self, enabled=True):

        self.enabled = enabled

    @abstractmethod
    def evaluate(self, observation: Observation):
        """
        Returns
        -------
        score : float
            Normalized score in [0,1]

        passed : bool
            Whether observation passes

        reason : str
            Reason if rejected
        """
        pass

    def need_fitting(self) -> bool:
        """Whether a population pass is required before per-observation eval.

        True only for filters configured with a population-based outlier
        criterion; the default is False.
        """
        return False

    def fit(self, observations):
        """Optional population pass over all observations.

        No-op by default; filters with ``need_fitting() == True`` override it
        to fit their population statistics once before ``evaluate`` runs.
        """
        pass


class ScoreFilter(BaseFilter):
    """Base for non-binary filters that score a raw per-observation stat.

    Implements both ``BaseFilter`` rejection criteria on the raw stat, once,
    so subclasses only supply the stat computation and the score mapping:

      * absolute threshold-based garbage rejection: ``stat`` below ``hard_min``
        or above ``hard_max`` is unusable regardless of the population;
      * population-based extreme-bad-outlier rejection: robust median/MAD
        z-score, fit once over the population (``fit``), rejecting the tail
        picked by ``direction``.

    ``evaluate`` returns ``(score, passed, reason)`` where a rejection reason
    is ``f"{reason}_threshold"`` / ``f"{reason}_outlier"`` so the per-reason
    sample folders group the two rejection modes cleanly.

    Subclasses must implement:

      * ``compute_stat(observation) -> float``: the raw stat in its natural
        units (published on ``observation.metrics.<stat_attr>``).
      * ``compute_score(stat) -> float``: map the stat to a 0..1 goodness
        score (higher = better).
    """

    def __init__(
        self,
        enabled=True,
        hard_min: float | None = None,
        hard_max: float | None = None,
        outlier_z: float | None = None,
        stat_attr: str | None = None,
        reason: str | None = None,
        direction: str = "low_bad",
        metric_aliases: tuple[str, ...] = (),
    ):

        super().__init__(enabled)

        # absolute garbage floor / ceiling on the raw stat (None disables)
        self.hard_min = hard_min
        self.hard_max = hard_max
        # population-based outlier removal; None disables the fit requirement
        self.outlier_z = outlier_z
        # where the raw stat is published on observation.metrics
        self.stat_attr = stat_attr
        # rejection reason base; _threshold / _outlier suffixes get appended
        self.reason = reason
        # "low_bad" (penalize the low tail) or "high_bad" (penalize the high tail)
        self.direction = direction
        # additional metric attributes to publish the raw stat on
        self.metric_aliases = metric_aliases
        # (median, robust_scale) of the raw stat, fit over the population
        self._robust = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    @abstractmethod
    def compute_stat(self, observation) -> float:
        """Raw per-observation stat in the metric's natural units."""
        pass

    @abstractmethod
    def compute_score(self, stat: float) -> float:
        """Map the raw stat to a 0..1 goodness score (higher = better)."""
        pass

    # ------------------------------------------------------------------ #
    # Population pass (outlier mode)
    # ------------------------------------------------------------------ #

    def need_fitting(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    def fit(self, observations):
        """Robust median/MAD of the raw stat, fit once over the population."""
        if self.outlier_z is None:
            return
        self._robust = fit_stat_robust(
            observations, self.compute_stat, enabled=self.enabled
        )

    # ------------------------------------------------------------------ #
    # Per-observation evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, observation):
        """Compute the raw stat and apply both rejection criteria.

        The score is ``compute_score(stat)``; the raw stat is published on
        ``observation.metrics.<stat_attr>`` (and any ``metric_aliases``).
        """
        if not self.enabled:
            return 1.0, True, ""

        stat = float(self.compute_stat(observation))
        if self.stat_attr is not None:
            setattr(observation.metrics, self.stat_attr, stat)
        for alias in self.metric_aliases:
            setattr(observation.metrics, alias, stat)

        score = self.compute_score(stat)

        # Absolute threshold-based garbage rejection: unusable regardless of
        # the population.
        if self.hard_min is not None and stat < self.hard_min:
            return 0.0, False, f"{self.reason}_threshold"
        if self.hard_max is not None and stat > self.hard_max:
            return 0.0, False, f"{self.reason}_threshold"

        # Population-based outlier rejection: the noticeably-bad tail.
        if outlier_rejected(stat, self._robust, self.outlier_z, self.direction):
            return score, False, f"{self.reason}_outlier"

        return score, True, self.reason
