from abc import abstractmethod

from .base import ScoreFilter
from .filter_utils import fit_robust_scores


class VincentSoftFilter(ScoreFilter):
    """Base for population-adapted soft pre-filters (ported from
    nit_view_selection/select_best_views.py).

    A soft filter computes and stores a raw per-observation stat, then
    ``fit_weights`` performs a population pass that derives a robust
    (median/MAD) typical scale from the data itself and turns the raw stats
    into selection weights in ``(0, 1]``.

    Being a ``ScoreFilter`` it also implements both ``BaseFilter`` rejection
    criteria on the raw stat (absolute garbage floor/ceiling + population
    outlier z), so a soft filter can act as a working pre-filter when those
    knobs are configured — while never rejecting during the weight pass.
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
        weight_attr: str | None = None,
        softness: float = 0.3,
    ):

        super().__init__(
            enabled=enabled,
            hard_min=hard_min,
            hard_max=hard_max,
            outlier_z=outlier_z,
            stat_attr=stat_attr,
            reason=reason,
            direction=direction,
        )

        self.weight_attr = weight_attr
        self.softness = softness

    @abstractmethod
    def compute_stat(self, observation) -> float:
        """Raw per-observation stat in the metric's natural units."""
        pass

    def fit_weights(self, observations):
        """Population pass: store (0,1] weights on each observation's metrics."""
        if not observations:
            return
        fit_robust_scores(
            observations,
            self.stat_attr,
            self.weight_attr,
            self.direction,
            self.softness,
        )
