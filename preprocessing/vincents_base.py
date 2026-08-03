from abc import abstractmethod

import numpy as np

from .base import BaseFilter
from .vincent_utils import fit_robust_scores


class VincentSoftFilter(BaseFilter):
    """Base for population-adapted soft pre-filters (ported from
    nit_view_selection/select_best_views.py).

    A soft filter never hard-rejects: ``evaluate`` computes and stores a raw
    per-observation stat, then ``fit_weights`` performs a population pass that
    derives a robust (median/MAD) typical scale from the data itself and turns
    the raw stats into selection weights in (0, 1].
    """

    # metric attribute holding the raw per-observation stat
    stat_attr = None
    # metric attribute holding the final population weight
    weight_attr = None
    # "low_bad" or "high_bad"
    direction = None
    # softness in robust-MADs
    softness = None

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        setattr(observation.metrics, self.stat_attr, float(self.compute_stat(observation)))

        return 1.0, True, ""

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
