import numpy as np

from .vincents_base import VincentSoftFilter

# --------------------------------------------------------------------------- #
# True module globals: the default values live here and are passed through the
# filter constructor (they are deliberately not defined as class attributes).
# --------------------------------------------------------------------------- #

AREA_SCORE_SOFTNESS = 0.3
DEFAULT_MAX_FRACTION = 0.20


class VincentsAreaFilter(VincentSoftFilter):
    """Soft mask-area pre-filter, ported from score_mask_area in
    nit_view_selection/select_best_views.py.

    Mask area tends to be a continuous spectrum rather than a tight cluster
    with rare outliers, so the softness is small to discriminate at all.
    Small masks are penalized (direction "low_bad").

    Besides deriving the ``(0, 1]`` selection weight (``metrics.vincents_area``)
    this filter is also a working pre-filter: ``evaluate`` reports a
    quality-scaled stat score (``metrics.vincent_area_fraction``) and
    implements both rejection criteria from ``BaseFilter`` — an absolute
    threshold-based garbage floor (``hard_min_area_fraction`` on the raw stat)
    and a population-based extreme-bad-outlier removal (``outlier_z``, fit once
    over the population via robust median/MAD).
    """

    def __init__(
        self,
        softness: float = AREA_SCORE_SOFTNESS,
        hard_min_area_fraction: float = 0.0,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(
            enabled=enabled,
            hard_min=hard_min_area_fraction if hard_min_area_fraction > 0.0 else None,
            outlier_z=outlier_z,
            stat_attr="vincent_area_fraction",
            reason="vincents_area",
            direction="low_bad",
            weight_attr="vincents_area",
            softness=softness,
        )

        self.max_fraction = max_fraction

    def compute_stat(self, observation) -> float:
        mask = observation.mask > 0

        pixel_count = float(np.sum(mask))

        canvas_area = float(mask.shape[0] * mask.shape[1])

        if canvas_area <= 0:
            return 0.0

        return pixel_count / canvas_area

    def compute_score(self, stat: float) -> float:
        """Area fraction scaled against a fixed global anchor (``max_fraction``)."""
        return min(stat / self.max_fraction, 1.0)
