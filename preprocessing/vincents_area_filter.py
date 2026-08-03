import numpy as np

from .vincents_base import VincentSoftFilter


class VincentsAreaFilter(VincentSoftFilter):
    """Soft mask-area pre-filter, ported from score_mask_area in
    nit_view_selection/select_best_views.py.

    Mask area tends to be a continuous spectrum rather than a tight cluster
    with rare outliers, so AREA_SCORE_SOFTNESS is small to discriminate at
    all. Small masks are penalized (direction "low_bad").
    """

    AREA_SCORE_SOFTNESS = 0.3

    stat_attr = "vincent_area_fraction"
    weight_attr = "vincents_area"
    direction = "low_bad"
    softness = AREA_SCORE_SOFTNESS

    def __init__(
        self,
        softness: float = AREA_SCORE_SOFTNESS,
        enabled=True,
    ):

        super().__init__(enabled)

        self.softness = softness

    def compute_stat(self, observation) -> float:

        mask = observation.mask > 0

        pixel_count = float(np.sum(mask))

        canvas_area = float(mask.shape[0] * mask.shape[1])

        if canvas_area <= 0:
            return 0.0

        return pixel_count / canvas_area
