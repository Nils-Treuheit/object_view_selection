import numpy as np

from .base import QualityMetric

# distance (px) from the frame border inside which the steep ramp bites
BORDER_ZONE_PX = 20.0


class CenternessQuality(QualityMetric):
    """How centered the visible object's center point is in the frame.

    Uses the mask centroid (center point) relative to the frame center:
    perfect 1.0 when the center point sits exactly at the frame center.
    Shifting the center point away from the center costs only a light
    (quadratic) decrease while it stays in the interior, but once the center
    point enters the ``BORDER_ZONE_PX``-px band along any image border the
    score drops off exponentially — objects whose center grazes the frame are
    crushed (same failure mode as truncation).
    """

    name = "centerness"

    def compute(self, observation):
        mask = observation.mask > 0
        if mask.ndim == 3:
            mask = mask.any(axis=-1)

        height, width = mask.shape

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return 0.0

        # object center point = mask centroid
        cx = float(xs.mean())
        cy = float(ys.mean())

        # normalized deviation of the center point from the frame center
        # (0 center, 1 frame edge)
        half_w = max((width - 1) / 2.0, 1e-6)
        half_h = max((height - 1) / 2.0, 1e-6)
        dx = abs(cx - (width - 1) / 2.0) / half_w
        dy = abs(cy - (height - 1) / 2.0) / half_h
        distance = min(np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0), 1.0)

        # light decrease in the center area: a quadratic falloff keeps the
        # punishment flat near the center and only bites further out
        centerness = 1.0 - distance ** 2

        # steep exponential ramp once the center point enters the border zone
        gap = min(
            cy,
            height - 1 - cy,
            cx,
            width - 1 - cx,
        )
        if gap < BORDER_ZONE_PX:
            centerness *= float(np.exp((gap - BORDER_ZONE_PX) * 0.5))

        return float(np.clip(centerness, 0.0, 1.0))
