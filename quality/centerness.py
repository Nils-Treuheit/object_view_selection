import numpy as np

from .base import QualityMetric

# distance (px) from the frame border inside which the exponential ramp bites
BORDER_RAMP_PX = 10.0


class CenternessQuality(QualityMetric):
    """How centered the visible object is in the frame.

    Based on the mask's bounding-box center relative to the frame center:
    very low punishment near the center, punishment increasing toward the
    edges, with an exponential punishment ramp within ``BORDER_RAMP_PX`` px
    of the image border (objects grazing the frame edge are the same failure
    mode as truncation and get crushed).
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

        bbox_x = float(xs.min())
        bbox_y = float(ys.min())
        bbox_x1 = float(xs.max())
        bbox_y1 = float(ys.max())
        bbox_cx = 0.5 * (bbox_x + bbox_x1)
        bbox_cy = 0.5 * (bbox_y + bbox_y1)

        # normalized deviation from the frame center (0 center, 1 frame edge)
        half_w = max((width - 1) / 2.0, 1e-6)
        half_h = max((height - 1) / 2.0, 1e-6)
        dx = abs(bbox_cx - (width - 1) / 2.0) / half_w
        dy = abs(bbox_cy - (height - 1) / 2.0) / half_h
        distance = min(np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0), 1.0)

        # low punishment near center, punishment increasing toward edges
        centerness = 1.0 - distance ** 1.5

        # exponential punishment ramp within BORDER_RAMP_PX of the frame border
        gap = min(
            bbox_y,
            height - 1 - bbox_y1,
            bbox_x,
            width - 1 - bbox_x1,
        )
        if gap < BORDER_RAMP_PX:
            centerness *= float(np.exp((gap - BORDER_RAMP_PX) * 0.5))

        return float(np.clip(centerness, 0.0, 1.0))
