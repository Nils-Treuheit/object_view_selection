import numpy as np

from ..base import BaseFilter


class BorderFilter(BaseFilter):

    def __init__(
        self,
        maximum_ratio=0.01,
        edge_maximum_ratio=0.25,
        enabled=True,
    ):

        super().__init__(enabled)

        self.maximum_ratio = maximum_ratio
        self.edge_maximum_ratio = edge_maximum_ratio

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = observation.mask > 0

        total = np.sum(mask)

        if total == 0:
            return 0, False, "empty_mask"

        border = np.zeros_like(mask)

        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        border_pixels = np.sum(mask & border)

        ring_ratio = border_pixels / total

        # Per-edge contact: length of mask pinned to each image frame edge,
        # normalized by the mask extent in the perpendicular direction.
        # An object cut off along an edge pins a large fraction of its
        # width/height to the frame; a fully-visible object that merely
        # grazes an edge only touches a few pixels.
        top_contact = np.sum(mask[0, :])
        bottom_contact = np.sum(mask[-1, :])
        left_contact = np.sum(mask[:, 0])
        right_contact = np.sum(mask[:, -1])

        rows = max(int(np.sum(mask.any(axis=1))), 1)
        cols = max(int(np.sum(mask.any(axis=0))), 1)

        edge_top = top_contact / cols
        edge_bottom = bottom_contact / cols
        edge_left = left_contact / rows
        edge_right = right_contact / rows

        edge_ratio = max(
            edge_top,
            edge_bottom,
            edge_left,
            edge_right,
        )

        observation.metrics.border_ratio = ring_ratio
        observation.metrics.edge_top_ratio = edge_top
        observation.metrics.edge_bottom_ratio = edge_bottom
        observation.metrics.edge_left_ratio = edge_left
        observation.metrics.edge_right_ratio = edge_right
        observation.metrics.edge_ratio = edge_ratio

        ring_score = 1.0 - min(
            ring_ratio / self.maximum_ratio,
            1.0,
        )

        edge_score = 1.0 - min(
            edge_ratio / self.edge_maximum_ratio,
            1.0,
        )

        score = min(ring_score, edge_score)

        passed = (
            ring_ratio <= self.maximum_ratio
            and edge_ratio <= self.edge_maximum_ratio
        )

        return score, passed, "border"
