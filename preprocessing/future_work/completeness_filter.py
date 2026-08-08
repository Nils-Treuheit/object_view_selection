import cv2
import numpy as np

from ..base import BaseFilter


class CompletenessFilter(BaseFilter):
    """
    Estimates how complete the visible object shape is.

    Metrics:
        - Solidity
        - Extent
        - Convexity

    Returns a normalized score in [0,1].
    """

    def __init__(
        self,
        minimum_score=0.75,
        weights=(0.4, 0.3, 0.3),
        enabled=True,
    ):
        super().__init__(enabled)

        self.minimum_score = minimum_score
        self.weights = weights

    def evaluate(self, observation):

        if not self.enabled:
            return 1.0, True, ""

        mask = (observation.mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            return 0.0, False, "empty_mask"

        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        if area == 0:
            return 0.0, False, "empty_mask"

        #
        # Solidity
        #

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        solidity = (
            area / hull_area
            if hull_area > 0
            else 0.0
        )

        #
        # Extent
        #

        x, y, w, h = cv2.boundingRect(contour)

        bbox_area = w * h

        extent = (
            area / bbox_area
            if bbox_area > 0
            else 0.0
        )

        #
        # Convexity
        #

        perimeter = cv2.arcLength(contour, True)

        hull_perimeter = cv2.arcLength(hull, True)

        convexity = (
            hull_perimeter / perimeter
            if perimeter > 0
            else 0.0
        )

        convexity = np.clip(convexity, 0.0, 1.0)

        #
        # Final completeness score
        #

        score = (
            self.weights[0] * solidity
            + self.weights[1] * extent
            + self.weights[2] * convexity
        )

        observation.metrics.solidity = solidity
        observation.metrics.extent = extent
        observation.metrics.convexity = convexity
        observation.metrics.completeness = score

        passed = score >= self.minimum_score

        return score, passed, "incomplete_shape"
