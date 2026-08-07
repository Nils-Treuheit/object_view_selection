import cv2
import numpy as np

from ..base import BaseFilter
from ..vincent_utils import robust_center_scale


class CompletenessFilter(BaseFilter):
    """Shape-completeness pre-filter: solidity + extent + convexity.

    Stat = **extent** (area / bounding-box area) used as the population-adapted
    metric — frames whose object occupies an unusually small fraction of its
    bounding box are noticeably fragmented, cropped or poorly segmented.

    Besides deriving the ``(0, 1]`` selection weight this filter is also a
    working pre-filter: ``evaluate`` reports an extent-based score and implements
    both rejection criteria from ``BaseFilter`` -- an absolute threshold-based
    garbage floor (``hard_min_extent`` on raw extent) and a population-based
    extreme-bad-outlier removal (``outlier_z``, fit once over the population via
    robust median/MAD).

    Metrics stored per observation:
        - ``solidity``          -- area / convex-hull area
        - ``extent``            -- area / bounding-box area        [raw stat]
        - ``convexity``         -- convex-hull perimeter / contour perimeter (clipped to [0,1])
        - ``completeness_weight`` -- robust population weight in (0, 1] on fit pass
    """

    # ---------- configuration constants ----------
    EXTENT_SCORE_SOFTNESS = 0.4                 # robust-MADs for weight falloff
    MIN_EXTENT_DEFAULT = 0.15                   # absolute garbage floor on extent
    MIN_SOLIDITY_DEFAULT = 0.30                 # absolute garage floor on solidity

    WEIGHTS = (0.4, 0.3, 0.3)                  # solidity : extent : convexity

    reason = "incomplete_shape"

    stat_attr = "extent"
    weight_attr = "completeness_weight"
    direction = "low_bad"                       # low extent => fragment / poor segmentation
    softness = EXTENT_SCORE_SOFTNESS

    def __init__(
        self,
        softness: float | None = None,
        min_solidity: float = MIN_SOLIDITY_DEFAULT,
        hard_min_extent: float = MIN_EXTENT_DEFAULT,
        threshold_min_solidity: float | None = None,
        threshold_min_extent: float | None = None,
        threshold_min_weight: float | None = None,
        outlier_z: float | None = None,
        enabled: bool = True,
    ):

        super().__init__(enabled)

        self.softness = softness if softness is not None else self.EXTENT_SCORE_SOFTNESS
        self.min_solidity = min_solidity            # legacy alias (also set hard_min_extent via weight floor)
        self.hard_min_extent = hard_min_extent      # absolute garbage floor on raw extent

        # reject_soft_variants layer thresholds (on the fit weight in (0,1])
        self.threshold_min_solidity = threshold_min_solidity
        self.threshold_min_extent = threshold_min_extent
        self.threshold_min_weight = threshold_min_weight

        # outlier_z on raw extent -> requires population pass
        self.outlier_z = outlier_z
        self._robust = None                         # (median, robust_scale) of raw extent

    def requires_fit(self) -> bool:
        return self.outlier_z is not None

    # ------------------------------------------------------------------ #
    # Fit -- single population pass over the raw stat                     #
    # ------------------------------------------------------------------ #

    def fit(self, observations):
        """Robust median/MAD of extent, fit once over the population.

        Skipped unless ``outlier_z`` is set. The robust statistics are used in
        ``evaluate`` (outlier mode) and by ``fit_weights`` (selection weight).
        """
        if self.outlier_z is None:
            return
        raw = []
        for obs in observations:
            if not self.enabled:
                continue
            stat, _ = self._compute_raw_stats(observation=obs)
            raw.append(stat)                          # stat here *is* extent
        if raw:
            median, robust_scale = robust_center_scale(np.array(raw, dtype=float))
            if robust_scale <= 0:
                robust_scale = 1.0
            self._robust = (median, robust_scale)

    # ------------------------------------------------------------------ #
    # Evaluate -- per-observation scoring + absolute + population reject   #
    # ------------------------------------------------------------------ #

    def evaluate(self, observation):
        """Compute mask solidity/extent/convexity, score on extent, and apply both rejects.

        Score is capped at 1 and equals the weighted ``completeness`` composite when no
        criterion trips; if the extent floor fires the score drops to ``0.0``.
        """
        if not self.enabled:
            return -1.0, True, ""

        solidity, extent_value, convexity_value = self._compute_raw_stats(observation)
        observation.metrics.solidity = solidity
        observation.metrics.extent     = extent_value
        observation.metrics.convexity  = convexity_value
        observation.metrics.completeness = score

        # -- absolute garbage floors --
        if extent_value < self.hard_min_extent:
            return 0.0, False, f"{self.reason}_threshold"
        if self.min_solidity > 0 and solidity < self.min_solidity:
            return 0.0, False, f"{self.reason}_threshold"

        # -- population outlier (extent, "low" tail) --
        if self.outlier_z is not None and self._robust is not None:
            median, robust_scale = self._robust
            z = (extent_value - median) / robust_scale
            if z <= -self.outlier_z:
                # still return the composite score so diagnostics are visible
                passed_score = (
                    self.WEIGHTS[0] * solidity
                    + self.WEIGHTS[1] * extent_value
                    + self.WEIGHTS[2] * convexity_value
                )
                return passed_score, False, f"{self.reason}_outlier"

        # -- pass --
        passed_score = (
            self.WEIGHTS[0] * solidity
            + self.WEIGHTS[1] * extent_value
            + self.WEIGHTS[2] * convexity_value
        )
        return passed_score, True, self.reason

    # ------------------------------------------------------------------ #
    # Helper: pull the raw mask statistics                                 #
    # ------------------------------------------------------------------ #

    def _compute_raw_stats(self, observation):
        mask = (observation.mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return 0.0, 0.0, 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0, 0.0, 0.0

        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(contour)
        extent     = area / (w * h) if (w * h) > 0 else 0.0

        convexity_raw = cv2.arcLength(hull, True) / cv2.arcLength(contour, True) if cv2.arcLength(contour, True) > 0 else 0.0
        convexity     = float(np.clip(convexity_raw, 0.0, 1.0))

        return solidity, extent, convexity

    # ------------------------------------------------------------------ #
    # Weight population pass (reuse fit_robust_scores)                     #
    # ------------------------------------------------------------------ #

    def fit_weights(self, observations):
        """Population weight in ``(0, 1]`` on ``observation.metrics.<self.weight_attr>``.

        Mirrors ``VincentsMotionBlurFilter`` + ``VincentSoftFilter``: a robust
        (median/MAD) one-sided half-Gaussian fall-off penalizes the "bad"
        (low = fragmented / poor-segmentation) side.  Softness defaults to
        ``0.4`` robust-MADs -- deliberately small to discriminate.

        This method is called automatically by ``apply_soft_filters`` after
        ``fit(observations)`` when ``requires_fit()`` is true.
        """
        if not observations:
            return
        from ..vincent_utils import fit_robust_scores
        fit_robust_scores(
            observations,
            self.stat_attr,
            self.weight_attr,
            self.direction,
            self.softness,
        )
