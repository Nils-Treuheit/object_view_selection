import cv2
import numpy as np

from ..base import BaseFilter
from ..vincent_utils import (
    compute_boundary_blur_variance,
    compute_boundary_tenengrad,
    mask_to_foreground,
    robust_center_scale,
)


class BlurFilter(BaseFilter):
    """Combined boundary-band sharpness pre-filter (Laplacian variance + Tenengrad).

    Two independent metrics, each with its own absolute garbage floor and
    population-based severe out-lier removal:

    1. **Raw statistical values** (per observation)<br>
       The **variance of the Laplacian** restricted to the boundary band, and the
       **mean Sobel magnitude (Tenengrad)** on the same band:

       ```python
       band = dilate(mask) XOR erode(mask)          # elliptical kernel, stroke_width

       laplacian         = Laplacian(gray, ksize=3)[band].var()
       tenengrad         = sqrt(SobelX(gray)^2 + SobelY(gray)^2)[band].mean()
       ```

       Both are implemented via helpers ``compute_boundary_blur_variance`` and
       ``compute_boundary_tenengrad`` from ``vincent_utils``.

    2. **Quality-scaled scores**<br>
       Each stat is scaled against a fixed global anchor so it is comparable
       across datasets:

       ```python
       lap_score      = min(laplacian / max_variance,      1.0)
       ten_score      = min(tenengrad   / max_tenengrad,    1.0)
       score          = 0.5 * lap_score     + 0.5 * ten_score
       ```

    3. **Threshold-based Filter (absolute garbage rejection)**<br>
       Each metric independently rejects frames whose boundary sharpness falls below
       its absolute floor — values ``< hard_min_variance`` (Laplacian) or
       ``< hard_min_tenengrad`` (Tenengrad) are unusable regardless of the population:

       ```python
       if stat < hard_min_stat:
           return (score, False, "blur_threshold")
       ```

    4. **Population-based Filter (relative outlier rejection)**<br>
       Robust median/MAD z-score of each raw stat, fit once over the population
       (``fit``, only when ``outlier_z`` is set):

       ```python
       z = (stat - median) / robust_scale
       z <= -outlier_z      -> reject with reason "blur_outlier"
       ```

       Boundary sharpness is a continuous spectrum, so the "low_bad" (blurred) tail
       is where the noticeably-bad outliers live.

    5. **Pass**<br>
       When both metrics pass threshold and outlier criteria the filter returns the
       quality-scaled combined score with ``passed = True`` and reason ``"blur"``.
    """

    LAP_MAX_VARIANCE      = 10000.0
    TEN_MAX_TENEGRAD      = 100.0
    DEFAULT_HARD_MIN_VAR  = 120.0      # absolute floor on raw Laplacian variance
    DEFAULT_HARD_MIN_TEN  = 35.0       # absolute floor on raw Tenengrad

    lap_stat_attr     = "laplacian"
    lap_weight_attr   = "vincent_boundary_blur_variance"
    ten_stat_attr     = "tenengrad"
    ten_weight_attr   = "bound_tenengrad"
    reason            = "blur"

    def __init__(
        self,
        stroke_width: int = 9,
        max_variance: float = LAP_MAX_VARIANCE,
        hard_min_variance: float = DEFAULT_HARD_MIN_VAR,
        max_tenengrad: float = TEN_MAX_TENEGRAD,
        hard_min_tenengrad: float = DEFAULT_HARD_MIN_TEN,
        outlier_z: float | None = None,
        threshold_min: float | None = None,       # kept for compatibility
        enabled: bool = True,
    ):

        super().__init__(enabled)

        self.stroke_width   = stroke_width
        self.max_variance   = max_variance         # Laplacian anchor
        self.hard_min_variance    = hard_min_variance
        self.max_tenengrad    = max_tenengrad      # Tenengrad anchor
        self.hard_min_tenengrad = hard_min_tenengrad
        self.outlier_z      = outlier_z
        self.threshold_min  = threshold_min        # kept for compatibility

        # (median, robust_scale) of each raw stat, fit over the population
        self._lap_robust    = None
        self._ten_robust    = None

    def requires_fit(self) -> bool:
        """Population pass needed for the outlier mode."""
        return self.outlier_z is not None

    # ------------------------------------------------------------------ #
    # Population statistics
    # ------------------------------------------------------------------ #

    def fit(self, observations):
        """Population pass: robust median/MAD of each raw blur stat.

        Skipped unless ``outlier_z`` is set.  Mirrors ``BorderLaplacianBlurFilter.fit``
        and ``VincentsMotionBlurFilter.fit``: robust statistics are fit over the population
        before the per-observation pass so ``evaluate`` can compare each stat against
        the distribution.
        """
        if self.outlier_z is None:
            return

        lap_values  = []
        ten_values  = []

        for obs in observations:
            if not self.enabled:
                continue
            foreground = mask_to_foreground(obs.mask)
            image      = obs.image
            if image is None:
                continue
            gray     = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lap_values.append(
                compute_boundary_blur_variance(gray, foreground, self.stroke_width)
            )
            ten_values.append(
                compute_boundary_tenengrad(gray, foreground, self.stroke_width)
            )

        if lap_values:
            median, s = robust_center_scale(np.array(lap_values, dtype=float))
            self._lap_robust = (median, max(s, 1.0))

        if ten_values:
            median, s = robust_center_scale(np.array(ten_values, dtype=float))
            self._ten_robust = (median, max(s, 1.0))

    # ------------------------------------------------------------------ #
    # Per-observation evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, observation):
        """Compute both stats, apply threshold + population rejection, return combined score."""
        if not self.enabled:
            return -1.0, True, ""

        # ----------------------------------------------------------- #
        # Laplacian path
        # ----------------------------------------------------------- #
        lap_passed, lap_score  = self._eval_laplacian(observation)

        # ----------------------------------------------------------- #
        # Tenengrad path
        # ----------------------------------------------------------- #
        ten_passed, ten_score  = self._eval_tenengrad(observation)

        # ----------------------------------------------------------- #
        # Combined score & final pass / fail decision
        # ----------------------------------------------------------- #
        score = 0.5 * lap_score + 0.5 * ten_score

        if not lap_passed:
            return score, False, f"{self.reason}_threshold"

        if not ten_passed:
            return score, False, f"{self.reason}_threshold"

        return score, True, self.reason

    def _eval_laplacian(self, observation):
        """Laplacian path: publish stat, check threshold, check population outlier."""
        foreground = mask_to_foreground(observation.mask)
        image      = observation.image

        if image is None:
            stat = 0.0
        else:
            gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            stat   = compute_boundary_blur_variance(gray, foreground, self.stroke_width)

        # Publish stat on metrics (used by downstream scorers & weights pass)
        setattr(observation.metrics, self.lap_stat_attr, stat)
        setattr(observation.metrics, self.lap_weight_attr, stat)

        # Quality-scaled score for this metric
        score = min(stat / self.max_variance, 1.0)

        # --- Absolute garbage floor ---
        if self.hard_min_variance > 0.0 and stat < self.hard_min_variance:
            return False, 0.0

        # --- Population-based outlier ---
        if self.outlier_z is not None and self._lap_robust is not None:
            median, robust_scale = self._lap_robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return False, score

        return True, score

    def _eval_tenengrad(self, observation):
        """Tenengrad path: publish stat, check threshold, check population outlier."""
        foreground = mask_to_foreground(observation.mask)
        image      = observation.image

        if image is None:
            stat = 0.0
        else:
            gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            stat   = compute_boundary_tenengrad(gray, foreground, self.stroke_width)

        # Publish stat on metrics
        setattr(observation.metrics, self.ten_stat_attr, stat)
        setattr(observation.metrics, self.ten_weight_attr, stat)

        # Quality-scaled score for this metric
        score = min(stat / self.max_tenengrad, 1.0)

        # --- Absolute garbage floor ---
        if self.hard_min_tenengrad > 0.0 and stat < self.hard_min_tenengrad:
            return False, 0.0

        # --- Population-based outlier ---
        if self.outlier_z is not None and self._ten_robust is not None:
            median, robust_scale = self._ten_robust
            z = (stat - median) / robust_scale
            if z <= -self.outlier_z:
                return False, score

        return True, score
