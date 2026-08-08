import cv2

from .base import ScoreFilter
from .vincent_utils import (
    compute_boundary_blur_variance,
    compute_boundary_tenengrad,
    mask_to_foreground,
)

# --------------------------------------------------------------------------- #
# True module globals: the default values live here and are passed through the
# filter constructors (they are deliberately not defined as class attributes).
# --------------------------------------------------------------------------- #

BORDER_BLUR_STROKE_WIDTH = 9
DEFAULT_MAX_VARIANCE = 20000.0
DEFAULT_HARD_MIN_VARIANCE = 4000.0

DEFAULT_MAX_TENENGRAD = 150.0
DEFAULT_HARD_MIN_TENENGRAD = 33.0


class BorderLaplacianBlurFilter(ScoreFilter):
    """Boundary-band sharpness pre-filter (Laplacian variance).

    Stat: variance of the Laplacian restricted to the band straddling the mask
    contour (``compute_boundary_blur_variance``), stored on ``metrics.laplacian``
    (and mirrored on ``metrics.vincent_boundary_blur_variance``). Higher =
    sharper object/background transition.

    Implements both rejection criteria mandated by ``BaseFilter`` via
    ``ScoreFilter``: an absolute threshold-based garbage floor
    (``hard_min_variance``) on the raw stat, and a population-based
    extreme-bad-outlier removal (``outlier_z``) using robust median/MAD
    z-scores. Returns a (0, 1] goodness score anchored at ``max_variance``
    when the observation passes both criteria.
    """

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_variance: float = DEFAULT_MAX_VARIANCE,
        hard_min_variance: float = DEFAULT_HARD_MIN_VARIANCE,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(
            enabled=enabled,
            hard_min=hard_min_variance,
            outlier_z=outlier_z,
            stat_attr="laplacian",
            reason="blur_laplacian",
            direction="low_bad",
            metric_aliases=("vincent_boundary_blur_variance",),
        )

        self.stroke_width = stroke_width
        self.max_variance = max_variance

    def compute_stat(self, observation) -> float:
        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return compute_boundary_blur_variance(gray, foreground, self.stroke_width)

    def compute_score(self, stat: float) -> float:
        """Stat scaled against a fixed global anchor (``max_variance``)."""
        return min(stat / self.max_variance, 1.0)


class BorderTenengradBlurFilter(ScoreFilter):
    """Boundary-band sharpness pre-filter (Tenengrad).

    Stat: mean Sobel magnitude restricted to the boundary band
    (``compute_boundary_tenengrad``), stored on ``metrics.tenengrad``.
    Higher = sharper structured gradients at the object contour.

    Companion to ``BorderLaplacianBlurFilter`` implementing both rejection
    criteria mandated by ``BaseFilter`` via ``ScoreFilter``: an absolute
    threshold-based garbage floor (``hard_min_tenengrad``) on the raw stat,
    and a population-based extreme-bad-outlier removal (``outlier_z``) using
    robust median/MAD z-scores. Returns a (0, 1] goodness score anchored at
    ``max_tenengrad`` when passed.
    """

    def __init__(
        self,
        stroke_width: int = BORDER_BLUR_STROKE_WIDTH,
        max_tenengrad: float = DEFAULT_MAX_TENENGRAD,
        hard_min_tenengrad: float = DEFAULT_HARD_MIN_TENENGRAD,
        outlier_z: float | None = None,
        enabled=True,
    ):

        super().__init__(
            enabled=enabled,
            hard_min=hard_min_tenengrad,
            outlier_z=outlier_z,
            stat_attr="tenengrad",
            reason="blur_tenengrad",
            direction="low_bad",
        )

        self.stroke_width = stroke_width
        self.max_tenengrad = max_tenengrad

    def compute_stat(self, observation) -> float:
        foreground = mask_to_foreground(observation.mask)

        image = observation.image
        if image is None:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return compute_boundary_tenengrad(gray, foreground, self.stroke_width)

    def compute_score(self, stat: float) -> float:
        """Stat scaled against a fixed global anchor (``max_tenengrad``)."""
        return min(stat / self.max_tenengrad, 1.0)
