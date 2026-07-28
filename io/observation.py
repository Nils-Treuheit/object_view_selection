from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Observation:

    id: int

    image_path: Path
    mask_path: Path
    object_hand_path: Optional[Path]

    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    object_hand: Optional[np.ndarray] = None

    quality: float = 0.0

    embedding: Optional[np.ndarray] = None

    rejected: bool = False

    rejection_reason: Optional[str] = None

    metrics: ObservationMetrics = field(
    	default_factory=ObservationMetrics
    )


