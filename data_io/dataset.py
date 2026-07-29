from pathlib import Path

import cv2

from .observation import Observation


class Dataset:

    def __init__(self, root: str):

        self.root = Path(root)

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.object_hands_dir = self.root / "object_hands"

        self.observations = []

        self._load()

    def _load(self):

        image_files = sorted(self.images_dir.glob("*.png"))

        for image_file in image_files:

            stem = image_file.stem

            mask_file = self.masks_dir / f"{stem}.png"

            hand_file = self.object_hands_dir / f"{stem}.png"

            if not mask_file.exists():
                continue

            obs = Observation(

                id=int(stem),

                image_path=image_file,

                mask_path=mask_file,

                object_hand_path=hand_file if hand_file.exists() else None,

            )

            self.observations.append(obs)

    def load_images(self):

        for obs in self.observations:

            obs.image = cv2.imread(str(obs.image_path))

            obs.image = cv2.cvtColor(obs.image, cv2.COLOR_BGR2RGB)

            obs.mask = cv2.imread(
                str(obs.mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if obs.object_hand_path:

                obs.object_hand = cv2.imread(
                    str(obs.object_hand_path),
                    cv2.IMREAD_GRAYSCALE,
                )

    def __len__(self):

        return len(self.observations)
