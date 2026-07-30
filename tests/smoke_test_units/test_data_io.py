"""
Smoke tests for data I/O module.
"""

import numpy as np

from tests.smoke_test_utils import check


DATA_ROOT = "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/bottle"


def test_data_io(data_root=DATA_ROOT):
    from data_io.dataset import Dataset
    from data_io.observation import Observation

    ds = Dataset(data_root)
    check(len(ds) > 0, f"Dataset loaded {len(ds)} observations")

    obs = ds.observations[0]
    check(isinstance(obs, Observation), "Observation is proper type")
    check(obs.image_path.exists(), f"Image path exists: {obs.image_path.name}")
    check(obs.mask_path.exists(), f"Mask path exists: {obs.mask_path.name}")

    ds.load_images()
    check(obs.image is not None, f"Image loaded, shape={obs.image.shape}")
    check(obs.mask is not None, f"Mask loaded, shape={obs.mask.shape}")
    check(obs.image.shape[:2] == obs.mask.shape[:2], "Image/mask dimensions match")
    check(obs.mask.dtype == np.uint8, "Mask is uint8")
    check(obs.image.shape[2] == 3, "Image is RGB")


DATA_IO_TESTS = [
    ("Data loading and observation", test_data_io),
]
