"""WHUOPTSAR dataset wrapper for MMOVMASamV4 unknown-negative training."""

import random

import numpy as np
import torch
from PIL import Image

from .whuoptsar_dataset import (
    WHUOPTSARDataset,
    _crop_arrays,
    _to_image_tensor,
    _to_mask_tensor,
)


class WHUOPTSARDatasetV4(WHUOPTSARDataset):
    """Return the regular target plus a mask for raw WHU unseen classes.

    WHUOPTSAR raw labels use IDs 1 (city) and 3 (water) for the two unseen
    classes. They remain ignore pixels in the five-class training target, but
    are exposed separately so MMOVMASamV4 can apply the unknown-negative loss.
    Raw 255 pixels remain ignored and are not included in ``unknown_mask``.
    """

    RAW_UNSEEN_IDS = (1, 3)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        cache_key = str(sample["rgb"])
        cached = self._array_cache.get(cache_key)
        if cached is None:
            with Image.open(sample["rgb"]) as image:
                rgb = np.array(image.convert("RGB"), copy=True)
            with Image.open(sample["sar"]) as image:
                sar = np.array(image.convert("RGB"), copy=True)
            with Image.open(sample["label"]) as image:
                raw_mask = np.array(image, dtype=np.int64, copy=True)
            cached = (rgb, sar, raw_mask)
            self._array_cache[cache_key] = cached

        rgb, sar, raw_mask = cached
        top = int(sample["top"])
        left = int(sample["left"])
        rgb, sar, raw_crop = _crop_arrays(
            rgb,
            sar,
            raw_mask,
            top,
            left,
            self.image_size,
            self.ignore_value,
        )
        unknown_mask = np.isin(raw_crop, self.RAW_UNSEEN_IDS)

        if self.train and self.augment_rotate_degrees > 0:
            angle = random.uniform(-self.augment_rotate_degrees, self.augment_rotate_degrees)
            rgb = np.asarray(
                Image.fromarray(rgb).rotate(
                    angle,
                    resample=Image.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
            )
            sar = np.asarray(
                Image.fromarray(sar).rotate(
                    angle,
                    resample=Image.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
            )
            raw_crop = np.asarray(
                Image.fromarray(raw_crop.astype(np.uint8)).rotate(
                    angle,
                    resample=Image.NEAREST,
                    fillcolor=self.ignore_value,
                ),
                dtype=np.int64,
            )
            unknown_mask = np.isin(raw_crop, self.RAW_UNSEEN_IDS)

        return {
            "rgb": _to_image_tensor(rgb),
            "sar": _to_image_tensor(sar),
            "mask": _to_mask_tensor(raw_crop, self.classes, self.ignore_value),
            "unknown_mask": torch.from_numpy(unknown_mask.copy()).bool(),
            "file_name": str(sample["rgb"]),
        }
