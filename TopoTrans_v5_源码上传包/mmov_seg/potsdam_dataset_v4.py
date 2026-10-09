"""Potsdam dataset wrapper that preserves raw unseen-pixel locations for v4."""

import torch

from .potsdam_dataset import PotsdamDataset


class PotsdamDatasetV4(PotsdamDataset):
    """Return the regular mapped target plus a raw unseen mask.

    Raw class IDs 2 (low vegetation) and 3 (tree) form ``unknown_mask``.
    Raw ID 255 is official clutter and deliberately remains excluded from both
    the seen loss and the v4 unknown-negative loss.
    """

    RAW_UNSEEN_IDS = (2, 3)

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        sample = self.samples[index]
        _, _, raw_mask = self._load_scene(sample)
        top, left = int(sample["top"]), int(sample["left"])
        size = self.image_size
        raw_crop = raw_mask[top : top + size, left : left + size]
        unknown_mask = (raw_crop == self.RAW_UNSEEN_IDS[0]) | (
            raw_crop == self.RAW_UNSEEN_IDS[1]
        )
        item["unknown_mask"] = torch.from_numpy(unknown_mask.copy()).bool()
        return item

