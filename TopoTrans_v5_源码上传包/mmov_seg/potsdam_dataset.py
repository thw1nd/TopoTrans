"""Potsdam RGB/DSM dataset for the MMOV-Seg-MASam OVSS protocol.

The active ``label`` files contain five foreground classes and 255 as ignore:
impervious surface, building, low vegetation, tree and car.  The official
clutter/background class is already encoded as 255 in this label variant.
Images are tiled without overlap and incomplete right/bottom tiles are
discarded, as required by the Potsdam protocol used in this project.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Image.MAX_IMAGE_PIXELS = None

POTSDAM_TEST_CLASSES = [
    "impervious surface",
    "building",
    "low vegetation",
    "tree",
    "car",
]
POTSDAM_TRAIN_CLASSES = [
    "impervious surface",
    "building",
    "car",
]
POTSDAM_SEEN_INDEXES = [0, 1, 4]
POTSDAM_UNSEEN_INDEXES = [2, 3]

_TEST_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
_TRAIN_MAPPING = {0: 0, 1: 1, 4: 2}


def _to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return torch.from_numpy(arr.astype(np.float32, copy=False)).permute(2, 0, 1)


def _to_contiguous_mask(
    mask: np.ndarray,
    classes: Sequence[str],
    ignore_value: int,
) -> torch.Tensor:
    out = np.full(mask.shape, ignore_value, dtype=np.int64)
    mapping = _TRAIN_MAPPING if list(classes) == POTSDAM_TRAIN_CLASSES else _TEST_MAPPING
    for source_id, target_id in mapping.items():
        out[mask == source_id] = target_id
    return torch.from_numpy(out)


def _align_to_reference(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Align a modality to RGB geometry without resizing.

    The local archive contains one DSM whose JPEG decode is 5999x6000 while
    RGB/label are 6000x6000.  A one-pixel edge replication is deterministic
    and avoids distorting DSM values through interpolation.
    """

    arr = arr[:height, :width]
    pad_h = max(0, height - arr.shape[0])
    pad_w = max(0, width - arr.shape[1])
    if pad_h or pad_w:
        if arr.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        arr = np.pad(arr, pad_width, mode="edge")
    return arr


def _tile_positions(height: int, width: int, tile_size: int, stride: int):
    if height < tile_size or width < tile_size:
        return []
    # The range intentionally stops at the last complete tile.  No padding
    # and no partial right/bottom tiles are used in this protocol.
    tops = range(0, height - tile_size + 1, stride)
    lefts = range(0, width - tile_size + 1, stride)
    return [(top, left) for top in tops for left in lefts]


class PotsdamDataset(Dataset):
    """Scene-level Potsdam split tiled into complete non-overlapping patches."""

    def __init__(
        self,
        root: str,
        split: str,
        classes: Sequence[str] = POTSDAM_TEST_CLASSES,
        image_size: int = 384,
        tile_stride: int = 384,
        ignore_value: int = 255,
        cache_scenes: int = 32,
        max_samples: int = 0,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.classes = list(classes)
        self.image_size = int(image_size)
        self.tile_stride = int(tile_stride)
        self.ignore_value = int(ignore_value)
        self.cache_scenes = max(0, int(cache_scenes))
        self._scene_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

        split_root = self.root / self.split
        self.rgb_dir = split_root / "rgb"
        self.dsm_dir = split_root / "dsm"
        self.label_dir = split_root / "label"
        samples: List[Dict[str, Path]] = []
        for rgb_path in sorted(self.rgb_dir.glob("*.tif"), key=lambda p: p.stem):
            dsm_path = self.dsm_dir / f"{rgb_path.stem}.jpg"
            label_path = self.label_dir / f"{rgb_path.stem}.tif"
            if not dsm_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing paired Potsdam files for {rgb_path.stem}: "
                    f"dsm={dsm_path.is_file()} label={label_path.is_file()}"
                )
            samples.append({"rgb": rgb_path, "dsm": dsm_path, "label": label_path})
        if not samples:
            raise RuntimeError(f"No Potsdam scenes found under {split_root}")
        self.scene_samples = samples
        self.samples = self._build_patch_samples(samples)
        if max_samples > 0:
            self.samples = self.samples[: int(max_samples)]

    def _build_patch_samples(self, scenes: List[Dict[str, Path]]):
        patches = []
        for scene in scenes:
            with Image.open(scene["label"]) as image:
                height, width = image.size[1], image.size[0]
            positions = _tile_positions(height, width, self.image_size, self.tile_stride)
            for top, left in positions:
                patches.append({**scene, "top": top, "left": left})
        return patches

    def _load_scene(self, sample: Dict[str, Path]):
        key = sample["rgb"].stem
        cached = self._scene_cache.get(key)
        if cached is not None:
            self._scene_cache.move_to_end(key)
            return cached

        with Image.open(sample["rgb"]) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        with Image.open(sample["dsm"]) as image:
            dsm = np.array(image, copy=True)
        with Image.open(sample["label"]) as image:
            mask = np.array(image, dtype=np.int64, copy=True)

        dsm = _align_to_reference(dsm, rgb.shape[0], rgb.shape[1])
        mask = _align_to_reference(mask, rgb.shape[0], rgb.shape[1])
        cached = (rgb, dsm, mask)
        if self.cache_scenes > 0:
            self._scene_cache[key] = cached
            self._scene_cache.move_to_end(key)
            while len(self._scene_cache) > self.cache_scenes:
                self._scene_cache.popitem(last=False)
        return cached

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rgb, dsm, mask = self._load_scene(sample)
        top, left = int(sample["top"]), int(sample["left"])
        size = self.image_size
        rgb_crop = rgb[top : top + size, left : left + size]
        dsm_crop = dsm[top : top + size, left : left + size]
        mask_crop = mask[top : top + size, left : left + size]
        if rgb_crop.shape[:2] != (size, size) or dsm_crop.shape[:2] != (size, size):
            raise RuntimeError(f"Incomplete Potsdam patch was not discarded: {sample}")
        return {
            "rgb": _to_image_tensor(rgb_crop),
            "sar": _to_image_tensor(dsm_crop),
            "mask": _to_contiguous_mask(mask_crop, self.classes, self.ignore_value),
            "file_name": str(sample["rgb"]),
            "scene_id": sample["rgb"].stem,
            "top": top,
            "left": left,
        }


def fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_value: int = 255):
    pred = pred.detach().view(-1).cpu()
    target = target.detach().view(-1).cpu()
    valid = (target != ignore_value) & (target >= 0) & (target < num_classes)
    if valid.sum() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    bins = num_classes * target[valid].to(torch.int64) + pred[valid].to(torch.int64)
    return torch.bincount(bins, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def miou_from_confmat(confmat: torch.Tensor):
    confmat = confmat.to(torch.float64)
    intersection = torch.diag(confmat)
    union = confmat.sum(1) + confmat.sum(0) - intersection
    valid = union > 0
    iou = torch.zeros_like(union)
    iou[valid] = intersection[valid] / union[valid]
    return iou, iou[valid].mean().item() if valid.any() else 0.0
