import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Image.MAX_IMAGE_PIXELS = None

WHUOPTSAR_TEST_CLASSES = ["farmland", "city", "village", "water", "forest", "road", "others"]
WHUOPTSAR_TRAIN_CLASSES = ["farmland", "village", "forest", "road", "others"]
WHUOPTSAR_SEEN_INDEXES = [0, 2, 4, 5, 6]
WHUOPTSAR_UNSEEN_INDEXES = [1, 3]


def _to_mask_tensor(arr: np.ndarray, classes: Sequence[str], ignore_value: int) -> torch.Tensor:
    out = np.full(arr.shape, ignore_value, dtype=np.int64)
    if list(classes) == WHUOPTSAR_TRAIN_CLASSES:
        mapping = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4}
    else:
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    for src, dst in mapping.items():
        out[arr == src] = dst
    return torch.from_numpy(out)


def _pad_array(arr: np.ndarray, size: Tuple[int, int], value: int):
    height, width = arr.shape[:2]
    target_h, target_w = size
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_spec = ((0, pad_h), (0, pad_w)) if arr.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(arr, pad_spec, mode="constant", constant_values=value)


def _crop_arrays(rgb: np.ndarray, sar: np.ndarray, mask: np.ndarray, top: int, left: int, size: Tuple[int, int], ignore_value: int):
    crop_h, crop_w = size
    rgb_crop = rgb[top: top + crop_h, left: left + crop_w]
    sar_crop = sar[top: top + crop_h, left: left + crop_w]
    mask_crop = mask[top: top + crop_h, left: left + crop_w]
    rgb_crop = _pad_array(rgb_crop, size, 0)
    sar_crop = _pad_array(sar_crop, size, 0)
    mask_crop = _pad_array(mask_crop, size, ignore_value)
    return rgb_crop, sar_crop, mask_crop


def _to_image_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)


def _tile_positions(height: int, width: int, tile_h: int, tile_w: int, stride: int):
    tops = list(range(0, max(height - tile_h + 1, 0), stride))
    lefts = list(range(0, max(width - tile_w + 1, 0), stride))
    return [(top, left) for top in tops for left in lefts]


class WHUOPTSARDataset(Dataset):
    """WHU-OPT-SAR optical/SAR semantic segmentation dataset for OVSS experiments."""

    def __init__(
        self,
        root: str,
        split: str = "all",
        classes: Sequence[str] = WHUOPTSAR_TEST_CLASSES,
        split_file: str = "",
        max_samples: int = 0,
        ignore_value: int = 255,
        image_size: Tuple[int, int] = (384, 384),
        train: bool = False,
        tile_stride: int = 384,
        augment_rotate_degrees: float = 0.0,
    ):
        self.root = Path(root)
        self.split = split
        self.classes = list(classes)
        self.split_file = split_file
        self.max_samples = int(max_samples)
        self.ignore_value = int(ignore_value)
        self.image_size = tuple(image_size)
        self.train = bool(train)
        self.tile_stride = int(tile_stride)
        self.augment_rotate_degrees = float(augment_rotate_degrees)
        # Cache each full scene once in RAM. Without this, every tile reopens
        # and decodes the same multi-megapixel TIFF from the HDD.
        self._array_cache = {}

        if split in ("train", "test") and not split_file and (self.root / split).is_dir():
            base = self.root / split
            opt_dir = base / "opt"
            sar_dir = base / "sar"
            label_dir = base / "labels"
        else:
            opt_dir = self.root / "opt"
            sar_dir = self.root / "sar"
            label_dir = self.root / "new_lbl"

        if not opt_dir.is_dir() or not sar_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Invalid WHUOPTSAR dataset folders under {self.root}")

        label_by_stem = {p.stem: p for p in label_dir.glob("*.tif")}
        samples: List[Dict[str, Path]] = []
        for opt_path in sorted(opt_dir.glob("*.tif"), key=lambda p: p.stem):
            sar_path = sar_dir / opt_path.name
            label_path = label_by_stem.get(opt_path.stem)
            if sar_path.is_file() and label_path is not None:
                samples.append({"rgb": opt_path, "sar": sar_path, "label": label_path})
        if not samples:
            raise RuntimeError(f"No WHUOPTSAR samples found under {self.root}")

        samples = self._apply_split(samples)
        if self.max_samples > 0:
            samples = samples[: self.max_samples]

        self.samples = self._build_tile_samples(samples)

    def _build_tile_samples(self, samples: List[Dict[str, Path]]):
        tile_h, tile_w = self.image_size
        tiled = []
        for sample in samples:
            mask = np.asarray(Image.open(sample["label"]), dtype=np.int64)
            height, width = mask.shape
            for top, left in _tile_positions(height, width, tile_h, tile_w, self.tile_stride):
                crop = mask[top: top + tile_h, left: left + tile_w]
                counts = np.bincount(crop.reshape(-1), minlength=256)[[0, 1, 2, 3, 4, 5, 6, 255]]
                valid = max(1, int(counts[:7].sum()))
                tiled.append({
                    **sample,
                    "top": top,
                    "left": left,
                    "class_ratios": (counts[:7] / valid).astype(np.float32),
                })
        return tiled

    def _apply_split(self, samples: List[Dict[str, Path]]):
        if self.split_file:
            split_path = Path(self.split_file)
            stems = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            sample_by_stem = {sample["rgb"].stem: sample for sample in samples}
            missing = [stem for stem in stems if stem not in sample_by_stem]
            if missing:
                raise FileNotFoundError(f"{len(missing)} stems from {split_path} are missing, e.g. {missing[:5]}")
            return [sample_by_stem[stem] for stem in stems]
        return samples

    def __len__(self):
        return len(self.samples)

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
                mask = np.array(image, dtype=np.int64, copy=True)
            cached = (rgb, sar, mask)
            self._array_cache[cache_key] = cached

        rgb, sar, mask = cached
        top = int(sample["top"])
        left = int(sample["left"])

        rgb, sar, mask = _crop_arrays(rgb, sar, mask, top, left, self.image_size, self.ignore_value)
        if self.train and self.augment_rotate_degrees > 0:
            angle = random.uniform(-self.augment_rotate_degrees, self.augment_rotate_degrees)
            rgb = np.asarray(Image.fromarray(rgb).rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0)))
            sar = np.asarray(Image.fromarray(sar).rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0)))
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).rotate(
                    angle,
                    resample=Image.NEAREST,
                    fillcolor=self.ignore_value,
                ),
                dtype=np.int64,
            )
        return {
            "rgb": _to_image_tensor(rgb),
            "sar": _to_image_tensor(sar),
            "mask": _to_mask_tensor(mask, self.classes, self.ignore_value),
            "file_name": str(sample["rgb"]),
        }
