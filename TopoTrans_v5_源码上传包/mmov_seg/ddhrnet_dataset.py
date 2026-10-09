from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DDHR_TEST_CLASSES = ["city", "road", "forest", "water", "farmland"]
DDHR_TRAIN_CLASSES = ["city", "forest", "farmland"]


def _to_rgb_tensor(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _label_to_contiguous(mask: np.ndarray, classes: Sequence[str], ignore_value: int = 255) -> np.ndarray:
    out = np.full(mask.shape, ignore_value, dtype=np.int64)
    if list(classes) == DDHR_TRAIN_CLASSES:
        mapping = {0: 0, 2: 1, 4: 2}
    else:
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    for src, dst in mapping.items():
        out[mask == src] = dst
    return out


class DDHRNetDataset(Dataset):
    """DDHRNet optical/SAR semantic segmentation split used by MM-OVSeg."""

    def __init__(
        self,
        root: str,
        region: str = "korea",
        weather: str = "cloud",
        split: str = "all",
        classes: Sequence[str] = DDHR_TEST_CLASSES,
        max_samples: int = 0,
        ignore_value: int = 255,
        split_file: str = "",
    ):
        self.root = Path(root)
        self.region = region
        self.weather = weather
        self.split = split
        self.classes = list(classes)
        self.max_samples = int(max_samples)
        self.ignore_value = int(ignore_value)
        self.split_file = split_file

        base = self.root / region / weather
        self.rgb_dir = base / "GF2"
        self.sar_dir = base / "GF3"
        self.label_dir = base / "label"
        if not self.rgb_dir.is_dir() or not self.sar_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Invalid DDHRNet split: {base}")

        label_by_stem = {p.stem: p for p in self.label_dir.glob("*.png")}
        samples: List[Dict[str, Path]] = []
        for rgb_path in sorted(self.rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            sar_path = self.sar_dir / f"{rgb_path.stem}.jpg"
            label_path = label_by_stem.get(rgb_path.stem)
            if sar_path.is_file() and label_path is not None:
                samples.append({"rgb": rgb_path, "sar": sar_path, "label": label_path})
        if not samples:
            raise RuntimeError(f"No DDHRNet samples found under {base}")
        samples = self._apply_split(samples)
        if self.max_samples > 0:
            samples = samples[: self.max_samples]
        self.samples = samples

    def _apply_split(self, samples: List[Dict[str, Path]]):
        if self.split_file:
            split_path = Path(self.split_file)
            with split_path.open("r", encoding="utf-8") as handle:
                stems = [line.strip() for line in handle if line.strip()]
            sample_by_stem = {sample["rgb"].stem: sample for sample in samples}
            missing = [stem for stem in stems if stem not in sample_by_stem]
            if missing:
                raise FileNotFoundError(f"{len(missing)} stems from {split_path} are missing, e.g. {missing[:5]}")
            selected = [sample_by_stem[stem] for stem in stems]
        elif self.split == "all":
            selected = samples
        elif self.region == "korea" and self.split == "train":
            selected = samples[:3087]
        elif self.region == "korea" and self.split in ("val", "test"):
            selected = samples[3087:]
        elif self.split in ("val", "test"):
            selected = samples
        else:
            raise ValueError(
                f"Unsupported DDHRNet split '{self.split}' for region '{self.region}'. "
                "Use korea/train, korea/val, or all/test for cross-domain regions."
            )
        return selected

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rgb = _to_rgb_tensor(sample["rgb"])
        sar = _to_rgb_tensor(sample["sar"])
        mask = np.asarray(Image.open(sample["label"]), dtype=np.int64)
        mask = _label_to_contiguous(mask, self.classes, self.ignore_value)
        return {
            "rgb": rgb,
            "sar": sar,
            "mask": torch.from_numpy(mask),
            "file_name": str(sample["rgb"]),
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
