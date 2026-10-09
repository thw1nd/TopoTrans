"""Export isolated MM-OVSeg-MASam V5 WHUOPTSAR visualizations."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Support direct execution via ``python tools/...py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_mmov_masam_whuoptsar_v5 import build_dataset, build_model, load_config


PALETTE = np.asarray(
    [
        [0, 102, 204],
        [0, 0, 255],
        [0, 255, 255],
        [255, 0, 0],
        [0, 167, 85],
        [255, 255, 0],
        [153, 102, 153],
    ],
    dtype=np.uint8,
)


def colorize(mask):
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        image[mask == class_id] = color
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/RSOVSS_WHUOPTSAR_v5_mgsr.yaml")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, train=False, max_samples=int(cfg.get("max_test_samples", 0)))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )
    model = build_model(cfg).to(device).eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "GT"
    prediction_dir = output_dir / "prediction"
    gt_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, batch in enumerate(tqdm(loader, desc="MMOVMASamV5 WHUOPTSAR", unit="patch")):
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            prediction = model(rgb, sar).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            target = batch["mask"][0].cpu().numpy().astype(np.int64)
            colorize(target).save(gt_dir / f"{index:05d}.png")
            colorize(prediction).save(prediction_dir / f"{index:05d}.png")
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
