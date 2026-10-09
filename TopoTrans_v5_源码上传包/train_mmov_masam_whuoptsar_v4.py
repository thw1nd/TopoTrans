"""Train MMOVMASamV4 on WHUOPTSAR with explicit unknown-negative loss."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmov_seg.mmov_masam_v4 import MMOVMASamV4
from mmov_seg.whuoptsar_dataset import (
    WHUOPTSAR_SEEN_INDEXES,
    WHUOPTSAR_TEST_CLASSES,
    WHUOPTSAR_TRAIN_CLASSES,
    WHUOPTSAR_UNSEEN_INDEXES,
)
from mmov_seg.whuoptsar_dataset_v4 import WHUOPTSARDatasetV4
from train_mmov_masam_whuoptsar import (
    append_metrics,
    build_class_aware_sampler,
    build_optimizer,
    evaluate,
    format_metrics,
    hmean_from_metrics,
    load_config,
    save_checkpoint,
)


def build_dataset(cfg, train: bool, max_samples: int):
    image_size = int(cfg.get("image_size", 384))
    return WHUOPTSARDatasetV4(
        cfg["data_root"],
        split=cfg.get("train_split" if train else "test_split", "all"),
        classes=WHUOPTSAR_TRAIN_CLASSES if train else WHUOPTSAR_TEST_CLASSES,
        split_file=cfg.get("train_split_file" if train else "test_split_file", ""),
        max_samples=max_samples,
        ignore_value=int(cfg["ignore_value"]),
        image_size=(image_size, image_size),
        train=train,
        tile_stride=int(cfg.get("eval_tile_stride", image_size)),
        augment_rotate_degrees=float(cfg.get("augment_rotate_degrees", 0.0)),
    )


def build_model(cfg):
    return MMOVMASamV4(
        train_class_json=cfg["train_class_json"],
        test_class_json=cfg["test_class_json"],
        masam_weights=cfg["masam_weights"],
        num_classes=int(cfg["num_classes"]),
        ignore_value=int(cfg["ignore_value"]),
        freeze_masam=bool(cfg.get("freeze_masam", True)),
        masam_weights_type=cfg.get("masam_weights_type", "sam2_trunk"),
        clip_finetune=cfg.get("clip_finetune", "attention"),
        predictor_version=cfg.get("predictor_version", "v3"),
        prompt_ensemble_type=cfg.get("prompt_ensemble_type", "single"),
        unknown_negative_weight=float(cfg.get("unknown_negative_weight", 1.0)),
        v2_margin_threshold=float(cfg.get("v2_margin_threshold", 0.04)),
        v2_margin_sharpness=float(cfg.get("v2_margin_sharpness", 40.0)),
        v2_affinity_channels=int(cfg.get("v2_affinity_channels", 32)),
        v2_guided_kernel=int(cfg.get("v2_guided_kernel", 3)),
        v2_guided_temperature=float(cfg.get("v2_guided_temperature", 0.07)),
        v2_guided_spatial_sigma=float(cfg.get("v2_guided_spatial_sigma", 1.0)),
        v2_guided_max_strength=float(cfg.get("v2_guided_max_strength", 1.0)),
        v2_guided_initial_strength=float(cfg.get("v2_guided_initial_strength", 0.01)),
        v2_guided_detach_margin=bool(cfg.get("v2_guided_detach_margin", True)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/RSOVSS_WHUOPTSAR_v4_unknown_negative.yaml",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.max_test_samples is not None:
        cfg["max_test_samples"] = args.max_test_samples

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    train_set = build_dataset(cfg, True, int(cfg.get("max_train_samples", 0)))
    test_set = build_dataset(cfg, False, int(cfg.get("max_test_samples", 0)))
    train_sampler = build_class_aware_sampler(train_set, cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )
    print(
        f"WHUOPTSAR v4 protocol: train={len(train_set)} tiles, seen={WHUOPTSAR_TRAIN_CLASSES} -> "
        f"test={len(test_set)} tiles, eval={WHUOPTSAR_TEST_CLASSES}, "
        f"unknown={[WHUOPTSAR_TEST_CLASSES[i] for i in WHUOPTSAR_UNSEEN_INDEXES]}, "
        f"unknown_negative_weight={cfg.get('unknown_negative_weight', 1.0)}"
    )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))

    if args.eval_only:
        metrics = evaluate(
            model,
            test_loader,
            device,
            int(cfg["num_classes"]),
            int(cfg["ignore_value"]),
        )
        print(format_metrics(metrics))
        with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        return

    epochs = int(cfg["epochs"])
    best_hmean = -1.0
    best_path = output_dir / "best.pt"
    if best_path.exists():
        try:
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_hmean = hmean_from_metrics(best_checkpoint.get("metrics", {}))
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            print(f"Warning: failed to read existing best checkpoint ({exc}).")

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "seen_loss": 0.0,
            "unknown_negative_loss": 0.0,
            "seen_pixel_fraction": 0.0,
            "unknown_pixel_fraction": 0.0,
        }
        progress = tqdm(train_loader, desc=f"v4 epoch {epoch}/{epochs}", leave=True)
        for batch in progress:
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            unknown_mask = batch["unknown_mask"].to(device, non_blocking=True)
            losses = model(rgb, sar, mask, unknown_mask)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            for key in totals:
                if key != "loss":
                    totals[key] += float(model.last_loss_stats[key].item())
            steps = max(1, progress.n)
            progress.set_postfix(
                loss=f"{totals['loss'] / steps:.5f}",
                seen=f"{totals['seen_loss'] / steps:.5f}",
                unknown=f"{totals['unknown_negative_loss'] / steps:.5f}",
            )

        steps = max(1, len(train_loader))
        train_stats = {key: value / steps for key, value in totals.items()}
        metrics = evaluate(
            model,
            test_loader,
            device,
            int(cfg["num_classes"]),
            int(cfg["ignore_value"]),
            desc=f"v4 full eval {epoch}/{epochs}",
        )
        print(
            f"v4 epoch={epoch}/{epochs} loss={train_stats['loss']:.5f} "
            f"seen_loss={train_stats['seen_loss']:.5f} "
            f"unknown_negative_loss={train_stats['unknown_negative_loss']:.5f} "
            f"{format_metrics(metrics)}",
            flush=True,
        )
        is_best = metrics["hmean"] > best_hmean
        if is_best:
            best_hmean = metrics["hmean"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)
        append_metrics(output_dir / "metrics_history.jsonl", epoch, train_stats["loss"], metrics, is_best)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
