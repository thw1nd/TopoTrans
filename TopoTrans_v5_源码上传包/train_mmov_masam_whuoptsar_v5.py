"""Train isolated MM-OVSeg-MASam V5 on WHUOPTSAR."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmov_seg.mmov_masam_v5 import MMOVMASamV5
from mmov_seg.checkpointing import (
    load_verified_checkpoint,
    save_verified_checkpoint,
)
from mmov_seg.whuoptsar_dataset import (
    WHUOPTSAR_TEST_CLASSES,
    WHUOPTSAR_TRAIN_CLASSES,
    WHUOPTSAR_UNSEEN_INDEXES,
)
from train_mmov_masam_whuoptsar import (
    append_metrics,
    build_class_aware_sampler,
    build_optimizer,
    evaluate,
    format_metrics,
    hmean_from_metrics,
    load_config,
)
from train_mmov_masam_whuoptsar_v4 import build_dataset


def build_model(cfg):
    return MMOVMASamV5(
        train_class_json=cfg["train_class_json"],
        test_class_json=cfg["test_class_json"],
        masam_weights=cfg["masam_weights"],
        num_classes=int(cfg["num_classes"]),
        ignore_value=int(cfg["ignore_value"]),
        freeze_masam=bool(cfg.get("freeze_masam", True)),
        masam_weights_type=cfg.get("masam_weights_type", "sam2_trunk"),
        clip_finetune=cfg.get("clip_finetune", "attention"),
        prompt_ensemble_type=cfg.get("prompt_ensemble_type", "single"),
        unknown_negative_weight=float(cfg.get("unknown_negative_weight", 1.0)),
        v2_margin_threshold=float(cfg.get("v2_margin_threshold", 0.04)),
        v2_margin_sharpness=float(cfg.get("v2_margin_sharpness", 40.0)),
        v2_guided_kernel=int(cfg.get("v2_guided_kernel", 3)),
        v2_guided_temperature=float(cfg.get("v2_guided_temperature", 0.07)),
        v2_guided_spatial_sigma=float(cfg.get("v2_guided_spatial_sigma", 1.0)),
        v2_guided_max_strength=float(cfg.get("v2_guided_max_strength", 1.0)),
        v2_guided_initial_strength=float(cfg.get("v2_guided_initial_strength", 0.01)),
        v2_guided_detach_margin=bool(cfg.get("v2_guided_detach_margin", True)),
        relation_dim=int(cfg.get("relation_dim", 64)),
        relation_temperature=float(cfg.get("relation_temperature", 0.07)),
        relation_max_strength=float(cfg.get("relation_max_strength", 1.0)),
        relation_initial_strength=float(cfg.get("relation_initial_strength", 0.1)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/RSOVSS_WHUOPTSAR_v5_mgsr.yaml")
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
        f"WHUOPTSAR V5 MG-SGR protocol: train={len(train_set)}, test={len(test_set)}, "
        f"seen={WHUOPTSAR_TRAIN_CLASSES}, eval={WHUOPTSAR_TEST_CLASSES}, "
        f"unseen={[WHUOPTSAR_TEST_CLASSES[i] for i in WHUOPTSAR_UNSEEN_INDEXES]}, "
        f"unknown_negative_weight={cfg.get('unknown_negative_weight', 1.0)}"
    )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    start_epoch = 0
    checkpoint = {}
    if args.resume:
        checkpoint = load_verified_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            restore_optimizer=not args.eval_only,
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"checkpoint={args.resume} strict={checkpoint['load_report']['strict']} "
            f"missing={len(checkpoint['load_report']['missing_keys'])} "
            f"unexpected={len(checkpoint['load_report']['unexpected_keys'])}"
        )

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, int(cfg["num_classes"]), int(cfg["ignore_value"]))
        print("FORMAL_RELOADED_METRICS " + format_metrics(metrics))
        with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        with (output_dir / "checkpoint_verification.json").open("w", encoding="utf-8") as handle:
            json.dump(checkpoint.get("load_report", {}), handle, indent=2, ensure_ascii=False)
        return

    best_path = output_dir / "best.pt"
    best_hmean = -1.0
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu")
        best_hmean = hmean_from_metrics(checkpoint.get("metrics", {}))

    for epoch in range(start_epoch + 1, int(cfg["epochs"]) + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "seen_loss": 0.0,
            "unknown_negative_loss": 0.0,
            "seen_pixel_fraction": 0.0,
            "unknown_pixel_fraction": 0.0,
        }
        progress = tqdm(train_loader, desc=f"v5 epoch {epoch}/{cfg['epochs']}", leave=True)
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
            progress.set_postfix(loss=f"{totals['loss'] / steps:.5f}")

        steps = max(1, len(train_loader))
        train_stats = {key: value / steps for key, value in totals.items()}
        metrics = evaluate(
            model,
            test_loader,
            device,
            int(cfg["num_classes"]),
            int(cfg["ignore_value"]),
            desc=f"v5 eval {epoch}/{cfg['epochs']}",
        )
        print(
            f"v5 [ONLINE-ONLY] epoch={epoch}/{cfg['epochs']} loss={train_stats['loss']:.5f} "
            f"seen_loss={train_stats['seen_loss']:.5f} "
            f"unknown_negative_loss={train_stats['unknown_negative_loss']:.5f} "
            f"{format_metrics(metrics)}",
            flush=True,
        )
        is_best = metrics["hmean"] > best_hmean
        if is_best:
            best_hmean = metrics["hmean"]
            save_verified_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                metrics,
                variant="v5_mgsr_mahr",
                config_path=args.config,
            )
        append_metrics(output_dir / "metrics_history.jsonl", epoch, train_stats["loss"], metrics, is_best)
        save_verified_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            metrics,
            variant="v5_mgsr_mahr",
            config_path=args.config,
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
