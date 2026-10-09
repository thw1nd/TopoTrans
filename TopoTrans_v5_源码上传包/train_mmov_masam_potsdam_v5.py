"""Train the isolated full MM-OVSeg-MASam V5 model on Potsdam.

V5 combines margin-guided semantic graph refinement (MG-SGR) with the
multi-scale auxiliary high-resolution refinement (MAHR) path.  This entry
point is intentionally separate from the existing v4 and ablation runners.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmov_seg.checkpointing import load_verified_checkpoint, save_verified_checkpoint
from mmov_seg.mmov_masam_v5 import MMOVMASamV5
from mmov_seg.potsdam_dataset import POTSDAM_TEST_CLASSES, POTSDAM_TRAIN_CLASSES
from train_mmov_masam_potsdam import build_optimizer, evaluate, format_metrics, load_config
from train_mmov_masam_potsdam_v4 import build_dataset


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
    parser.add_argument("--config", default="configs/RSOVSS_Potsdam_v5.yaml")
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

    output_dir = Path(cfg["output_root"])
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
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
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
        f"Potsdam V5 protocol: train={len(train_set)}, test={len(test_set)}, "
        f"seen={POTSDAM_TRAIN_CLASSES}, eval={POTSDAM_TEST_CLASSES}, "
        f"epochs={cfg['epochs']}"
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
        report = checkpoint["load_report"]
        print(
            f"checkpoint={args.resume} strict={report['strict']} "
            f"missing={len(report['missing_keys'])} unexpected={len(report['unexpected_keys'])}"
        )

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, int(cfg["num_classes"]), int(cfg["ignore_value"]))
        print("FORMAL_RELOADED_METRICS " + format_metrics(metrics))
        (output_dir / "evaluation_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "checkpoint_verification.json").write_text(
            json.dumps(checkpoint.get("load_report", {}), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    best_path = output_dir / "best.pt"
    best_hmean = -1.0
    if best_path.exists():
        prior = torch.load(best_path, map_location="cpu")
        best_hmean = float(prior.get("metrics", {}).get("hmean", -1.0))

    for epoch in range(start_epoch + 1, int(cfg["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "seen_loss": 0.0, "unknown_negative_loss": 0.0}
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
            totals["seen_loss"] += float(model.last_loss_stats["seen_loss"].item())
            totals["unknown_negative_loss"] += float(model.last_loss_stats["unknown_negative_loss"].item())
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
            f"v5 epoch={epoch} loss={train_stats['loss']:.5f} "
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
                variant="potsdam_v5_mgsr_mahr",
                config_path=args.config,
            )
        save_verified_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            metrics,
            variant="potsdam_v5_mgsr_mahr",
            config_path=args.config,
        )
        record = {"epoch": epoch, "is_best_hmean": is_best, **train_stats, **metrics}
        with (output_dir / "metrics_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
