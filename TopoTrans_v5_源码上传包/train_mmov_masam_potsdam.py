"""Train/evaluate MMOV-Seg-MASam on the local ISPRS Potsdam protocol."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from mmov_seg.mmov_masam import MMOVMASam
from mmov_seg.potsdam_dataset import (
    POTSDAM_SEEN_INDEXES,
    POTSDAM_TEST_CLASSES,
    POTSDAM_TRAIN_CLASSES,
    POTSDAM_UNSEEN_INDEXES,
    PotsdamDataset,
    fast_confusion_matrix,
    miou_from_confmat,
)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_optimizer(model, cfg):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = float(cfg["base_lr"])
        if "clip_model" in name:
            lr *= float(cfg.get("clip_lr_multiplier", 0.01))
        params.append({"params": [param], "lr": lr, "weight_decay": float(cfg.get("weight_decay", 1e-4))})
    return torch.optim.AdamW(params, lr=float(cfg["base_lr"]))


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_value, desc="eval"):
    model.eval()
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    per_image_mious = []
    for batch in tqdm(loader, desc=desc, leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        dsm = batch["sar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(rgb, dsm).argmax(dim=1)
        confmat += fast_confusion_matrix(pred, mask, num_classes, ignore_value)
        for image_pred, image_mask in zip(pred, mask):
            _, image_miou = miou_from_confmat(
                fast_confusion_matrix(image_pred, image_mask, num_classes, ignore_value)
            )
            per_image_mious.append(image_miou)

    iou, miou = miou_from_confmat(confmat)
    confmat_float = confmat.to(torch.float64)
    true_positive = torch.diag(confmat_float)
    ground_truth = confmat_float.sum(dim=1)
    predicted = confmat_float.sum(dim=0)
    class_accuracy = torch.zeros_like(true_positive)
    valid_accuracy = ground_truth > 0
    class_accuracy[valid_accuracy] = true_positive[valid_accuracy] / ground_truth[valid_accuracy]
    class_f1 = torch.zeros_like(true_positive)
    f1_denominator = ground_truth + predicted
    valid_f1 = f1_denominator > 0
    class_f1[valid_f1] = 2.0 * true_positive[valid_f1] / f1_denominator[valid_f1]

    seen_miou = iou[POTSDAM_SEEN_INDEXES].mean().item()
    unseen_miou = iou[POTSDAM_UNSEEN_INDEXES].mean().item()
    hmean = 2.0 * seen_miou * unseen_miou / (seen_miou + unseen_miou) if seen_miou + unseen_miou > 0 else 0.0
    return {
        "dataset": "potsdam_original_test",
        "class_order": POTSDAM_TEST_CLASSES,
        "seen_indexes": POTSDAM_SEEN_INDEXES,
        "unseen_indexes": POTSDAM_UNSEEN_INDEXES,
        "mIoU": miou,
        "seen_mIoU": seen_miou,
        "unseen_mIoU": unseen_miou,
        "hmean": hmean,
        "IoU": iou.tolist(),
        "ACC": class_accuracy.tolist(),
        "F1": class_f1.tolist(),
        "mACC": class_accuracy[valid_accuracy].mean().item() if valid_accuracy.any() else 0.0,
        "mF1": class_f1[valid_f1].mean().item() if valid_f1.any() else 0.0,
        "pACC": true_positive.sum().item() / confmat_float.sum().item() if confmat_float.sum() > 0 else 0.0,
        "confusion_matrix": confmat.tolist(),
        "per_image_mIoU_mean": sum(per_image_mious) / max(1, len(per_image_mious)),
    }


def trainable_state_dict(model):
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {name: tensor for name, tensor in model.state_dict().items() if name in trainable_names}


def save_checkpoint(path, model, optimizer, epoch, metrics=None, include_optimizer=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": trainable_state_dict(model),
        "metrics": metrics or {},
        "trainable_only": True,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    tmp_path = path.with_name(f".{path.stem}.epoch{epoch:03d}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except (RuntimeError, OSError) as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        print(f"Warning: failed to update {path}: {exc}", flush=True)


def format_metrics(metrics):
    class_iou = ",".join(
        f"{name}={value:.4f}" for name, value in zip(POTSDAM_TEST_CLASSES, metrics["IoU"])
    )
    return (
        f"mIoU={metrics['mIoU']:.4f} "
        f"seen_mIoU={metrics['seen_mIoU']:.4f} "
        f"unseen_mIoU={metrics['unseen_mIoU']:.4f} "
        f"hmean={metrics['hmean']:.4f} "
        f"mACC={metrics['mACC']:.4f} "
        f"mF1={metrics['mF1']:.4f} "
        f"pACC={metrics['pACC']:.4f} "
        f"IoU({class_iou})"
    )


def build_dataset(cfg, train: bool, max_samples: int):
    split_key = "train_split" if train else "test_split"
    classes = POTSDAM_TRAIN_CLASSES if train else POTSDAM_TEST_CLASSES
    return PotsdamDataset(
        cfg["data_root"],
        split=cfg[split_key],
        classes=classes,
        image_size=int(cfg.get("image_size", 384)),
        tile_stride=int(cfg.get("tile_stride", 384)),
        ignore_value=int(cfg["ignore_value"]),
        cache_scenes=int(cfg.get("cache_scenes", 32)),
        max_samples=max_samples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/RSOVSS_Potsdam_v1.yaml")
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
        f"Potsdam protocol: train={cfg['train_split']} scenes={len(train_set.scene_samples)} "
        f"patches={len(train_set)} seen={POTSDAM_TRAIN_CLASSES} -> "
        f"test={cfg['test_split']} scenes={len(test_set.scene_samples)} patches={len(test_set)} "
        f"eval={POTSDAM_TEST_CLASSES} unseen="
        f"{[POTSDAM_TEST_CLASSES[i] for i in POTSDAM_UNSEEN_INDEXES]}"
    )

    model = MMOVMASam(
        train_class_json=cfg["train_class_json"],
        test_class_json=cfg["test_class_json"],
        masam_weights=cfg["masam_weights"],
        num_classes=int(cfg["num_classes"]),
        ignore_value=int(cfg["ignore_value"]),
        freeze_masam=bool(cfg.get("freeze_masam", True)),
        masam_weights_type=cfg.get("masam_weights_type", "sam2_trunk"),
        predictor_version=cfg.get("predictor_version", "v1"),
        prompt_ensemble_type=cfg.get("prompt_ensemble_type", "single"),
        v2_margin_threshold=float(cfg.get("v2_margin_threshold", 0.04)),
        v2_margin_sharpness=float(cfg.get("v2_margin_sharpness", 40.0)),
        v2_affinity_kernel=int(cfg.get("v2_affinity_kernel", 3)),
        v2_affinity_temperature=float(cfg.get("v2_affinity_temperature", 0.07)),
        v2_affinity_channels=int(cfg.get("v2_affinity_channels", 32)),
        v2_decoder_kernel=int(cfg.get("v2_decoder_kernel", 3)),
        v2_decoder_max_strength=float(cfg.get("v2_decoder_max_strength", 0.2)),
        v2_decoder_max_delta=float(cfg.get("v2_decoder_max_delta", 0.5)),
        v2_guided_kernel=int(cfg.get("v2_guided_kernel", 3)),
        v2_guided_temperature=float(cfg.get("v2_guided_temperature", 0.07)),
        v2_guided_spatial_sigma=float(cfg.get("v2_guided_spatial_sigma", 1.0)),
        v2_guided_max_strength=float(cfg.get("v2_guided_max_strength", 1.0)),
        v2_guided_initial_strength=float(cfg.get("v2_guided_initial_strength", 0.01)),
        v2_guided_detach_margin=bool(cfg.get("v2_guided_detach_margin", True)),
    ).to(device)
    optimizer = build_optimizer(model, cfg)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, int(cfg["num_classes"]), int(cfg["ignore_value"]))
        print(format_metrics(metrics))
        metrics_path = output_dir / "evaluation_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        print(f"Saved evaluation metrics to {metrics_path}")
        return

    epochs = int(cfg.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("Please set a positive 'epochs' value in the YAML or pass --epochs.")
    best_hmean = -1.0
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=True)
        for batch in progress:
            rgb = batch["rgb"].to(device, non_blocking=True)
            dsm = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            losses = model(rgb, dsm, mask)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{epoch_loss / max(1, progress.n):.5f}")

        metrics = evaluate(
            model,
            test_loader,
            device,
            int(cfg["num_classes"]),
            int(cfg["ignore_value"]),
            desc=f"full eval {epoch}/{epochs}",
        )
        train_loss = epoch_loss / max(1, len(train_loader))
        print(
            f"full eval epoch={epoch}/{epochs} samples={len(test_set)} "
            f"train_loss={train_loss:.5f} {format_metrics(metrics)}",
            flush=True,
        )
        is_best = metrics["hmean"] > best_hmean
        if is_best:
            best_hmean = metrics["hmean"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, include_optimizer=True)
        with (output_dir / "metrics_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "is_best_hmean": is_best, **metrics}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
