import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

from mmov_seg.ddhrnet_dataset import fast_confusion_matrix, miou_from_confmat
from mmov_seg.mmov_masam import MMOVMASam
from mmov_seg.whuoptsar_dataset import (
    WHUOPTSAR_SEEN_INDEXES,
    WHUOPTSAR_TEST_CLASSES,
    WHUOPTSAR_TRAIN_CLASSES,
    WHUOPTSAR_UNSEEN_INDEXES,
    WHUOPTSARDataset,
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
    for batch in tqdm(loader, desc=desc, leave=False):
        rgb = batch["rgb"].to(device)
        sar = batch["sar"].to(device)
        mask = batch["mask"].to(device)
        logits = model(rgb, sar)
        pred = logits.argmax(dim=1)
        confmat += fast_confusion_matrix(pred, mask, num_classes, ignore_value)
    iou, miou = miou_from_confmat(confmat)
    confmat_float = confmat.to(torch.float64)
    true_positive = torch.diag(confmat_float)
    ground_truth_count = confmat_float.sum(dim=1)
    predicted_count = confmat_float.sum(dim=0)
    acc_valid = ground_truth_count > 0
    f1_valid = (ground_truth_count + predicted_count) > 0
    class_acc = torch.zeros_like(true_positive)
    class_f1 = torch.zeros_like(true_positive)
    class_acc[acc_valid] = true_positive[acc_valid] / ground_truth_count[acc_valid]
    class_f1[f1_valid] = (
        2.0
        * true_positive[f1_valid]
        / (ground_truth_count[f1_valid] + predicted_count[f1_valid])
    )
    macc = class_acc[acc_valid].mean().item() if acc_valid.any() else 0.0
    mf1 = class_f1[f1_valid].mean().item() if f1_valid.any() else 0.0
    seen_miou = iou[WHUOPTSAR_SEEN_INDEXES].mean().item()
    unseen_miou = iou[WHUOPTSAR_UNSEEN_INDEXES].mean().item()
    hmean = 0.0
    if seen_miou + unseen_miou > 0:
        hmean = 2.0 * seen_miou * unseen_miou / (seen_miou + unseen_miou)
    return {
        "mIoU": miou,
        "mACC": macc,
        "mF1": mf1,
        "seen_mIoU": seen_miou,
        "unseen_mIoU": unseen_miou,
        "hmean": hmean,
        "IoU": iou.tolist(),
        "ACC": class_acc.tolist(),
        "F1": class_f1.tolist(),
    }


def save_checkpoint(path, model, optimizer, epoch, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics or {},
    }
    tmp_path = path.with_name(f".{path.stem}.epoch{epoch:03d}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except RuntimeError as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        fallback = path.with_name(f"{path.stem}_epoch_{epoch:03d}{path.suffix}")
        print(
            f"Warning: failed to update {path} ({exc}). "
            f"Saving this checkpoint as {fallback} instead.",
            flush=True,
        )
        torch.save(payload, fallback)
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        fallback = path.with_name(f"{path.stem}_epoch_{epoch:03d}{path.suffix}")
        print(
            f"Warning: failed to update {path} ({exc}). "
            f"Saving this checkpoint as {fallback} instead.",
            flush=True,
        )
        torch.save(payload, fallback)


def format_metrics(metrics):
    class_iou = ",".join(
        f"{name}={iou:.4f}" for name, iou in zip(WHUOPTSAR_TEST_CLASSES, metrics["IoU"])
    )
    return (
        f"mIoU={metrics['mIoU']:.4f} "
        f"mACC={metrics['mACC']:.4f} "
        f"mF1={metrics['mF1']:.4f} "
        f"seen_mIoU={metrics['seen_mIoU']:.4f} "
        f"unseen_mIoU={metrics['unseen_mIoU']:.4f} "
        f"hmean={metrics['hmean']:.4f} "
        f"IoU({class_iou})"
    )


def hmean_from_metrics(metrics):
    if not metrics:
        return -1.0
    if "hmean" in metrics:
        return float(metrics["hmean"])
    seen_miou = float(metrics.get("seen_mIoU", 0.0))
    unseen_miou = float(metrics.get("unseen_mIoU", 0.0))
    if seen_miou + unseen_miou <= 0:
        return 0.0
    return 2.0 * seen_miou * unseen_miou / (seen_miou + unseen_miou)


def append_metrics(path, epoch, train_loss, metrics, is_best):
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "is_best_hmean": is_best,
        **metrics,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(cfg, train: bool, max_samples: int):
    image_size = int(cfg.get("image_size", 384))
    return WHUOPTSARDataset(
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


def build_class_aware_sampler(dataset, cfg):
    if not bool(cfg.get("class_aware_sampler", False)):
        return None
    focus_weights = {
        2: float(cfg.get("sampler_village_weight", 4.0)),
        5: float(cfg.get("sampler_road_weight", 12.0)),
        6: float(cfg.get("sampler_others_weight", 8.0)),
    }
    min_ratio = float(cfg.get("sampler_min_ratio", 0.01))
    weights = []
    zero_weight_tiles = 0
    for sample in dataset.samples:
        ratios = sample["class_ratios"]
        if float(ratios.sum()) <= 0:
            weights.append(0.0)
            zero_weight_tiles += 1
            continue
        weight = 1.0
        for class_id, scale in focus_weights.items():
            ratio = float(ratios[class_id])
            if ratio >= min_ratio:
                weight += scale * ratio
        weights.append(weight)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("sampler_seed", 0)))
    if zero_weight_tiles:
        print(f"WHUOPTSAR sampler: excluded all-ignore tiles={zero_weight_tiles}")
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/RSOVSS_WHUOPTSAR.yaml")
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

    train_set = build_dataset(cfg, train=True, max_samples=int(cfg.get("max_train_samples", 0)))
    test_set = build_dataset(cfg, train=False, max_samples=int(cfg.get("max_test_samples", 0)))

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
        f"WHUOPTSAR protocol: train={len(train_set)} tiles, seen={WHUOPTSAR_TRAIN_CLASSES} -> "
        f"test={len(test_set)} tiles, eval={WHUOPTSAR_TEST_CLASSES}, unseen="
        f"{[WHUOPTSAR_TEST_CLASSES[i] for i in WHUOPTSAR_UNSEEN_INDEXES]}"
    )
    if train_sampler is not None:
        print("WHUOPTSAR sampler: class-aware focus=village/road/others")

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
        metrics = evaluate(model, test_loader, device, int(cfg["num_classes"]), int(cfg["ignore_value"]), desc="eval")
        print(format_metrics(metrics))
        return

    epochs = int(cfg.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("Please set a positive 'epochs' value in the YAML or pass --epochs.")
    best_hmean = -1.0
    best_path = output_dir / "best.pt"
    if best_path.exists():
        try:
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_hmean = hmean_from_metrics(best_checkpoint.get("metrics", {}))
            print(f"Loaded existing best hmean={best_hmean:.4f} from {best_path}", flush=True)
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            print(f"Warning: failed to read existing best checkpoint ({exc}).", flush=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=True)
        for batch in progress:
            rgb = batch["rgb"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            losses = model(rgb, sar, mask)
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
            f"train_loss={train_loss:.5f} "
            f"{format_metrics(metrics)}",
            flush=True,
        )
        is_best = metrics["hmean"] > best_hmean
        if is_best:
            best_hmean = metrics["hmean"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics)
        append_metrics(output_dir / "metrics_history.jsonl", epoch, train_loss, metrics, is_best)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
