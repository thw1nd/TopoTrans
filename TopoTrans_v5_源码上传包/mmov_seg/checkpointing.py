"""Verified checkpoint helpers shared by isolated training entry points."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


CHECKPOINT_FORMAT = 2


def cpu_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return an independent CPU snapshot of every registered model tensor."""

    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def model_state_digest(state: Dict[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes, and values of a model state dictionary."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_verified_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
    *,
    variant: str,
    config_path: str,
) -> Dict[str, Any]:
    """Save a portable checkpoint with a verifiable model snapshot."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = cpu_model_state(model)
    payload: Dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "model": state,
        "model_state_sha256": model_state_digest(state),
        "model_tensor_count": len(state),
        "variant": str(variant),
        "config_path": str(config_path),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    tmp_path = path.with_name(
        f".{path.stem}.epoch{int(epoch):03d}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return payload


def load_verified_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_optimizer: bool = False,
) -> Dict[str, Any]:
    """Load and validate a verified checkpoint.

    New checkpoints are loaded strictly. Legacy checkpoints retain backward
    compatibility but report their missing and unexpected keys explicitly.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain a 'model' state.")

    saved_state = checkpoint["model"]
    current_state = model.state_dict()
    saved_keys = set(saved_state)
    current_keys = set(current_state)
    missing = sorted(current_keys - saved_keys)
    unexpected = sorted(saved_keys - current_keys)
    shape_mismatch = sorted(
        name
        for name in saved_keys & current_keys
        if tuple(saved_state[name].shape) != tuple(current_state[name].shape)
    )
    if checkpoint.get("checkpoint_format", 0) >= CHECKPOINT_FORMAT:
        if missing or unexpected or shape_mismatch:
            raise RuntimeError(
                "Verified checkpoint structure mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"shape_mismatch={shape_mismatch[:5]}"
            )
        expected_digest = checkpoint.get("model_state_sha256")
        actual_digest = model_state_digest(saved_state)
        if expected_digest and expected_digest != actual_digest:
            raise RuntimeError(
                f"Checkpoint model digest mismatch for {path}: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        model.load_state_dict(saved_state, strict=True)
        loaded_digest = model_state_digest(model.state_dict())
        if expected_digest and expected_digest != loaded_digest:
            raise RuntimeError(
                f"Loaded model digest mismatch for {path}: "
                f"expected={expected_digest}, actual={loaded_digest}"
            )
    else:
        incompatible = model.load_state_dict(saved_state, strict=False)
        missing = sorted(set(incompatible.missing_keys))
        unexpected = sorted(set(incompatible.unexpected_keys))

    if restore_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    checkpoint["load_report"] = {
        "strict": checkpoint.get("checkpoint_format", 0) >= CHECKPOINT_FORMAT,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "model_tensor_count": len(saved_state),
        "model_state_sha256": model_state_digest(saved_state),
        "loaded_model_state_sha256": model_state_digest(model.state_dict()),
    }
    return checkpoint
