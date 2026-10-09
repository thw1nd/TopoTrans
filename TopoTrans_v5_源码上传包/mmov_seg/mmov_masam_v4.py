"""Potsdam v4: v3 architecture with explicit unknown-negative supervision.

This module intentionally leaves :mod:`mmov_seg.mmov_masam` unchanged.  The
only behavioral difference is the training loss: pixels belonging to the raw
Potsdam unseen classes are trained as negative for every seen-class logit.
Official clutter pixels remain ignored.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .mmov_masam import MMOVMASam


class MMOVMASamV4(MMOVMASam):
    """Isolated v4 loss wrapper around the established v3 model.

    ``unknown_negative_weight=1`` matches the all-negative treatment used by
    CAT-Seg/SED for ignored pixels, except that this implementation applies it
    only to known raw unseen pixels and continues to ignore Potsdam clutter.
    """

    def __init__(self, *args, unknown_negative_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.unknown_negative_weight = float(unknown_negative_weight)
        if self.unknown_negative_weight < 0:
            raise ValueError("unknown_negative_weight must be non-negative.")
        self.last_loss_stats = {}

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        unknown_mask: Optional[torch.Tensor] = None,
    ):
        if target is None:
            return super().forward(rgb, sar)
        if unknown_mask is None:
            raise ValueError("MMOVMASamV4 training requires an explicit unknown_mask.")

        logits = self.forward_logits(rgb, sar)
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits_nhwc = logits.permute(0, 2, 3, 1)

        seen_mask = target != self.ignore_value
        unknown_mask = unknown_mask.to(device=target.device, dtype=torch.bool)
        if unknown_mask.shape != target.shape:
            raise ValueError(
                f"unknown_mask shape {tuple(unknown_mask.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        if (seen_mask & unknown_mask).any():
            raise ValueError("seen target pixels and unknown_mask must not overlap.")

        seen_targets = torch.zeros_like(logits_nhwc)
        if seen_mask.any():
            seen_targets[seen_mask] = F.one_hot(
                target[seen_mask],
                num_classes=logits.shape[1],
            ).to(logits.dtype)

        # Average over classes first. With weight=1, the final joint pixel
        # average is equivalent to BCE over all seen+unknown pixel-class terms.
        seen_loss_map = F.binary_cross_entropy_with_logits(
            logits_nhwc,
            seen_targets,
            reduction="none",
        ).mean(dim=-1)
        unknown_loss_map = F.softplus(logits_nhwc).mean(dim=-1)

        seen_count = seen_mask.sum().to(logits.dtype)
        unknown_count = unknown_mask.sum().to(logits.dtype)
        seen_sum = seen_loss_map[seen_mask].sum()
        unknown_sum = unknown_loss_map[unknown_mask].sum()
        weighted_count = seen_count + self.unknown_negative_weight * unknown_count

        if weighted_count.item() > 0:
            loss_sem_seg = (
                seen_sum + self.unknown_negative_weight * unknown_sum
            ) / weighted_count
        else:
            loss_sem_seg = logits_nhwc.sum() * 0.0

        zero = logits_nhwc.sum() * 0.0
        seen_loss = seen_sum / seen_count.clamp_min(1.0) if seen_mask.any() else zero
        unknown_loss = (
            unknown_sum / unknown_count.clamp_min(1.0)
            if unknown_mask.any()
            else zero
        )
        total_pixels = torch.tensor(target.numel(), device=target.device, dtype=logits.dtype)
        self.last_loss_stats = {
            "seen_loss": seen_loss.detach(),
            "unknown_negative_loss": unknown_loss.detach(),
            "seen_pixel_fraction": (seen_count / total_pixels).detach(),
            "unknown_pixel_fraction": (unknown_count / total_pixels).detach(),
        }
        return {"loss_sem_seg": loss_sem_seg}

