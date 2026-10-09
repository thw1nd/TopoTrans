# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from .utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from .sam2_utils import DropPath, MLP

# lora layer
from loralib.layers import Linear


# -------------------------
# SSCL-aligned local dilation per stage (match ViTDet/SSCL)
# stage: 1  2  3  4
LOCAL_DILATION_PER_STAGE = {1: 1, 2: 1, 3: 2, 4: 3}
# -------------------------


def _ensure_stage_tuple(values: Tuple[int, int, int, int], name: str) -> Tuple[int, int, int, int]:
    if not isinstance(values, tuple) and not isinstance(values, list):
        raise TypeError(f"{name} must be a length-4 tuple/list, got {type(values).__name__}")
    if len(values) != 4:
        raise ValueError(f"{name} must be length 4 for stages 1..4, got {values}")
    return tuple(int(v) for v in values)


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class LocalBiasBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = SeparableConv(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
        )
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C]
        """
        y = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        y = self.conv(y)
        scale = torch.tanh(self.gamma)
        y = scale * y
        y = y.permute(0, 2, 3, 1)  # [B, H, W, C]
        return x + y


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


# -------------------------
# SSCLFuseModule v2:
#   - Low-frequency (structure) guidance -> concat -> local conv -> shared LoRA ΔK/ΔV
#   - High-frequency conflict -> prior-guided sparse semantic prototypes -> Kmix tokens (NO Vmix)
#   - Only used in global-attn blocks (window_size == 0)
# -------------------------


class LearnableLowPass2D(nn.Module):
    """
    Depthwise low-pass filter with positive & normalized kernels per channel.
    Input/Output: BHWC
    """
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.kernel_size = kernel_size
        self.w = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))

    def forward(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        x = x_bhwc.permute(0, 3, 1, 2)  # B,C,H,W
        w = F.softplus(self.w)
        w = w / (w.sum(dim=(2, 3), keepdim=True) + 1e-6)
        y = F.conv2d(x, w, padding=self.kernel_size // 2, groups=x.shape[1])
        return y.permute(0, 2, 3, 1)



def _compute_proto_topk(
    H: int,
    W: int,
    ratio: float,
    min_tokens: int = 4,
    max_tokens: int = 64,
) -> int:
    """Convert a dense token ratio into a compact prototype budget.

    The returned K is intentionally much smaller than H*W. In the revised
    selector, K controls the number of differentiable sparse prototype queries,
    not the number of hard Top-K indices.
    """
    L = max(1, H * W)
    dense_budget = max(min_tokens, int(round(L * float(ratio))))
    k = int(round(math.sqrt(float(dense_budget))))
    k = max(min_tokens, min(k, max_tokens, L))
    return k


def _spatial_zscore(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Per-sample spatial z-score for BHWC maps with a singleton channel."""
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
    var = flat.var(dim=1, keepdim=True, unbiased=False).view(B, 1, 1, 1)
    return (x - mean) / torch.sqrt(var + eps)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation.

    This is a differentiable sparse alternative to softmax. It maps logits to a
    probability simplex while allowing exact zeros, which is suitable for sparse
    prototype routing without hard Top-K indices.
    """
    if dim < 0:
        dim = logits.dim() + dim

    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted = torch.sort(z, dim=dim, descending=True).values
    z_cumsum = z_sorted.cumsum(dim)

    r = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = -1
    r = r.view(view_shape)

    support = 1 + r * z_sorted > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(dim, k.long() - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


class MultiFreqPrototypeSelector(nn.Module):
    """Prior-guided differentiable sparse multi-frequency prototype selector.

    This selector keeps the optimized score/routing/attention path we discussed:
      1) interpretable prior score dominated by normalized high-frequency conflict,
         with high-frequency energy and low-frequency semantic reliability as auxiliary priors;
      2) zero-gated learnable residual score;
      3) sparsemax routing instead of hard Top-K indices;
      4) K-only prototype projection for query-aware attention bias.

    The important change here is the prototype *content*: it is no longer built
    only from high-frequency evidence plus fused low-frequency context. It now
    explicitly includes both high-frequency and low-frequency modality evidence:
      - |x_hp|, |y_hp|, refined high-frequency conflict;
      - |x_lp|, |y_lp|, refined low-frequency conflict;
      - fused low-frequency shared semantic context.

    Input tensors are BHWC. Output K-only tokens are [B, K, heads, head_dim].
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        shared_rank: int = 16,
        shared_alpha: int = 16,
        max_proto_tokens: int = 64,
        route_temperature: float = 0.7,
        prior_conf_weight: float = 1.0,
        prior_energy_weight: float = 0.35,
        prior_sem_weight: float = 0.35,
        residual_hidden_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_proto_tokens = int(max_proto_tokens)
        self.route_temperature = float(route_temperature)
        self.prior_conf_weight = float(prior_conf_weight)
        self.prior_energy_weight = float(prior_energy_weight)
        self.prior_sem_weight = float(prior_sem_weight)

        # Multi-frequency prototype content. Spatial size is preserved.
        # 7C = |x_hp|, |y_hp|, hf_conf, |x_lp|, |y_lp|, lf_conf, fused_lp.
        self.fuse = SeparableConv(7 * dim, dim, kernel_size=3, stride=1, dilation=1)
        self.content_norm = nn.LayerNorm(dim)

        # Low-frequency semantic reliability. This is part of the interpretable
        # prior, not an unconstrained replacement for the prior.
        self.sem_norm = nn.LayerNorm(dim)
        self.sem_proj = nn.Linear(dim, 1)

        hidden = max(16, int(round(dim * residual_hidden_ratio)))
        self.score_residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start from pure prior-guided routing; learn residual correction later.
        self.score_res_gamma = nn.Parameter(torch.zeros(1))

        # K learnable queries for differentiable sparse routing. The first k are
        # used according to the stage-specific prototype budget.
        self.proto_queries = nn.Parameter(torch.randn(self.max_proto_tokens, dim) * 0.02)

        # K-only prototype projection for the attention-bias branch.
        self.proj_k = Linear(dim, dim_out, r=shared_rank, lora_alpha=shared_alpha)

    def _compute_prior_score(
        self,
        abs_x_hp: torch.Tensor,
        abs_y_hp: torch.Tensor,
        hf_conf_raw: torch.Tensor,
        fused_lp: torch.Tensor,
    ) -> torch.Tensor:
        """Build an interpretable BHWC score map before sparse routing.

        We keep the score design from the optimized version: high-frequency
        conflict remains the dominant selection prior because it is more tightly
        related to boundary/detail errors, while low-frequency information
        enters as semantic reliability and prototype content.
        """
        eps = 1e-6

        # Per-position high-frequency response intensity.
        ex = abs_x_hp.mean(dim=-1, keepdim=True)
        ey = abs_y_hp.mean(dim=-1, keepdim=True)
        total_energy = ex + ey

        # Normalized cross-modal high-frequency conflict. This reduces sensitivity
        # to absolute response-scale differences across modalities.
        ec = hf_conf_raw.mean(dim=-1, keepdim=True)
        rel_conflict = ec / (total_energy + eps)

        # Low-frequency semantic reliability. log() turns it into an additive
        # log-prior and penalizes unreliable locations smoothly.
        sem = torch.sigmoid(self.sem_proj(self.sem_norm(fused_lp))).clamp(min=eps, max=1.0)
        sem_log = torch.log(sem)

        s_conf = _spatial_zscore(rel_conflict)
        s_energy = _spatial_zscore(torch.log1p(total_energy))
        s_sem = _spatial_zscore(sem_log)

        return (
            self.prior_conf_weight * s_conf
            + self.prior_energy_weight * s_energy
            + self.prior_sem_weight * s_sem
        )

    def forward(
        self,
        abs_x_hp: torch.Tensor,
        abs_y_hp: torch.Tensor,
        hf_conf_raw: torch.Tensor,
        hf_conf_feat: torch.Tensor,
        abs_x_lp: torch.Tensor,
        abs_y_lp: torch.Tensor,
        lf_conf_feat: torch.Tensor,
        fused_lp: torch.Tensor,
        ratio: float,
        min_tokens: int,
    ) -> torch.Tensor:
        B, H, W, C = fused_lp.shape
        k = _compute_proto_topk(H, W, ratio, min_tokens=min_tokens, max_tokens=self.max_proto_tokens)

        # Multi-frequency prototype content feature. No spatial downsampling.
        feat = torch.cat(
            [
                abs_x_hp,
                abs_y_hp,
                hf_conf_feat,
                abs_x_lp,
                abs_y_lp,
                lf_conf_feat,
                fused_lp,
            ],
            dim=-1,
        )
        feat = self.fuse(feat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B,H,W,C]

        # Prior-guided score plus a zero-gated learnable residual.
        # The score is still dominated by normalized high-frequency conflict,
        # but the selected prototype content contains both high- and low-frequency evidence.
        score_prior = self._compute_prior_score(abs_x_hp, abs_y_hp, hf_conf_raw, fused_lp)  # [B,H,W,1]
        score_res = self.score_residual(feat)  # [B,H,W,1]
        score = score_prior + torch.tanh(self.score_res_gamma) * score_res

        tokens = feat.reshape(B, H * W, C)
        route_tokens = F.normalize(self.content_norm(feat).reshape(B, H * W, C), p=2, dim=-1)
        queries = F.normalize(self.proto_queries[:k], p=2, dim=-1).unsqueeze(0).expand(B, -1, -1)

        # Query-dependent routing logits with an additive interpretable prior.
        # Shape: [B,K,H*W]. The score term is shared by all prototype queries,
        # while learnable queries make individual prototypes content-adaptive.
        logits = torch.matmul(queries, route_tokens.transpose(-2, -1))
        logits = logits + score.reshape(B, 1, H * W)
        logits = logits / max(self.route_temperature, 1e-4)

        route = _sparsemax(logits, dim=-1)  # [B,K,H*W], sparse and differentiable
        proto_tok = torch.matmul(route, tokens)  # [B,K,C]

        k_mix = self.proj_k(proto_tok).reshape(B, k, self.num_heads, -1)
        return k_mix


class SSCLFuseModuleV2(nn.Module):
    """
    Returns:
        delta_kv_x: [B, L, 2, heads, head_dim]
        delta_kv_y: [B, L, 2, heads, head_dim]
            Shared low-frequency guidance for K/V, returned for both branches.

        k_mix: [B, K, heads, head_dim]
            Multi-frequency semantic conflict prototypes as K-only tokens.

    Design note:
        The modality-specific LoRA branch is the original qkv LoRA in
        MultiScaleAttention and is applied directly to the raw normalized
        modality features x/y. We do not feed low-pass-filtered features into a
        separate branch-specific ΔK/ΔV path here, because the role of the
        specific branch is to adapt each modality's Q/K/V in the remote-sensing
        scene, not to become another low-frequency residual path.

        Low-pass filtering is used only for:
          1) shared low-frequency K/V guidance;
          2) semantic reliability;
          3) low-frequency content in the prototype selector.
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        shared_rank: int = 16,
        shared_alpha: int = 16,
        branch_rank: int = 16,
        branch_alpha: int = 16,
        lp_kernel: int = 5,
        fuse_kernel: int = 3,
        conflict_kernel: int = 3,
        mix_token_ratio: float = 0.1,
        min_mix_tokens: int = 4,
        detach_conflict: bool = True,
        conf_semantic_gate: bool = True,
        conf_semantic_gate_smooth_ks: int = 3,
        conf_semantic_gate_scale_init: float = 0.0,
        prototype_max_tokens: int = 64,
        prototype_nms_kernel: int = 3,
        prototype_route_temperature: float = 0.7,
        prototype_prior_conf_weight: float = 1.0,
        prototype_prior_energy_weight: float = 0.35,
        prototype_prior_sem_weight: float = 0.35,
        prototype_residual_hidden_ratio: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.mix_token_ratio = float(mix_token_ratio)
        self.min_mix_tokens = int(min_mix_tokens)
        self.detach_conflict = bool(detach_conflict)

        # Kept as constructor arguments for external config compatibility.
        # They are intentionally not used: no branch-specific low-pass ΔK/ΔV path.
        self.branch_rank = int(branch_rank)
        self.branch_alpha = int(branch_alpha)

        # semantic confidence prior for conflict: learn a smooth gate from low-frequency fused semantics
        self.use_conf_semantic_gate = bool(conf_semantic_gate)
        self.conf_semantic_gate_smooth_ks = int(conf_semantic_gate_smooth_ks)
        self.sem_norm = nn.LayerNorm(dim)
        self.sem_proj = nn.Linear(dim, 1)
        self.sem_scale = nn.Parameter(torch.tensor(float(conf_semantic_gate_scale_init)))

        # learnable low-pass and low-frequency fusion
        self.lowpass = LearnableLowPass2D(dim, kernel_size=lp_kernel)
        self.fuse_conv = SeparableConv(2 * dim, dim, kernel_size=fuse_kernel, stride=1, dilation=1)

        # Shared LoRA for ΔK/ΔV: preserves cross-modal low-frequency alignment.
        self.shared_kv = Linear(dim, 2 * dim_out, r=shared_rank, lora_alpha=shared_alpha)

        # Separate refinement for high- and low-frequency discrepancy.
        self.hf_conf_refine = SeparableConv(dim, dim, kernel_size=conflict_kernel, stride=1, dilation=1)
        self.lf_conf_refine = SeparableConv(dim, dim, kernel_size=conflict_kernel, stride=1, dilation=1)

        # Differentiable multi-frequency semantic prototype selector.
        # prototype_nms_kernel is intentionally unused in the revised selector:
        # hard NMS/Top-K is replaced by sparsemax-based differentiable routing.
        self.proto_selector = MultiFreqPrototypeSelector(
            dim=dim,
            dim_out=dim_out,
            num_heads=num_heads,
            shared_rank=shared_rank,
            shared_alpha=shared_alpha,
            max_proto_tokens=prototype_max_tokens,
            route_temperature=prototype_route_temperature,
            prior_conf_weight=prototype_prior_conf_weight,
            prior_energy_weight=prototype_prior_energy_weight,
            prior_sem_weight=prototype_prior_sem_weight,
            residual_hidden_ratio=prototype_residual_hidden_ratio,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x, y: [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert y.shape[:3] == (B, H, W) and y.shape[3] == C, "SSCLFuseModuleV2 expects same spatial shape"

        # ----- low-frequency structure -----
        x_lp = self.lowpass(x)
        y_lp = self.lowpass(y)

        xy = torch.cat([x_lp, y_lp], dim=-1)  # BHWC, 2C
        fused_lp = self.fuse_conv(xy.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # BHWC, C

        sem_weight = None
        if self.use_conf_semantic_gate:
            sem = torch.sigmoid(self.sem_proj(self.sem_norm(fused_lp)))  # [B,H,W,1]
            ks = self.conf_semantic_gate_smooth_ks
            if ks is not None and ks > 1:
                sem_bchw = sem.permute(0, 3, 1, 2)
                sem_bchw = F.avg_pool2d(sem_bchw, kernel_size=ks, stride=1, padding=ks // 2)
                sem = sem_bchw.permute(0, 2, 3, 1)
            scale = torch.tanh(self.sem_scale)
            sem_weight = 1.0 + scale * (sem - 0.5) * 2.0

        # ----- shared low-frequency ΔK/ΔV only -----
        # The modality-specific LoRA qkv path has already been computed from raw x/y
        # in MultiScaleAttention. Here we only add a common low-frequency semantic prior.
        delta_kv_shared = self.shared_kv(fused_lp)
        delta_kv_shared = delta_kv_shared.reshape(B, H * W, 2, self.num_heads, -1)

        # Returned twice to keep the downstream attention code explicit and symmetric.
        delta_kv_x = delta_kv_shared
        delta_kv_y = delta_kv_shared

        # ----- multi-frequency semantic prototypes (Kmix) -----
        k_mix = None
        if self.mix_token_ratio > 0.0 and self.min_mix_tokens > 0:
            x_hp = x - x_lp
            y_hp = y - y_lp

            if sem_weight is not None:
                # Apply semantic reliability only to high-frequency evidence. Low-frequency
                # evidence is kept as structure/semantic content rather than gated away.
                x_hp = x_hp * sem_weight
                y_hp = y_hp * sem_weight

            proto_x_hp = x_hp.detach() if self.detach_conflict else x_hp
            proto_y_hp = y_hp.detach() if self.detach_conflict else y_hp

            # Low-frequency content is included in the prototype content. Detaching it with
            # the conflict branch keeps the selector from over-optimizing the low-pass filter
            # through prototype routing; set detach_conflict=False if you want full gradients.
            proto_x_lp = x_lp.detach() if self.detach_conflict else x_lp
            proto_y_lp = y_lp.detach() if self.detach_conflict else y_lp

            abs_x_hp = torch.abs(proto_x_hp)
            abs_y_hp = torch.abs(proto_y_hp)
            hf_conf_raw = torch.abs(proto_x_hp - proto_y_hp)
            hf_conf_feat = self.hf_conf_refine(hf_conf_raw.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            abs_x_lp = torch.abs(proto_x_lp)
            abs_y_lp = torch.abs(proto_y_lp)
            lf_conf_raw = torch.abs(proto_x_lp - proto_y_lp)
            lf_conf_feat = self.lf_conf_refine(lf_conf_raw.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            k_mix = self.proto_selector(
                abs_x_hp=abs_x_hp,
                abs_y_hp=abs_y_hp,
                hf_conf_raw=hf_conf_raw,
                hf_conf_feat=hf_conf_feat,
                abs_x_lp=abs_x_lp,
                abs_y_lp=abs_y_lp,
                lf_conf_feat=lf_conf_feat,
                fused_lp=fused_lp,
                ratio=self.mix_token_ratio,
                min_tokens=self.min_mix_tokens,
            )

        return delta_kv_x, delta_kv_y, k_mix


class MultiScaleAttention(nn.Module):
    @staticmethod
    def _make_conf_head_mask(num_heads: int, frac: float, mode: str = "interleave", seed: int = 0) -> torch.Tensor:
        """Create a [num_heads] float mask with 1s on selected heads.

        Notes:
          - Uses ceil to avoid Python round(0.5)->0 behavior.
          - mode:
              * "first":       first N heads (legacy behavior)
              * "interleave":  evenly spaced heads (deterministic, recommended)
              * "fixed_random": fixed random permutation with a given seed (deterministic)
        """
        frac = float(frac)
        if frac <= 0.0 or num_heads <= 0:
            return torch.zeros(num_heads)

        mix_heads = int(math.ceil(frac * num_heads))
        mix_heads = max(1, min(num_heads, mix_heads))

        if mode == "first":
            idx = torch.arange(mix_heads)
        elif mode == "interleave":
            idx = torch.linspace(0, num_heads - 1, steps=mix_heads).round().long()
            idx = torch.unique(idx)
            if idx.numel() < mix_heads:
                remaining = [i for i in range(num_heads) if i not in idx.tolist()]
                pad = torch.tensor(remaining[: mix_heads - idx.numel()], dtype=idx.dtype)
                idx = torch.cat([idx, pad], dim=0)
        elif mode == "fixed_random":
            g = torch.Generator()
            g.manual_seed(int(seed))
            idx = torch.randperm(num_heads, generator=g)[:mix_heads]
        else:
            raise ValueError(f"Unknown conf_head_select_mode: {mode}")

        mask = torch.zeros(num_heads)
        mask[idx] = 1.0
        return mask

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        specific_attn_rank: int = 16,
        specific_attn_alpha: int = 16,
        # SSCL v2 controls (only active in global-attn blocks)
        enable_sscl: bool = False,
        shared_rank: int = 16,
        shared_alpha: int = 16,
        branch_rank: int = 16,
        branch_alpha: int = 16,
        sscl_lp_kernel: int = 5,
        mix_token_ratio: float = 0.1,
        mix_heads_frac: float = 0.25,
        detach_conflict: bool = True,
        conf_semantic_gate: bool = True,
        conf_semantic_gate_smooth_ks: int = 3,
        conf_semantic_gate_scale_init: float = 0.0,
        conf_head_select_mode: str = "interleave",
        conf_head_select_seed: int = 0,
        attn_dropout_p: float = 0.05,
        # kept for config compatibility; query-aware prototype bias no longer uses hard top-k aggregation
        conf_topk: int = 4,
        prototype_route_temperature: float = 0.7,
        prototype_prior_conf_weight: float = 1.0,
        prototype_prior_energy_weight: float = 0.35,
        prototype_prior_sem_weight: float = 0.35,
        prototype_residual_hidden_ratio: float = 0.25,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool

        # attention dropout (typically only enabled in global-attn blocks)
        self.attn_dropout_p = float(attn_dropout_p)
        # kept for external configs; not used by query-aware prototype bias
        self.conf_topk = int(conf_topk)
        # ---- attention LoRA config ----
        self.specific_attn_rank = int(specific_attn_rank)
        self.specific_attn_alpha = int(specific_attn_alpha)

        # ---- specific-modality attention lora ----
        self.qkv = Linear(dim, dim_out * 3, r=self.specific_attn_rank, lora_alpha=self.specific_attn_alpha)
        self.aux1_qkv = Linear(dim, dim_out * 3, r=self.specific_attn_rank, lora_alpha=self.specific_attn_alpha)

        self.proj = nn.Linear(dim_out, dim_out)
        self.aux1_proj = nn.Linear(dim_out, dim_out)

        # ---- SSCL v2 (only global attn) ----
        self.enable_sscl = bool(enable_sscl)
        if self.enable_sscl:
            self.sscl_fuse = SSCLFuseModuleV2(
                dim=dim,
                dim_out=dim_out,
                num_heads=num_heads,
                shared_rank=shared_rank,
                shared_alpha=shared_alpha,
                branch_rank=branch_rank,
                branch_alpha=branch_alpha,
                lp_kernel=sscl_lp_kernel,
                mix_token_ratio=mix_token_ratio,
                detach_conflict=detach_conflict,
                conf_semantic_gate=conf_semantic_gate,
                conf_semantic_gate_smooth_ks=conf_semantic_gate_smooth_ks,
                conf_semantic_gate_scale_init=conf_semantic_gate_scale_init,
                prototype_route_temperature=prototype_route_temperature,
                prototype_prior_conf_weight=prototype_prior_conf_weight,
                prototype_prior_energy_weight=prototype_prior_energy_weight,
                prototype_prior_sem_weight=prototype_prior_sem_weight,
                prototype_residual_hidden_ratio=prototype_residual_hidden_ratio,
            )

            # head-wise gates: start from 0 => safe warm-up (like vitdet cross_gamma)
            self.shared_gamma = nn.Parameter(torch.zeros(num_heads))  # for ΔK/ΔV
            self.conf_gamma = nn.Parameter(torch.zeros(num_heads))    # for conflict bias

                        # which heads are allowed to use conflict bias
            self.conf_head_select_mode = conf_head_select_mode
            self.conf_head_select_seed = conf_head_select_seed
            head_mask = self._make_conf_head_mask(
                num_heads=num_heads,
                frac=mix_heads_frac,
                mode=self.conf_head_select_mode,
                seed=self.conf_head_select_seed,
            )
            self.register_buffer("conf_head_mask", head_mask, persistent=False)
        else:
            self.sscl_fuse = None

    def _apply_conf_bias(
        self,
        q_bhld: torch.Tensor,     # [B, heads, Lq, D]
        k_bhld: torch.Tensor,     # [B, heads, Lk, D]
        v_bhld: torch.Tensor,     # [B, heads, Lk, D]
        k_mix_bhmd: torch.Tensor, # [B, heads, Lmix, D]
        conf_gamma: torch.Tensor, # [heads]
        head_mask: torch.Tensor,  # [heads]
    ) -> torch.Tensor:
        """
        Query-aware prototype bias.

        Old behavior:
            bias_j = sim(k_j, p), broadcast to every query.

        Revised behavior:
            bias_ij = sum_m sim(q_i, p_m) * sim(k_j, p_m).

        Therefore, a key is enhanced only when the current query and the key are
        both related to the same conflict-aware prototype. The prototype still
        acts on the K/matching side; V remains the original semantic content.
        """
        dropout_p = self.attn_dropout_p if self.training else 0.0

        q_norm = F.normalize(q_bhld, p=2, dim=-1)
        k_norm = F.normalize(k_bhld, p=2, dim=-1)
        p_norm = F.normalize(k_mix_bhmd, p=2, dim=-1)

        # [B, heads, Lq, M] and [B, heads, Lk, M]
        q_proto = torch.matmul(q_norm, p_norm.transpose(-2, -1))
        k_proto = torch.matmul(k_norm, p_norm.transpose(-2, -1))

        # Query-key bias through shared prototype coordinates:
        # [B, heads, Lq, M] x [B, heads, M, Lk] -> [B, heads, Lq, Lk]
        m = max(1, k_mix_bhmd.shape[-2])
        raw = torch.matmul(q_proto, k_proto.transpose(-2, -1)) / math.sqrt(float(m))
        raw = torch.tanh(raw)  # bounded additive logit bias

        gamma = torch.tanh(conf_gamma).view(1, -1, 1, 1)
        mask = head_mask.view(1, -1, 1, 1)
        bias = (raw * gamma * mask).to(dtype=q_bhld.dtype)

        return F.scaled_dot_product_attention(
            q_bhld,
            k_bhld,
            v_bhld,
            attn_mask=bias,
            dropout_p=dropout_p,
        )

    def forward(self, x, y):
        """please note: x.shape must same as y.shape"""
        B, H, W, _ = x.shape
        B_aux1, H_aux1, W_aux1, _ = y.shape
        assert (B, H, W) == (B_aux1, H_aux1, W_aux1), "x/y spatial shapes must match"

        # qkv with shape (B, H * W, 3, nHead, C)
        qkv_x = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        qkv_aux1 = self.aux1_qkv(y).reshape(B, H_aux1 * W_aux1, 3, self.num_heads, -1)

        # q, k, v with shape (B, H * W, nheads, C)
        q_x, k_x, v_x = torch.unbind(qkv_x, 2)
        q_y, k_y, v_y = torch.unbind(qkv_aux1, 2)

        # -------- SSCL v2: only in global-attn blocks --------
        k_mix = None
        if self.enable_sscl and self.sscl_fuse is not None:
            delta_kv_x, delta_kv_y, k_mix = self.sscl_fuse(x, y)
            # delta_kv_x/y: B,L,2,heads,D; k_mix: B,Lmix,heads,D
            delta_k_x, delta_v_x = torch.unbind(delta_kv_x, dim=2)
            delta_k_y, delta_v_y = torch.unbind(delta_kv_y, dim=2)

            shared_scale = torch.tanh(self.shared_gamma).view(1, 1, self.num_heads, 1)
            k_x = k_x + delta_k_x * shared_scale
            v_x = v_x + delta_v_x * shared_scale
            k_y = k_y + delta_k_y * shared_scale
            v_y = v_y + delta_v_y * shared_scale

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q_x = do_pool(q_x.reshape(B, H, W, -1), self.q_pool)
            H, W = q_x.shape[1:3]  # downsampled shape
            q_x = q_x.reshape(B, H * W, self.num_heads, -1)

            q_y = do_pool(q_y.reshape(B_aux1, H_aux1, W_aux1, -1), self.q_pool)
            H_aux1, W_aux1 = q_y.shape[1:3]  # downsampled shape
            q_y = q_y.reshape(B_aux1, H_aux1 * W_aux1, self.num_heads, -1)

        # Torch SDPA expects [B, nheads, L, D]
        qx = q_x.transpose(1, 2)

        dropout_p = self.attn_dropout_p if self.training else 0.0

        # X attention (optionally with conflict bias)
        kx = k_x.transpose(1, 2)
        vx = v_x.transpose(1, 2)
        if k_mix is not None and self.enable_sscl:
            km = k_mix.transpose(1, 2)  # B,heads,Lmix,D
            x_out = self._apply_conf_bias(qx, kx, vx, km, self.conf_gamma, self.conf_head_mask)
        else:
            x_out = F.scaled_dot_product_attention(qx, kx, vx, dropout_p=dropout_p)

        # Y attention
        qy = q_y.transpose(1, 2)
        ky2 = k_y.transpose(1, 2)
        vy2 = v_y.transpose(1, 2)
        if k_mix is not None and self.enable_sscl:
            km = k_mix.transpose(1, 2)
            y_out = self._apply_conf_bias(qy, ky2, vy2, km, self.conf_gamma, self.conf_head_mask)
        else:
            y_out = F.scaled_dot_product_attention(qy, ky2, vy2, dropout_p=dropout_p)

        # Transpose back
        x_out = x_out.transpose(1, 2).reshape(B, H, W, -1)
        y_out = y_out.transpose(1, 2).reshape(B_aux1, H_aux1, W_aux1, -1)

        x_out = self.proj(x_out)
        y_out = self.aux1_proj(y_out)

        return x_out, y_out


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        specific_attn_rank: int = 16,
        specific_attn_alpha: int = 16,
        mlp_rank: int = 16,
        mlp_alpha: int = 16,
        # SSCL v2 controls
        enable_sscl: bool = False,
        shared_rank: int = 16,
        shared_alpha: int = 16,
        branch_rank: int = 16,
        branch_alpha: int = 16,
        sscl_lp_kernel: int = 5,
        mix_token_ratio: float = 0.1,
        mix_heads_frac: float = 0.25,
        detach_conflict: bool = True,
        conf_semantic_gate: bool = True,
        conf_semantic_gate_smooth_ks: int = 3,
        conf_semantic_gate_scale_init: float = 0.0,
        conf_head_select_mode: str = "interleave",
        conf_head_select_seed: int = 0,
        attn_dropout_p: float = 0.0,
        conf_topk: int = 4,
        prototype_route_temperature: float = 0.7,
        prototype_prior_conf_weight: float = 1.0,
        prototype_prior_energy_weight: float = 0.35,
        prototype_prior_sem_weight: float = 0.35,
        prototype_residual_hidden_ratio: float = 0.25,
        # local block dilation aligned with SSCL
        local_dilation: int = 1,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        # ---- modality-related norm ----
        self.x_norm1 = norm_layer(dim)
        self.aux1_norm1 = norm_layer(dim)

        # ---- lora config ----
        self.specific_attn_rank = int(specific_attn_rank)
        self.specific_attn_alpha = int(specific_attn_alpha)
        self.mlp_rank = int(mlp_rank)
        self.mlp_alpha = int(mlp_alpha)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        # ---- core attn with lora (+ optional SSCL v2 in global blocks) ----
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            specific_attn_rank=self.specific_attn_rank,
            specific_attn_alpha=self.specific_attn_alpha,
            num_heads=num_heads,
            q_pool=self.pool,
            enable_sscl=enable_sscl,
            shared_rank=shared_rank,
            shared_alpha=shared_alpha,
            branch_rank=branch_rank,
            branch_alpha=branch_alpha,
            sscl_lp_kernel=sscl_lp_kernel,
            mix_token_ratio=mix_token_ratio,
            mix_heads_frac=mix_heads_frac,
            detach_conflict=detach_conflict,
            conf_semantic_gate=conf_semantic_gate,
            conf_semantic_gate_smooth_ks=conf_semantic_gate_smooth_ks,
            conf_semantic_gate_scale_init=conf_semantic_gate_scale_init,
            conf_head_select_mode=conf_head_select_mode,
            conf_head_select_seed=conf_head_select_seed,
            attn_dropout_p=attn_dropout_p,
            conf_topk=conf_topk,
            prototype_route_temperature=prototype_route_temperature,
            prototype_prior_conf_weight=prototype_prior_conf_weight,
            prototype_prior_energy_weight=prototype_prior_energy_weight,
            prototype_prior_sem_weight=prototype_prior_sem_weight,
            prototype_residual_hidden_ratio=prototype_residual_hidden_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ---- norm ----
        self.x_norm2 = norm_layer(dim_out)
        self.aux1_norm2 = norm_layer(dim_out)

        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
            r=self.mlp_rank,
            alpha=self.mlp_alpha,
        )

        self.aux1_mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
            r=self.mlp_rank,
            alpha=self.mlp_alpha,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
            self.aux1_proj = nn.Linear(dim, dim_out)

        # ---- local block (dilation aligned with SSCL/VitDet) ----
        self.local_x = LocalBiasBlock(dim_out, kernel_size=3, dilation=local_dilation)
        self.local_aux1 = LocalBiasBlock(dim_out, kernel_size=3, dilation=local_dilation)

    def forward(self, x, y):
        shortcut = x  # B, H, W, C
        x = self.x_norm1(x)

        shortcut_y = y
        y = self.aux1_norm1(y)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
            shortcut_y = do_pool(self.aux1_proj(y), self.pool)

        # Window partition
        window_size = self.window_size
        use_window = window_size > 0
        if use_window:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

            aux1_H, aux1_W = y.shape[1], y.shape[2]
            y, aux1_pad_hw = window_partition(y, window_size)

        # Window Attention + Q Pooling (if stage change)
        x, y = self.attn(x, y)

        if self.q_stride:
            # Shapes have changed due to Q pooling
            if use_window:
                window_size = max(1, self.window_size // self.q_stride[0])
            H, W = shortcut.shape[1:3]
            aux1_H, aux1_W = shortcut_y.shape[1:3]

            if use_window:
                pad_h = (window_size - H % window_size) % window_size
                pad_w = (window_size - W % window_size) % window_size
                pad_hw = (H + pad_h, W + pad_w)

                aux1_pad_h = (window_size - aux1_H % window_size) % window_size
                aux1_pad_w = (window_size - aux1_W % window_size) % window_size
                aux1_pad_hw = (aux1_H + aux1_pad_h, aux1_W + aux1_pad_w)

        # Reverse window partition
        if use_window:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
            y = window_unpartition(y, window_size, aux1_pad_hw, (aux1_H, aux1_W))

        x = shortcut + self.drop_path(x)
        x = self.local_x(x)
        x = x + self.drop_path(self.mlp(self.x_norm2(x)))

        y = shortcut_y + self.drop_path(y)
        y = self.local_aux1(y)
        y = y + self.drop_path(self.aux1_mlp(self.aux1_norm2(y)))
        return x, y


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        x_in_chans: int = 3,  # rgb
        y_in_chans: int = 3,  # dsm or sar
        specific_attn_rank_per_stage: Tuple[int, int, int, int] = (16, 32, 64, 64),
        specific_attn_alpha_per_stage: Tuple[int, int, int, int] = (16, 32, 64, 64),
        mlp_rank_per_stage: Tuple[int, int, int, int] = (64, 64, 64, 64),
        mlp_alpha_per_stage: Tuple[int, int, int, int] = (64, 64, 64, 64),
        # SSCL v2 (shared branch + conflict Kmix bias)
        shared_rank_per_stage: Tuple[int, int, int, int] = (0, 0, 32, 32),
        shared_alpha_per_stage: Tuple[int, int, int, int] = (0, 0, 32, 32),
        branch_rank_per_stage: Tuple[int, int, int, int] = (0, 0, 32, 32),
        branch_alpha_per_stage: Tuple[int, int, int, int] = (0, 0, 32, 32),
        sscl_lp_kernel: int = 5,
        mix_token_ratio: float = 0.1,
        mix_heads_frac: float = 0.25,
        detach_conflict: bool = True,
        conf_semantic_gate: bool = True,
        conf_semantic_gate_smooth_ks: int = 3,
        conf_semantic_gate_scale_init: float = 0.0,
        conf_head_select_mode: str = "interleave",
        conf_head_select_seed: int = 0,
        # stage-aware conflict mix (stages 1..4). Set to None to use mix_token_ratio/mix_heads_frac for all stages.
        mix_token_ratio_per_stage: Tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.1),
        mix_heads_frac_per_stage: Tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.25),
        # attention dropout for global-attn blocks
        attn_dropout_p_global: float = 0.1,
        # kept for config compatibility; query-aware prototype bias no longer uses hard top-k aggregation
        conf_topk: int = 4,
        # prior-guided sparse prototype routing controls
        prototype_route_temperature: float = 0.7,
        prototype_prior_conf_weight: float = 1.0,
        prototype_prior_energy_weight: float = 0.35,
        prototype_prior_sem_weight: float = 0.35,
        prototype_residual_hidden_ratio: float = 0.25,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.2,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        # ---- explicit stage-wise LoRA configs ----
        specific_attn_rank_per_stage = _ensure_stage_tuple(specific_attn_rank_per_stage, "specific_attn_rank_per_stage")
        specific_attn_alpha_per_stage = _ensure_stage_tuple(specific_attn_alpha_per_stage, "specific_attn_alpha_per_stage")
        mlp_rank_per_stage = _ensure_stage_tuple(mlp_rank_per_stage, "mlp_rank_per_stage")
        mlp_alpha_per_stage = _ensure_stage_tuple(mlp_alpha_per_stage, "mlp_alpha_per_stage")
        shared_rank_per_stage = _ensure_stage_tuple(shared_rank_per_stage, "shared_rank_per_stage")
        shared_alpha_per_stage = _ensure_stage_tuple(shared_alpha_per_stage, "shared_alpha_per_stage")
        branch_rank_per_stage = _ensure_stage_tuple(branch_rank_per_stage, "branch_rank_per_stage")
        branch_alpha_per_stage = _ensure_stage_tuple(branch_alpha_per_stage, "branch_alpha_per_stage")

        if mix_token_ratio_per_stage is None:
            mix_token_ratio_per_stage = (float(mix_token_ratio),) * 4
        if mix_heads_frac_per_stage is None:
            mix_heads_frac_per_stage = (float(mix_heads_frac),) * 4
        assert len(mix_token_ratio_per_stage) == 4 and len(mix_heads_frac_per_stage) == 4,             "mix_token_ratio_per_stage and mix_heads_frac_per_stage must be length-4 tuples for stages 1..4"

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        # ---- patch embed layer ----
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, in_chans=x_in_chans)
        self.patch_embed_y = PatchEmbed(embed_dim=embed_dim, in_chans=y_in_chans)

        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            # local dilation aligned with SSCL
            local_dilation = LOCAL_DILATION_PER_STAGE.get(cur_stage, 1)

            # enable SSCL v2 ONLY in global-attn blocks (window_size == 0)
            enable_sscl = (window_size == 0)

            block = MultiScaleBlock(
                specific_attn_rank=specific_attn_rank_per_stage[cur_stage - 1],
                specific_attn_alpha=specific_attn_alpha_per_stage[cur_stage - 1],
                mlp_rank=mlp_rank_per_stage[cur_stage - 1],
                mlp_alpha=mlp_alpha_per_stage[cur_stage - 1],
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                enable_sscl=enable_sscl,
                shared_rank=shared_rank_per_stage[cur_stage - 1],
                shared_alpha=shared_alpha_per_stage[cur_stage - 1],
                branch_rank=branch_rank_per_stage[cur_stage - 1],
                branch_alpha=branch_alpha_per_stage[cur_stage - 1],
                sscl_lp_kernel=sscl_lp_kernel,
                mix_token_ratio=mix_token_ratio_per_stage[cur_stage - 1],
                mix_heads_frac=mix_heads_frac_per_stage[cur_stage - 1],
                detach_conflict=detach_conflict,
                conf_semantic_gate=conf_semantic_gate,
                conf_semantic_gate_smooth_ks=conf_semantic_gate_smooth_ks,
                conf_semantic_gate_scale_init=conf_semantic_gate_scale_init,
                conf_head_select_mode=conf_head_select_mode,
                conf_head_select_seed=conf_head_select_seed,
                attn_dropout_p=(attn_dropout_p_global if window_size == 0 else 0.0),
                conf_topk=conf_topk,
                prototype_route_temperature=prototype_route_temperature,
                prototype_prior_conf_weight=prototype_prior_conf_weight,
                prototype_prior_energy_weight=prototype_prior_energy_weight,
                prototype_prior_sem_weight=prototype_prior_sem_weight,
                prototype_residual_hidden_ratio=prototype_residual_hidden_ratio,
                local_dilation=local_dilation,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x, y):
        x = self.patch_embed(x)
        y = self.patch_embed_y(y)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])
        y = y + self._get_pos_embed(y.shape[1:3])

        outputs = []
        outputs_y = []
        for i, blk in enumerate(self.blocks):
            x, y = blk(x, y)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                feats_y = y.permute(0, 3, 1, 2)
                outputs.append(feats)
                outputs_y.append(feats_y)

        return outputs, outputs_y

    def get_layer_id(self, layer_name):
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 1, 512, 512)

    encoder = Hiera(
        x_in_chans=3,
        y_in_chans=1,
        specific_attn_rank_per_stage=(16, 32, 64, 64),
        specific_attn_alpha_per_stage=(16, 32, 64, 64),
        shared_rank_per_stage=(0, 0, 32, 32),
        shared_alpha_per_stage=(0, 0, 32, 32),
        branch_rank_per_stage=(0, 0, 32, 32),
        branch_alpha_per_stage=(0, 0, 32, 32),
        # recommended defaults for generalization
        drop_path_rate=0.2,
        attn_dropout_p_global=0.1,
        conf_topk=4,
        mix_token_ratio_per_stage=(0.0, 0.0, 0.05, 0.1),
        mix_heads_frac_per_stage=(0.0, 0.0, 0.25, 0.25),
        detach_conflict=True,
    )

    x_list, y_list = encoder(x, y)
    for i in range(len(x_list)):
        print(x_list[i].shape, y_list[i].shape)
