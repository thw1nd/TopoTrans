import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ------------------------- 0) helpers -------------------------
def conv1x1(in_c: int, out_c: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=bias)


def dwconv3x3(c: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=bias)


# ------------------------- 1) ModalityDrop -------------------------
class ModalityDrop(nn.Module):
    def __init__(self, p_drop: float = 0.1, mode: str = "zero"):
        super().__init__()
        self.p = float(p_drop)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.training) or self.p <= 0:
            return x, y
        r = torch.rand((), device=x.device)
        if r < self.p / 2:
            y = torch.zeros_like(y) if self.mode == "zero" else y.std() * torch.randn_like(y)
        elif r < self.p:
            x = torch.zeros_like(x) if self.mode == "zero" else x.std() * torch.randn_like(x)
        return x, y


# ------------------------- 2) Fixed filters -------------------------
class FixedHPF(nn.Module):
    def __init__(self, channels: int, learnable: bool = False):
        super().__init__()
        self.hpf = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        with torch.no_grad():
            k = torch.tensor([[0.0, -1.0, 0.0],
                              [-1.0, 4.0, -1.0],
                              [0.0, -1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.hpf.weight.copy_(k.repeat(channels, 1, 1, 1))
        self.hpf.weight.requires_grad_(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hpf(x)


class FixedLPF(nn.Module):
    """Depthwise blur used to expose low-frequency structure explicitly."""
    def __init__(self, channels: int, kernel_size: int = 5, learnable: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = int(kernel_size)
        self.lpf = nn.Conv2d(
            channels, channels, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, groups=channels, bias=False
        )
        with torch.no_grad():
            k = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
            k = k / float(self.kernel_size * self.kernel_size)
            self.lpf.weight.copy_(k.repeat(channels, 1, 1, 1))
        self.lpf.weight.requires_grad_(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lpf(x)


# ------------------------- 3) Role-aware router -------------------------
class PrimaryRouter2M(nn.Module):
    """Global primary prior pi=[pi_x, pi_y], shape [B,2,1,1].

    Hybrid design:
      - keep the old Fusion's modality-directed reliability idea by penalizing
        modality-specific noisy high-frequency responses;
      - keep the New_Fusion's explicit low/high decomposition;
      - avoid the shared-denominator cancellation problem by using cross-modal
        disagreement as a confidence controller rather than as the same divisor
        for both modality logits.
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        eps: float = 1e-6,
        min_confidence: float = 0.25,
    ):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.eps = float(eps)
        self.min_confidence = float(min(max(min_confidence, 0.0), 0.95))

        # low_x, low_y, detail_x, detail_y, dlow, dhigh, noise_x, noise_y
        self.calib = nn.Sequential(
            conv1x1(channels * 8, mid),
            nn.ReLU(inplace=True),
            conv1x1(mid, 2),
        )
        self.learn_scale = nn.Parameter(torch.tensor(0.0))
        self.analytic_scale = nn.Parameter(torch.tensor(1.0))
        self.noise_scale = nn.Parameter(torch.tensor(1.0))
        self.disagreement_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1)

    @staticmethod
    def _detail_noise(h: torch.Tensor) -> torch.Tensor:
        # A small local fluctuation proxy. It is not used as a hard noise label;
        # it only lowers the analytic confidence of a modality when its high-
        # frequency response is highly unstable.
        return torch.abs(h - F.avg_pool2d(h, kernel_size=3, stride=1, padding=1))

    def forward(
        self,
        lx: torch.Tensor,
        ly: torch.Tensor,
        hx: torch.Tensor,
        hy: torch.Tensor,
        dl: torch.Tensor,
        dh: torch.Tensor,
    ) -> torch.Tensor:
        ax = torch.abs(hx)
        ay = torch.abs(hy)

        low_x = self._pool(torch.abs(lx))
        low_y = self._pool(torch.abs(ly))
        detail_x = self._pool(ax)
        detail_y = self._pool(ay)
        dlow = self._pool(torch.abs(dl))
        dhigh = self._pool(torch.abs(dh))
        noise_x = self._pool(self._detail_noise(ax))
        noise_y = self._pool(self._detail_noise(ay))

        low_x_s = low_x.mean(dim=1, keepdim=True)
        low_y_s = low_y.mean(dim=1, keepdim=True)
        detail_x_s = detail_x.mean(dim=1, keepdim=True)
        detail_y_s = detail_y.mean(dim=1, keepdim=True)
        dlow_s = dlow.mean(dim=1, keepdim=True)
        dhigh_s = dhigh.mean(dim=1, keepdim=True)
        noise_x_s = noise_x.mean(dim=1, keepdim=True)
        noise_y_s = noise_y.mean(dim=1, keepdim=True)

        # Modality evidence: low-frequency structure + detail richness.
        evidence_x = low_x_s + detail_x_s + self.eps
        evidence_y = low_y_s + detail_y_s + self.eps

        # Modality-specific penalty inherited from the old Fusion. Unlike a
        # shared discrepancy denominator, this term does not vanish under
        # softmax because it is different for T1 and T2.
        penalty_x = 1.0 + self.noise_scale.abs() * noise_x_s
        penalty_y = 1.0 + self.noise_scale.abs() * noise_y_s
        reliability_x = evidence_x / (penalty_x + self.eps)
        reliability_y = evidence_y / (penalty_y + self.eps)

        modality_logits = torch.log(torch.cat([reliability_x, reliability_y], dim=1) + self.eps)
        modality_logits = modality_logits - modality_logits.mean(dim=1, keepdim=True)

        # Cross-modal disagreement controls confidence rather than the x/y
        # direction. This avoids the shared-denominator cancellation issue.
        disagreement = (dlow_s + dhigh_s) / (evidence_x + evidence_y + self.eps)
        confidence = torch.exp(-self.disagreement_scale.abs() * disagreement)
        confidence = self.min_confidence + (1.0 - self.min_confidence) * confidence

        analytic_logits = self.analytic_scale.abs() * confidence * modality_logits

        # Zero-gated learned calibration uses full pooled statistics.
        feat = torch.cat([low_x, low_y, detail_x, detail_y, dlow, dhigh, noise_x, noise_y], dim=1)
        calib_logits = self.calib(feat)
        logits = analytic_logits + torch.tanh(self.learn_scale) * calib_logits
        return logits.softmax(dim=1)


# ------------------------- 4) Channel prior -------------------------
class FreqAwareGate(nn.Module):
    """Return channel prior weights w_x, w_y with shape [B,C,1,1].

    Channel selection is treated as a low-frequency semantic problem.
    So this module consumes low-frequency statistics and low-frequency discrepancy,
    while the spatial conflict module will own high-frequency discrepancy.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            conv1x1(channels * 3, mid),
            nn.ReLU(inplace=True),
            conv1x1(mid, channels * 2),
        )

    def forward(
        self,
        lx: torch.Tensor,
        ly: torch.Tensor,
        dl: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        prior_strength: float = 0.0,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low_x = F.adaptive_avg_pool2d(torch.abs(lx), 1)
        low_y = F.adaptive_avg_pool2d(torch.abs(ly), 1)
        dlow = F.adaptive_avg_pool2d(torch.abs(dl), 1)

        logits = self.mlp(torch.cat([low_x, low_y, dlow], dim=1))
        B, twoC, _, _ = logits.shape
        C = twoC // 2
        logits = logits.view(B, 2, C, 1, 1)

        if (pi is not None) and (prior_strength > 0):
            prior = pi.clamp_min(eps).log() * float(prior_strength)
            logits = logits + prior.view(B, 2, 1, 1, 1)

        gates = logits.softmax(dim=1)
        return gates[:, 0, :, :, :], gates[:, 1, :, :, :]


# ------------------------- 5) Spatial prior -------------------------
class SpatialConflictGate(nn.Module):
    """Directional spatial reliability gate.

    The old gate consumed only |dh| and |dl|, which are symmetric under modality
    exchange and therefore reveal where modalities disagree but not which modality
    is locally more reliable. This version additionally observes |hx| and |hy| so
    the 2-channel output can learn modality-directed spatial reliability.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.net = nn.Sequential(
            conv1x1(channels * 4, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, kernel_size=3, padding=1, bias=True),
        )

    @staticmethod
    def _normalize_map(x: torch.Tensor, eps: float) -> torch.Tensor:
        # Per-sample, per-channel normalization keeps this gate from simply
        # following feature magnitude scale changes across stages/domains.
        denom = x.mean(dim=(2, 3), keepdim=True).clamp_min(eps)
        return torch.tanh(x / denom)

    def forward(
        self,
        hx: torch.Tensor,
        hy: torch.Tensor,
        dh: torch.Tensor,
        dl: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
        prior_strength: float = 0.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        hx_a = self._normalize_map(torch.abs(hx), eps)
        hy_a = self._normalize_map(torch.abs(hy), eps)
        dh_a = self._normalize_map(torch.abs(dh), eps)
        dl_a = self._normalize_map(torch.abs(dl), eps)
        logits = self.net(torch.cat([hx_a, hy_a, dh_a, dl_a], dim=1))

        if (pi is not None) and (prior_strength > 0):
            prior = pi.clamp_min(eps).log() * float(prior_strength)
            logits = logits + prior

        return logits.softmax(dim=1).unsqueeze(2)


# ------------------------- 6) Conflict-guided high-frequency residual -------------------------
class HighFreqResidual(nn.Module):
    """A lighter HF branch.

    It only consumes discrepancy features and is gated by the deficiency map,
    so it becomes a conditional detail supplement instead of an unconditional
    second fusion stream.
    """
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 16)
        self.proj1 = conv1x1(channels, mid, bias=True)
        self.dw = dwconv3x3(mid)
        self.proj2 = conv1x1(mid, channels, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._hp_frozen = False

    def set_hp_frozen(self, flag: bool):
        self._hp_frozen = bool(flag)

    def forward(self, base: torch.Tensor, dh: torch.Tensor, deficiency: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._hp_frozen:
            return base
        t = torch.abs(dh)
        t = self.proj2(self.act(self.dw(self.act(self.proj1(t)))))
        gate = 1.0 if deficiency is None else deficiency
        return base + torch.tanh(self.gamma) * gate * t


# ------------------------- 7) Deficiency-guided asymmetric fusion -------------------------
class PrimaryDeficiencyEstimator(nn.Module):
    """Estimate where primary still misses information.

    Revised logic:
      - analytic cues are normalized before sigmoid to avoid scale-driven saturation;
      - deficiency remains a conditional compensation mask, not an almost-always-on
        auxiliary injection;
      - learnable refinement is still zero-gated initially.
    """
    def __init__(self, channels: int, reduction: int = 8, learn_scale_init: float = 0.0, eps: float = 1e-6):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.eps = float(eps)
        self.refine = nn.Sequential(
            conv1x1(channels * 4, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.delta = nn.Parameter(torch.tensor(0.5))
        self.learn_scale = nn.Parameter(torch.tensor(float(learn_scale_init)))

    def _spatial_norm(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.mean(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return torch.tanh(x / denom)

    def forward(
        self,
        primary: torch.Tensor,
        auxiliary: torch.Tensor,
        dl: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dlow_raw = torch.abs(dl).mean(dim=1, keepdim=True)
        aux_gap_raw = torch.abs(auxiliary - primary).mean(dim=1, keepdim=True)

        # Normalize raw magnitudes before sigmoid; otherwise feature scale can make
        # the compensation mask saturate and turn auxiliary correction almost always on.
        dlow = self._spatial_norm(dlow_raw)
        aux_gap = self._spatial_norm(aux_gap_raw)

        if pi is None:
            prior_uncertainty = torch.zeros_like(dlow)
        else:
            prior_uncertainty = (1.0 - pi.max(dim=1, keepdim=True)[0]).expand_as(dlow)

        analytic = (
            self.alpha.abs() * dlow
            + self.gamma.abs() * aux_gap
            + self.delta.abs() * prior_uncertainty
        )
        learn = self.refine(torch.cat([primary, auxiliary, torch.abs(auxiliary - primary), torch.abs(dl)], dim=1))
        return torch.sigmoid(analytic + torch.tanh(self.learn_scale) * learn)


class AuxiliaryCompensator(nn.Module):
    def __init__(self, channels: int, scale_init: float = 0.0):
        super().__init__()
        mid = max(channels // 2, 16)
        self.net = nn.Sequential(
            conv1x1(channels * 2, mid),
            nn.ReLU(inplace=True),
            dwconv3x3(mid),
            nn.ReLU(inplace=True),
            conv1x1(mid, channels),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([primary, auxiliary - primary], dim=1))
        return torch.tanh(self.gamma) * delta


class MultiScaleFreqSEFusionV2(nn.Module):
    """Role-aware, lighter fusion for [4x, 8x, 16x, 32x] features.

    Information roles:
      - global routing: low/high-frequency statistics + discrepancy statistics
      - channel gating: low-frequency semantics + low-frequency discrepancy
      - spatial gating: discrepancy maps (mainly high-frequency)
      - deficiency: auxiliary gap + low-frequency discrepancy + router uncertainty
      - high-frequency residual: conditional detail supplement only
    """
    def __init__(
        self,
        in_channels_list: List[int],
        reduction: int = 8,
        highfreq_top_k: int = 1,  # number of early scales using HF residual
        spatial_top_k: int = 2,  # number of early scales using spatial gate
        p_modality_drop: float = 0.1,
        hpf_learnable: bool = False,
        lpf_kernel_size: int = 5,
        se_bal_lambda: float = 1e-4,
        lf_align_lambda: float = 2e-3,
        freeze_hp_epochs: int = 3,
        prior_strength: float = 0.4,
        eps: float = 1e-6,
        deficiency_learn_scale_init: float = 0.00,
        compensation_scale_init: float = 0.00,
    ):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.highfreq_top_k = int(max(0, highfreq_top_k))
        self.spatial_top_k = int(max(0, spatial_top_k))
        self.se_bal_lambda = float(se_bal_lambda)
        self.lf_align_lambda = float(lf_align_lambda)
        self.freeze_hp_epochs = int(max(0, freeze_hp_epochs))
        self.prior_strength = float(prior_strength)
        self.eps = float(eps)

        self.hpf_shared = nn.ModuleList([FixedHPF(c, learnable=hpf_learnable) for c in self.in_channels_list])
        self.lpf_shared = nn.ModuleList([FixedLPF(c, kernel_size=lpf_kernel_size, learnable=False) for c in self.in_channels_list])

        self.router = nn.ModuleList([PrimaryRouter2M(c, reduction=reduction, eps=eps) for c in self.in_channels_list])
        self.channel_fusion = nn.ModuleList([FreqAwareGate(c, reduction=reduction) for c in self.in_channels_list])

        self.spatial_fusion = nn.ModuleList([
            SpatialConflictGate(c) if i < self.spatial_top_k else nn.Identity()
            for i, c in enumerate(self.in_channels_list)
        ])
        self.hf_blocks = nn.ModuleList([
            HighFreqResidual(c) if i < self.highfreq_top_k else nn.Identity()
            for i, c in enumerate(self.in_channels_list)
        ])
        self.deficiency_blocks = nn.ModuleList([
            PrimaryDeficiencyEstimator(c, reduction=reduction, learn_scale_init=deficiency_learn_scale_init, eps=eps)
            for c in self.in_channels_list
        ])
        self.compensation_blocks = nn.ModuleList([
            AuxiliaryCompensator(c, scale_init=compensation_scale_init)
            for c in self.in_channels_list
        ])

        self.mod_drop = ModalityDrop(p_drop=p_modality_drop, mode="zero")
        self._aux_loss: Optional[torch.Tensor] = None
        self._cur_epoch = 0

        self.last_pi: List[torch.Tensor] = []
        self.last_wxc: List[torch.Tensor] = []
        self.last_wyc: List[torch.Tensor] = []
        self.last_ws: List[torch.Tensor] = []

        # Analysis caches for RPCF deficiency-error association.
        # These tensors are detached and overwritten at every forward pass.
        self.last_deficiency: List[torch.Tensor] = []
        self.last_primary: List[torch.Tensor] = []
        self.last_auxiliary: List[torch.Tensor] = []
        self.last_delta: List[torch.Tensor] = []
        self.last_fused_before_hf: List[torch.Tensor] = []
        self.last_fused: List[torch.Tensor] = []

        # When True, forward() returns the primary-only features instead of
        # the full compensated RPCF features. Use this only during analysis to
        # obtain primary-only logits through the normal decoder.
        self.return_primary_only: bool = False

    def set_epoch(self, epoch: int):
        self._cur_epoch = int(epoch)
        freeze = self._cur_epoch < self.freeze_hp_epochs
        for blk in self.hf_blocks:
            if isinstance(blk, HighFreqResidual):
                blk.set_hp_frozen(freeze)

    def reset_aux_loss(self):
        self._aux_loss = None

    def reset_priors(self):
        self.last_pi = []
        self.last_wxc = []
        self.last_wyc = []
        self.last_ws = []
        self.reset_analysis_cache()

    def reset_analysis_cache(self):
        self.last_deficiency = []
        self.last_primary = []
        self.last_auxiliary = []
        self.last_delta = []
        self.last_fused_before_hf = []
        self.last_fused = []

    def set_return_primary_only(self, flag: bool):
        self.return_primary_only = bool(flag)

    def get_last_rpcf_analysis(self, detach: bool = True):
        def _pack(xs):
            return [x.detach() for x in xs] if detach else list(xs)
        return {
            'deficiency': _pack(self.last_deficiency),
            'primary': _pack(self.last_primary),
            'auxiliary': _pack(self.last_auxiliary),
            'delta': _pack(self.last_delta),
            'fused_before_hf': _pack(self.last_fused_before_hf),
            'fused': _pack(self.last_fused),
            'pi': _pack(self.last_pi),
            'wxc': _pack(self.last_wxc),
            'wyc': _pack(self.last_wyc),
            'ws': _pack(self.last_ws),
        }

    def get_last_priors(self, detach: bool = True):
        if detach:
            return {
                'pi':  [p.detach() for p in self.last_pi],
                'wxc': [p.detach() for p in self.last_wxc],
                'wyc': [p.detach() for p in self.last_wyc],
                'ws':  [p.detach() for p in self.last_ws],
            }
        return {
            'pi':  list(self.last_pi),
            'wxc': list(self.last_wxc),
            'wyc': list(self.last_wyc),
            'ws':  list(self.last_ws),
        }

    def get_aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            device = self.hpf_shared[0].hpf.weight.device
            return torch.zeros((), dtype=torch.float32, device=device)
        return self._aux_loss

    def _se_balance_reg(self, w_x: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        if self.se_bal_lambda <= 0:
            return w_x.new_zeros(())
        p = w_x.mean(dim=(0, 2, 3)).clamp(self.eps, 1.0 - self.eps)
        t = pi[:, 0, :, :].mean().clamp(self.eps, 1.0 - self.eps)
        kl = p * (p.log() - t.log()) + (1 - p) * ((1 - p).log() - (1 - t).log())
        return kl.mean() * self.se_bal_lambda

    def _lowfreq_align(self, lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
        if self.lf_align_lambda <= 0:
            return lx.new_zeros(())
        px = F.adaptive_avg_pool2d(lx, 1).flatten(1)
        py = F.adaptive_avg_pool2d(ly, 1).flatten(1)
        cos = F.cosine_similarity(px, py, dim=1).mean()
        return (1.0 - cos) * self.lf_align_lambda

    def forward(self, feats_x: List[torch.Tensor], feats_y: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats_x) == len(feats_y) == len(self.in_channels_list)

        self.reset_aux_loss()
        self.reset_priors()
        out: List[torch.Tensor] = []
        aux_loss: Optional[torch.Tensor] = None
        n = len(self.in_channels_list)

        for i, (x, y, hpf, lpf) in enumerate(zip(feats_x, feats_y, self.hpf_shared, self.lpf_shared)):
            x_in, y_in = self.mod_drop(x, y) if self.training else (x, y)
            B, C, H, W = x_in.shape

            # Shared decomposition per scale.
            hx = hpf(x_in)
            hy = hpf(y_in)
            lx = lpf(x_in)
            ly = lpf(y_in)
            dh = torch.abs(hx - hy)
            dl = torch.abs(lx - ly)

            # 1) global primary prior
            pi = self.router[i](lx, ly, hx, hy, dl, dh)
            self.last_pi.append(pi)

            # 2) channel prior (semantic / low-frequency battlefield)
            w_x_c, w_y_c = self.channel_fusion[i](
                lx, ly, dl, pi=pi, prior_strength=self.prior_strength, eps=self.eps
            )
            self.last_wxc.append(w_x_c)
            self.last_wyc.append(w_y_c)

            # 3) spatial prior (discrepancy battlefield)
            if isinstance(self.spatial_fusion[i], SpatialConflictGate):
                ws = self.spatial_fusion[i](hx, hy, dh, dl, pi=pi, prior_strength=self.prior_strength, eps=self.eps)
            else:
                ws = pi.view(B, 2, 1, 1, 1).expand(B, 2, 1, H, W)
            self.last_ws.append(ws)

            # 4) primary / auxiliary formation
            wc = torch.stack([w_x_c, w_y_c], dim=1)  # [B,2,C,1,1]
            w_primary = wc * ws
            w_x = w_primary[:, 0]
            w_y = w_primary[:, 1]
            denom = (w_x + w_y).clamp_min(self.eps)

            primary = (w_x * x_in + w_y * y_in) / denom
            auxiliary = (w_y * x_in + w_x * y_in) / denom

            # 5) deficiency-guided auxiliary compensation
            deficiency = self.deficiency_blocks[i](primary, auxiliary, dl, pi=pi)
            delta = self.compensation_blocks[i](primary, auxiliary)
            f_before_hf = primary + deficiency * delta
            f = f_before_hf

            # Cache RPCF internal states for mechanism analysis.
            # The first scale usually has the highest spatial resolution and is
            # often the most interpretable for M_def/error-map visualization.
            self.last_deficiency.append(deficiency.detach())
            self.last_primary.append(primary.detach())
            self.last_auxiliary.append(auxiliary.detach())
            self.last_delta.append(delta.detach())
            self.last_fused_before_hf.append(f_before_hf.detach())

            # regularization
            if self.training:
                reg = self._se_balance_reg(w_x_c, pi)
                if (i >= n - 2) and (self.lf_align_lambda > 0):
                    reg = reg + self._lowfreq_align(lx, ly)
                aux_loss = reg if aux_loss is None else (aux_loss + reg)

            # 6) conditional high-frequency detail supplement
            hf_blk = self.hf_blocks[i]
            if isinstance(hf_blk, HighFreqResidual):
                f = hf_blk(f, dh, deficiency=deficiency)

            self.last_fused.append(f.detach())

            # During analysis, return primary-only features so that the normal
            # decoder can produce primary-only logits. During normal training /
            # testing this flag must stay False.
            out.append(primary if self.return_primary_only else f)

        self._aux_loss = out[0].new_zeros(()) if aux_loss is None else aux_loss
        return out
