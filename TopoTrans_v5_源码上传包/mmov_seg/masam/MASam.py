import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .Encoder import Hiera
from .Fusion import MultiScaleFreqSEFusionV2


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            )
        )


class SeparableConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
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
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, h, w = x.size()
        if w % ps != 0:
            x = F.pad(x, (0, ps - w % ps), mode='reflect')
        if h % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - h % ps), mode='reflect')
        return x

    @staticmethod
    def pad_out(x):
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        b, c, h, w = x.shape
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        _, _, hp, wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv,
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads,
            d=c // self.num_heads,
            hh=hp // self.ws,
            ww=wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )

        dots = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = (dots.softmax(dim=-1) @ v)
        attn = rearrange(
            attn,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads,
            d=c // self.num_heads,
            hh=hp // self.ws,
            ww=wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )
        attn = attn[:, :, :h, :w]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        return out[:, :, :h, :w]


class Block(nn.Module):
    def __init__(
        self,
        dim=256,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.08,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU6,
        norm_layer=nn.BatchNorm2d,
        window_size=8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        return self.post_conv(x)


class PriorPropagatedMAFRN(nn.Module):
    """Lightweight prior-aware refinement head used at 4x scale only."""
    def __init__(
        self,
        in_channels: int = 64,
        decode_channels: int = 64,
        use_hp_inject: bool = True,
        detach_prior: bool = True,
        prior_gate_scale_init: float = 0.0,
        prior_hp_scale_init: float = 0.0,
    ):
        super().__init__()
        self.eps = 1e-8
        self.use_hp_inject = bool(use_hp_inject)
        self.detach_prior = bool(detach_prior)
        c = int(decode_channels)

        self.pre_conv = Conv(in_channels, c, kernel_size=1)
        self.post_conv = ConvBNReLU(c, c, kernel_size=3)
        self.fuse_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        self.exp_local = SeparableConvBN(c, c, kernel_size=3)
        self.exp_context = SeparableConvBN(c, c, kernel_size=3, dilation=2)
        self.exp_bottl = ConvBN(c, c, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(c, 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.prior_gate_scale = nn.Parameter(torch.tensor(float(prior_gate_scale_init)))
        self.prior_hp_scale = nn.Parameter(torch.tensor(float(prior_hp_scale_init)))

        self.res_align = Conv(in_channels, c, kernel_size=1)
        self.blur_dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        with torch.no_grad():
            k = torch.ones(1, 1, 3, 3) / 9.0
            self.blur_dw.weight.copy_(k.repeat(c, 1, 1, 1))
        self.blur_dw.weight.requires_grad_(False)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

        self.shortcut = ConvBN(c, c, kernel_size=1)
        self.proj = SeparableConvBN(c, c, kernel_size=3)
        self.act = nn.ReLU6()

    @staticmethod
    def _prior_to_map(prior, h: int, w: int, device, dtype):
        if prior is None:
            return None
        pi = None
        ws = None
        if isinstance(prior, dict):
            pi = prior.get('pi', None)
            ws = prior.get('ws', None)
        elif isinstance(prior, (tuple, list)):
            if len(prior) > 0:
                pi = prior[0]
            if len(prior) > 1:
                ws = prior[1]
        else:
            pi = prior

        if ws is not None:
            ws_map = ws[:, :, 0, :, :] if ws.dim() == 5 else ws
            return ws_map.to(device=device, dtype=dtype)

        if pi is not None:
            if pi.dim() == 4:
                pi_map = pi
            elif pi.dim() == 2:
                pi_map = pi[:, :, None, None]
            else:
                return None
            return pi_map.expand(-1, -1, h, w).to(device=device, dtype=dtype)
        return None

    def forward(self, x, res, prior=None):
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res_p = self.pre_conv(res)
        w = F.softplus(self.fuse_logits)
        fuse_w = w / (torch.sum(w, dim=0, keepdim=True) + self.eps)
        base = self.post_conv(fuse_w[0] * res_p + fuse_w[1] * x_up)

        b, c, h, w = base.shape
        prior_map = self._prior_to_map(prior, h, w, device=base.device, dtype=base.dtype)
        if (prior_map is not None) and self.detach_prior:
            prior_map = prior_map.detach()

        logits = self.gate(base)
        if prior_map is not None:
            py = prior_map[:, 1:2, :, :]
            px = 1.0 - py
            bias = torch.cat([py, px, torch.zeros_like(py)], dim=1)
            logits = logits + self.prior_gate_scale * bias

        mix = F.softmax(logits, dim=1)
        moe_out = (
            mix[:, 0:1] * self.exp_local(base)
            + mix[:, 1:2] * self.exp_context(base)
            + mix[:, 2:3] * self.exp_bottl(base)
        )
        out = base + moe_out

        if self.use_hp_inject:
            res_q = self.res_align(res)
            hp = res_q - self.blur_dw(res_q)
            if prior_map is not None:
                py = prior_map[:, 1:2, :, :]
                out = out + self.gamma * (hp * (1.0 + self.prior_hp_scale * py))
            else:
                out = out + self.gamma * hp

        out = self.proj(out) + self.shortcut(out)
        return self.act(out)


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels=(96, 192, 384, 768),
        decode_channels=64,
        dropout=0.05,
        window_size=4,
        num_classes=6,
    ):
        super().__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.p1 = PriorPropagatedMAFRN(
            in_channels=encoder_channels[-4],
            decode_channels=decode_channels,
            use_hp_inject=True,
            detach_prior=True,
        )
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1),
        )
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w, priors=None):
        x = self.b4(self.pre_conv(res4))
        x = self.b3(self.p3(x, res3))
        x = self.b2(self.p2(x, res2))

        prior_4x = None
        if priors is not None:
            if isinstance(priors, dict):
                pi_list = priors.get('pi', None)
                ws_list = priors.get('ws', None)
                wxc_list = priors.get('wxc', None)
                wyc_list = priors.get('wyc', None)
                if (ws_list is not None) and (len(ws_list) > 0):
                    prior_4x = {
                        'pi': (pi_list[0] if (pi_list is not None and len(pi_list) > 0) else None),
                        'ws': ws_list[0],
                        'wxc': (wxc_list[0] if (wxc_list is not None and len(wxc_list) > 0) else None),
                        'wyc': (wyc_list[0] if (wyc_list is not None and len(wyc_list) > 0) else None),
                    }
                elif (pi_list is not None) and (len(pi_list) > 0):
                    prior_4x = {'pi': pi_list[0]}
            else:
                prior_4x = priors

        x = self.p1(x, res1, prior=prior_4x)
        x = self.segmentation_head(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MASam(nn.Module):
    def __init__(
        self,
        embed_dim=96,
        n_class=6,
        decoder_chans=64,
        # Kept for compatibility with older training scripts. If the explicit
        # per-stage tuples below are None, these values will be used to build
        # simple stage-wise defaults.
        lora_r=8,
        lora_alpha=8,
        k_rank_lora=2,
        k_lora_alpha=2,
        v_rank_lora=2,
        v_lora_alpha=2,
        num_last_blocks_no_detach=3,
        # New Encoder_multifreq_sparse.py interface: stage-wise modality-specific LoRA.
        specific_attn_rank_per_stage=None,
        specific_attn_alpha_per_stage=None,
        mlp_rank_per_stage=None,
        mlp_alpha_per_stage=None,
        # Shared low-frequency semantic KV branch. It is intentionally enabled
        # only in deeper global-attention stages by default.
        shared_rank_per_stage=(0, 0, 16, 32),
        shared_alpha_per_stage=(0, 0, 16, 32),
        # Kept for Encoder compatibility. In the revised Encoder, branch-specific
        # low-pass ΔK/ΔV is intentionally disabled / unused.
        branch_rank_per_stage=(0, 0, 0, 0),
        branch_alpha_per_stage=(0, 0, 0, 0),
        # Multi-frequency sparse prototype controls.
        sscl_lp_kernel=5,
        mix_token_ratio_per_stage=(0.0, 0.0, 0.25, 0.15),
        mix_heads_frac_per_stage=(0.0, 0.0, 0.25, 0.25),
        prototype_route_temperature=0.90,
        prototype_prior_conf_weight=1.0,
        prototype_prior_energy_weight=0.20,
        prototype_prior_sem_weight=0.25,
        prototype_residual_hidden_ratio=0.4,
        conf_semantic_gate=True,
        conf_semantic_gate_smooth_ks=5,
        conf_semantic_gate_scale_init=0.10,
        detach_conflict=True,
        attn_dropout_p_global=0.08,
        drop_path_rate=0.10,
        conf_topk=6,
        # Fusion controls. With the new Encoder already using multi-frequency
        # prototypes, keep the later fusion prior relatively mild.
        fusion_highfreq_top_k=2,
        fusion_spatial_top_k=2,
        fusion_freeze_hp_epochs=3,
        fusion_prior_strength=0.5,
        fusion_modality_drop=0.00,
        fusion_reduction=8,
        fusion_hpf_learnable=False,
        fusion_lpf_kernel_size=5,
        fusion_se_bal_lambda=1e-4,
        fusion_lf_align_lambda=2e-3,
        fusion_deficiency_learn_scale_init=0.0,
        fusion_compensation_scale_init=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seg_classes = n_class

        # Preserve old attributes so existing logs/checkpoints/training scripts
        # that read them do not break.
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.k_rank_lora = k_rank_lora
        self.k_lora_alpha = k_lora_alpha
        self.v_rank_lora = v_rank_lora
        self.v_lora_alpha = v_lora_alpha
        self.no_detach = num_last_blocks_no_detach

        # If explicit per-stage settings are not given, use conservative defaults
        # for the new multi-frequency sparse prototype Encoder. These are less
        # aggressive than the old all-stage specific_rank=64 setting and avoid
        # overfitting shallow stages.
        if specific_attn_rank_per_stage is None:
            specific_attn_rank_per_stage = (16, 32, 32, 48)
        if specific_attn_alpha_per_stage is None:
            specific_attn_alpha_per_stage = specific_attn_rank_per_stage
        if mlp_rank_per_stage is None:
            mlp_rank_per_stage = (16, 32, 32, 48)
        if mlp_alpha_per_stage is None:
            mlp_alpha_per_stage = mlp_rank_per_stage

        self.encoder = Hiera(
            embed_dim=self.embed_dim,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9, 10],
            x_in_chans=3,
            y_in_chans=3,
            specific_attn_rank_per_stage=specific_attn_rank_per_stage,
            specific_attn_alpha_per_stage=specific_attn_alpha_per_stage,
            mlp_rank_per_stage=mlp_rank_per_stage,
            mlp_alpha_per_stage=mlp_alpha_per_stage,
            shared_rank_per_stage=shared_rank_per_stage,
            shared_alpha_per_stage=shared_alpha_per_stage,
            branch_rank_per_stage=branch_rank_per_stage,
            branch_alpha_per_stage=branch_alpha_per_stage,
            sscl_lp_kernel=sscl_lp_kernel,
            drop_path_rate=drop_path_rate,
            attn_dropout_p_global=attn_dropout_p_global,
            conf_topk=conf_topk,
            mix_token_ratio_per_stage=mix_token_ratio_per_stage,
            mix_heads_frac_per_stage=mix_heads_frac_per_stage,
            prototype_route_temperature=prototype_route_temperature,
            prototype_prior_conf_weight=prototype_prior_conf_weight,
            prototype_prior_energy_weight=prototype_prior_energy_weight,
            prototype_prior_sem_weight=prototype_prior_sem_weight,
            prototype_residual_hidden_ratio=prototype_residual_hidden_ratio,
            conf_semantic_gate_scale_init=conf_semantic_gate_scale_init,
            conf_semantic_gate=conf_semantic_gate,
            conf_semantic_gate_smooth_ks=conf_semantic_gate_smooth_ks,
            detach_conflict=detach_conflict,
        )

        self.fusion_module = MultiScaleFreqSEFusionV2(
            in_channels_list=[self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
            reduction=fusion_reduction,
            highfreq_top_k=fusion_highfreq_top_k,
            spatial_top_k=fusion_spatial_top_k,
            p_modality_drop=fusion_modality_drop,
            hpf_learnable=fusion_hpf_learnable,
            lpf_kernel_size=fusion_lpf_kernel_size,
            se_bal_lambda=fusion_se_bal_lambda,
            lf_align_lambda=fusion_lf_align_lambda,
            freeze_hp_epochs=fusion_freeze_hp_epochs,
            prior_strength=fusion_prior_strength,
            deficiency_learn_scale_init=fusion_deficiency_learn_scale_init,
            compensation_scale_init=fusion_compensation_scale_init,
        )

        self.decoder = Decoder(
            encoder_channels=(self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8),
            num_classes=self.seg_classes,
            decode_channels=decoder_chans,
        )

    def get_aux_loss(self):
        if hasattr(self.fusion_module, 'get_aux_loss'):
            return self.fusion_module.get_aux_loss()
        device = next(self.parameters()).device
        return torch.zeros((), device=device)

    def set_epoch(self, epoch: int):
        if hasattr(self.fusion_module, 'set_epoch'):
            self.fusion_module.set_epoch(epoch)

    def forward(self, rgb, aux1):
        h, w = rgb.size()[-2:]
        x_list, aux1_list = self.encoder(rgb, aux1)
        fuse_list = self.fusion_module(x_list, aux1_list)
        priors = self.fusion_module.get_last_priors(detach=True) if hasattr(self.fusion_module, 'get_last_priors') else None
        return self.decoder(fuse_list[0], fuse_list[1], fuse_list[2], fuse_list[3], h, w, priors=priors)
