from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masam import MASam


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "net"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _strip_prefixes(state_dict):
    cleaned = OrderedDict()
    prefixes = ("net.", "module.", "model.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _sam2_trunk_state_dict(checkpoint):
    state_dict = _unwrap_state_dict(checkpoint)
    trunk = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("image_encoder.trunk."):
            trunk[key[len("image_encoder.trunk."):]] = value
        elif key.startswith(("blocks.", "patch_embed.", "pos_embed", "pos_embed_window", "norm", "x_norm")):
            trunk[key] = value
    return trunk


def _copy_compatible(model_state, target_key, source_tensor, loaded):
    if target_key in model_state and tuple(model_state[target_key].shape) == tuple(source_tensor.shape):
        model_state[target_key].copy_(source_tensor)
        loaded.add(target_key)


def load_sam2_trunk_into_masam(masam: MASam, weights: str):
    checkpoint = torch.load(weights, map_location="cpu")
    trunk = _sam2_trunk_state_dict(checkpoint)
    model_state = masam.state_dict()
    loaded = set()

    for src_key, tensor in trunk.items():
        _copy_compatible(model_state, f"encoder.{src_key}", tensor, loaded)

        if src_key.startswith("patch_embed."):
            _copy_compatible(model_state, f"encoder.patch_embed_y.{src_key[len('patch_embed.'):]}",
                             tensor, loaded)
            continue

        if src_key.startswith("blocks."):
            if ".norm1" in src_key:
                _copy_compatible(model_state, f"encoder.{src_key.replace('.norm1', '.x_norm1')}", tensor, loaded)
                _copy_compatible(model_state, f"encoder.{src_key.replace('.norm1', '.aux1_norm1')}", tensor, loaded)
                continue
            if ".norm2" in src_key:
                _copy_compatible(model_state, f"encoder.{src_key.replace('.norm2', '.x_norm2')}", tensor, loaded)
                _copy_compatible(model_state, f"encoder.{src_key.replace('.norm2', '.aux1_norm2')}", tensor, loaded)
                continue
            if ".attn.proj." in src_key:
                _copy_compatible(model_state, f"encoder.{src_key.replace('.attn.proj.', '.attn.aux1_proj.')}",
                                 tensor, loaded)
                continue
            if ".mlp.layers." in src_key:
                _copy_compatible(model_state, f"encoder.{src_key.replace('.mlp.', '.aux1_mlp.')}", tensor, loaded)
                continue

    masam.load_state_dict(model_state, strict=False)
    print(f"[MASamDINOBranch] loaded SAM2 trunk {weights}; loaded_keys={len(loaded)}")
    return loaded


def load_compatible_masam_checkpoint_into_masam(masam: MASam, weights: str):
    """Load only exact-shape parameters from a task-specific MASam checkpoint.

    The Potsdam GeoSeg checkpoint contains a task-specific decoder and a
    different LoRA-rank configuration.  The MMOV branch uses the encoder and
    fusion features, so exact-shape filtering lets us reuse the compatible
    RGB/DSM representation without silently loading incompatible decoder or
    adapter tensors.
    """

    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = _strip_prefixes(_unwrap_state_dict(checkpoint))
    model_state = masam.state_dict()
    compatible = OrderedDict()
    shape_mismatch = []
    unexpected = []
    for key, tensor in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(tensor.shape) == tuple(model_state[key].shape):
            compatible[key] = tensor
        else:
            shape_mismatch.append((key, tuple(tensor.shape), tuple(model_state[key].shape)))

    model_state.update(compatible)
    masam.load_state_dict(model_state, strict=False)
    print(
        f"[MASamDINOBranch] loaded compatible Potsdam MASam checkpoint {weights}; "
        f"loaded={len(compatible)} shape_mismatch={len(shape_mismatch)} "
        f"unexpected={len(unexpected)}"
    )
    return {
        "loaded": list(compatible),
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
    }


def _freeze_for_lora_rpcf_training(masam: MASam):
    for param in masam.parameters():
        param.requires_grad = False

    for name, param in masam.encoder.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True

    for param in masam.fusion_module.parameters():
        param.requires_grad = True


class MASamDINOBranch(nn.Module):
    """Map MASam fused multi-scale features to MM-OVSeg's DINO branch contract."""

    def __init__(
        self,
        weights: Optional[str] = None,
        embed_dim: int = 96,
        decoder_chans: int = 64,
        freeze_masam: bool = True,
        output_resolution: Tuple[int, int] = (384, 384),
        weights_type: str = "sam2_trunk",
        guidance_source: str = "upsampled_stage3",
    ):
        super().__init__()
        self.output_resolution = tuple(output_resolution)
        self.guidance_source = str(guidance_source).lower()
        if self.guidance_source not in ("upsampled_stage3", "native_pyramid"):
            raise ValueError(
                f"Unknown guidance_source: {guidance_source}. "
                "Available: upsampled_stage3/native_pyramid."
            )
        self.masam = MASam(embed_dim=embed_dim, n_class=7, decoder_chans=decoder_chans)
        # Module-analysis baselines may request the native fused stage-3
        # representation as a frozen multimodal proxy.  The production
        # predictor continues to use the projected contract below.
        self.return_fused_proxy = False
        if weights:
            if weights_type == "sam2_trunk":
                load_sam2_trunk_into_masam(self.masam, weights)
            elif weights_type == "masam":
                checkpoint = torch.load(weights, map_location="cpu")
                state_dict = _strip_prefixes(_unwrap_state_dict(checkpoint))
                missing, unexpected = self.masam.load_state_dict(state_dict, strict=False)
                print(
                    f"[MASamDINOBranch] loaded MASam checkpoint {weights}; "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
            elif weights_type == "masam_potsdam":
                load_compatible_masam_checkpoint_into_masam(self.masam, weights)
            else:
                raise ValueError(f"Unknown weights_type: {weights_type}")
        if freeze_masam:
            _freeze_for_lora_rpcf_training(self.masam)

        stage3_dim = embed_dim * 4
        self.down_proj = nn.Conv2d(stage3_dim, 512, kernel_size=1)
        if self.guidance_source == "native_pyramid":
            self.guidance_8x = nn.Conv2d(embed_dim * 2, 256, kernel_size=1)
            self.guidance_4x = nn.Conv2d(embed_dim, 128, kernel_size=1)
        else:
            self.guidance_8x = nn.Conv2d(stage3_dim, 256, kernel_size=1)
            self.guidance_4x = nn.Conv2d(stage3_dim, 128, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor):
        rgb = F.interpolate(
            rgb, size=self.output_resolution, mode="bilinear", align_corners=False
        )
        sar = F.interpolate(
            sar, size=self.output_resolution, mode="bilinear", align_corners=False
        )
        x_list, sar_list = self.masam.encoder(rgb, sar)
        fuse_list = self.masam.fusion_module(x_list, sar_list)

        stage3_feat = fuse_list[2]
        dino_feat_down = stage3_feat if self.return_fused_proxy else self.down_proj(stage3_feat)
        if self.guidance_source == "native_pyramid":
            guidance_8x = fuse_list[1]
            guidance_4x = fuse_list[0]
        else:
            guidance_8x = F.interpolate(
                stage3_feat, scale_factor=2, mode="bilinear", align_corners=False
            )
            guidance_4x = F.interpolate(
                stage3_feat, scale_factor=4, mode="bilinear", align_corners=False
            )
        dino_guidance = [
            self.guidance_8x(guidance_8x),
            self.guidance_4x(guidance_4x),
        ]
        return dino_feat_down, dino_guidance
