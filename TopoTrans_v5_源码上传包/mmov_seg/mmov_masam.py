import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .masam_adapter import MASamDINOBranch
from .modeling.transformer.MMOVPredictor import MMOVPredictor
from .modeling.transformer.MMOVPredictorV2 import MMOVPredictorV2


class MMOVMASam(nn.Module):
    """MM-OVSeg with MASam fuse_list replacing the DINO RGB/SAR branch."""

    def __init__(
        self,
        train_class_json: str,
        test_class_json: str,
        masam_weights: str,
        num_classes: int,
        ignore_value: int = 255,
        freeze_masam: bool = True,
        masam_weights_type: str = "sam2_trunk",
        clip_pretrained: str = "ViT-B/16",
        clip_finetune: str = "attention",
        input_size: Tuple[int, int] = (384, 384),
        predictor_version: str = "v1",
        prompt_ensemble_type: str = "single",
        v2_margin_threshold: float = 0.04,
        v2_margin_sharpness: float = 40.0,
        v2_affinity_kernel: int = 3,
        v2_affinity_temperature: float = 0.07,
        v2_affinity_channels: int = 32,
        v2_decoder_kernel: int = 3,
        v2_decoder_max_strength: float = 0.2,
        v2_decoder_max_delta: float = 0.5,
        v2_guided_kernel: int = 3,
        v2_guided_temperature: float = 0.07,
        v2_guided_spatial_sigma: float = 1.0,
        v2_guided_max_strength: float = 1.0,
        v2_guided_initial_strength: float = 0.01,
        v2_guided_detach_margin: bool = True,
    ):
        super().__init__()
        self.ignore_value = int(ignore_value)
        self.num_classes = int(num_classes)
        self.input_size = tuple(input_size)
        self.predictor_version = str(predictor_version).lower()

        self.clip_pixel_mean = nn.Parameter(
            torch.tensor([122.7709383, 116.7460125, 104.09373615]).view(1, 3, 1, 1),
            requires_grad=False,
        )
        self.clip_pixel_std = nn.Parameter(
            torch.tensor([68.5005327, 66.6321579, 70.3231630]).view(1, 3, 1, 1),
            requires_grad=False,
        )

        v2_versions = ("v2", "v2_decoder", "v2_guided", "v3")
        if self.predictor_version not in ("v1", "baseline", *v2_versions):
            raise ValueError(
                f"Unknown predictor_version: {predictor_version}. "
                "Available: v1/baseline/v2/v2_decoder/v2_guided/v3."
            )
        masam_branch_cls = MASamDINOBranch
        predictor_cls = MMOVPredictorV2 if self.predictor_version in v2_versions else MMOVPredictor

        predictor_kwargs = dict(
            train_class_json=train_class_json,
            test_class_json=test_class_json,
            clip_pretrained=clip_pretrained,
            prompt_ensemble_type=prompt_ensemble_type,
            text_guidance_dim=512,
            text_guidance_proj_dim=128,
            appearance_guidance_dim=512,
            appearance_guidance_proj_dim=128,
            prompt_depth=0,
            prompt_length=0,
            decoder_dims=[64, 32],
            decoder_clip_guidance_dims=[256, 128],
            decoder_clip_guidance_proj_dims=[32, 16],
            decoder_dino_guidance_dims=[256, 128],
            decoder_dino_guidance_proj_dims=[32, 16],
            decoder_guidance_dims=[256, 128],
            decoder_guidance_proj_dims=[32, 16],
            use_clip_corr=True,
            use_dino_corr=True,
            fusion_type="query_guided",
            num_heads=4,
            num_layers=2,
            hidden_dims=128,
            pooling_sizes=[2, 2],
            feature_resolution=[24, 24],
            window_sizes=12,
            attention_type="linear",
        )
        if self.predictor_version in v2_versions:
            predictor_kwargs.update(
                margin_threshold=v2_margin_threshold,
                margin_sharpness=v2_margin_sharpness,
                affinity_kernel=v2_affinity_kernel,
                affinity_temperature=v2_affinity_temperature,
                max_guidance_channels=v2_affinity_channels,
                refinement_mode=(
                    "guided_upsampling"
                    if self.predictor_version == "v2_guided"
                    else "multiscale_guided_upsampling"
                    if self.predictor_version == "v3"
                    else "local_affinity"
                ),
                guided_kernel=v2_guided_kernel,
                guided_temperature=v2_guided_temperature,
                guided_spatial_sigma=v2_guided_spatial_sigma,
                guided_max_strength=v2_guided_max_strength,
                guided_initial_strength=v2_guided_initial_strength,
                guided_detach_margin=v2_guided_detach_margin,
                use_spatial_decoder=self.predictor_version == "v2_decoder",
                decoder_kernel=v2_decoder_kernel,
                decoder_max_strength=v2_decoder_max_strength,
                decoder_max_delta=v2_decoder_max_delta,
            )
        self.predictor = predictor_cls(**predictor_kwargs)

        masam_branch_kwargs = {
            "weights": masam_weights,
            "freeze_masam": freeze_masam,
            "output_resolution": input_size,
            "weights_type": masam_weights_type,
            "guidance_source": (
                "native_pyramid"
                if self.predictor_version in ("v2_guided", "v3")
                else "upsampled_stage3"
            ),
        }
        self.masam_branch = masam_branch_cls(**masam_branch_kwargs)

        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15]
        self.layers: List[torch.Tensor] = []
        for idx in self.layer_indexes:
            self.predictor.clip_model.visual.transformer.resblocks[idx].register_forward_hook(
                lambda _m, _inp, out: self.layers.append(out)
            )

        clip_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(clip_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(clip_dim, 128, kernel_size=4, stride=4)

        self._set_clip_trainable(clip_finetune)

    @staticmethod
    def load_classes(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _set_clip_trainable(self, mode: str):
        for name, param in self.predictor.clip_model.named_parameters():
            if mode == "attention":
                param.requires_grad = ("attn" in name and ("q_proj" in name or "v_proj" in name)) or ("position" in name)
            elif mode == "full":
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _clip_inputs(self, rgb: torch.Tensor):
        rgb = F.interpolate(rgb, size=self.input_size, mode="bilinear", align_corners=False)
        return (rgb - self.clip_pixel_mean.to(rgb.dtype)) / self.clip_pixel_std.to(rgb.dtype)

    def _masam_inputs(self, x: torch.Tensor):
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

    def forward_logits(self, rgb: torch.Tensor, sar: torch.Tensor):
        self.layers = []
        clip_images = self._clip_inputs(rgb)
        clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode="bilinear", align_corners=False)

        clip_features = self.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        clip_tokens = clip_features[:, 1:, :]
        res3 = rearrange(clip_tokens, "b (h w) c -> b c h w", h=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(h w) b c -> b c h w", h=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(h w) b c -> b c h w", h=24)

        dino_feat_down, dino_guidance = self.masam_branch(
            self._masam_inputs(rgb), self._masam_inputs(sar)
        )

        clip_guidance: Dict[str, torch.Tensor] = {
            "res5": self.upsample2(res5),
            "res4": self.upsample1(res4),
            "res3": res3,
        }
        return self.predictor(res3, dino_feat_down, clip_guidance, dino_guidance)

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor, target: torch.Tensor = None):
        logits = self.forward_logits(rgb, sar)
        if target is None:
            return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        valid = target != self.ignore_value
        logits_nhwc = logits.permute(0, 2, 3, 1)
        onehot = torch.zeros_like(logits_nhwc)
        onehot[valid] = F.one_hot(target[valid], num_classes=logits.shape[1]).float()
        loss_map = F.binary_cross_entropy_with_logits(
            logits_nhwc,
            onehot,
            reduction="none",
        )
        if valid.any():
            loss_sem_seg = loss_map[valid].mean()
        else:
            # Keep the graph valid for an all-ignore tile without adding a loss signal.
            loss_sem_seg = logits_nhwc.sum() * 0.0
        return {"loss_sem_seg": loss_sem_seg}
