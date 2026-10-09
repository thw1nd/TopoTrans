import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from mmov_seg.third_party import clip
from mmov_seg.third_party import imagenet_templates


class MarginGatedAffinityRefiner(nn.Module):
    """Class-agnostic MASam affinity refinement for raw CLIP correlation logits."""

    def __init__(
        self,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        affinity_kernel: int = 3,
        affinity_temperature: float = 0.07,
        max_guidance_channels: int = 32,
    ):
        super().__init__()
        if affinity_kernel % 2 != 1:
            raise ValueError("affinity_kernel must be odd.")
        self.margin_threshold = float(margin_threshold)
        self.margin_sharpness = float(margin_sharpness)
        self.affinity_kernel = int(affinity_kernel)
        self.affinity_temperature = float(affinity_temperature)
        self.max_guidance_channels = int(max_guidance_channels)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _local_affinity_smooth(self, logits: torch.Tensor, guidance: torch.Tensor):
        b, num_classes, h, w = logits.shape
        guidance = F.interpolate(guidance, size=(h, w), mode="bilinear", align_corners=False)
        if self.max_guidance_channels > 0 and guidance.shape[1] > self.max_guidance_channels:
            guidance = guidance[:, : self.max_guidance_channels]
        guidance = F.normalize(guidance, dim=1)

        k = self.affinity_kernel
        pad = k // 2
        guidance_patches = F.unfold(guidance, kernel_size=k, padding=pad)
        guidance_patches = guidance_patches.view(b, guidance.shape[1], k * k, h * w)
        guidance_center = guidance.flatten(2).unsqueeze(2)
        affinity = (guidance_patches * guidance_center).sum(dim=1)
        affinity = affinity / max(self.affinity_temperature, 1e-6)
        affinity = affinity.softmax(dim=1)

        logit_patches = F.unfold(logits, kernel_size=k, padding=pad)
        logit_patches = logit_patches.view(b, num_classes, k * k, h * w)
        smoothed = (logit_patches * affinity.unsqueeze(1)).sum(dim=2)
        return smoothed.view(b, num_classes, h, w)

    def forward(self, clip_logits_low: torch.Tensor, clip_margin_low: torch.Tensor, guidance: torch.Tensor):
        if guidance is None:
            return F.interpolate(
                clip_logits_low,
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            )

        target_size = guidance.shape[-2:]
        clip_logits = F.interpolate(
            clip_logits_low,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        margin = F.interpolate(
            clip_margin_low.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        low_margin_gate = torch.sigmoid((self.margin_threshold - margin) * self.margin_sharpness)
        affinity_logits = self._local_affinity_smooth(clip_logits, guidance)
        return clip_logits + self.gamma * low_margin_gate * (affinity_logits - clip_logits)


class MASamGuidedSemanticUpsampler(nn.Module):
    """Transport coarse CLIP logits to fine MASam nodes with shared spatial weights."""

    def __init__(
        self,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        candidate_kernel: int = 3,
        affinity_temperature: float = 0.07,
        spatial_sigma: float = 1.0,
        max_guidance_channels: int = 32,
        max_strength: float = 1.0,
        initial_strength: float = 0.01,
        detach_margin: bool = True,
    ):
        super().__init__()
        if candidate_kernel % 2 != 1:
            raise ValueError("candidate_kernel must be odd.")
        if affinity_temperature <= 0:
            raise ValueError("affinity_temperature must be positive.")
        if spatial_sigma <= 0:
            raise ValueError("spatial_sigma must be positive.")
        if max_strength <= 0:
            raise ValueError("max_strength must be positive.")
        if not 0 < initial_strength < max_strength:
            raise ValueError("initial_strength must be between 0 and max_strength.")

        self.margin_threshold = float(margin_threshold)
        self.margin_sharpness = float(margin_sharpness)
        self.candidate_kernel = int(candidate_kernel)
        self.affinity_temperature = float(affinity_temperature)
        self.max_guidance_channels = int(max_guidance_channels)
        self.max_strength = float(max_strength)
        self.detach_margin = bool(detach_margin)

        initial_ratio = initial_strength / max_strength
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.strength_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

        pad = candidate_kernel // 2
        offsets = torch.arange(-pad, pad + 1, dtype=torch.float32)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        distance_sq = offset_x.square() + offset_y.square()
        spatial_bias = -distance_sq.flatten() / (2.0 * spatial_sigma * spatial_sigma)
        self.register_buffer("spatial_bias", spatial_bias.view(1, -1, 1, 1), persistent=False)

    def _reduce_guidance_channels(self, guidance: torch.Tensor):
        max_channels = self.max_guidance_channels
        if max_channels <= 0 or guidance.shape[1] <= max_channels:
            return guidance

        # Deterministic group averaging uses every channel without learning seen-class weights.
        channel_groups = torch.chunk(guidance, max_channels, dim=1)
        return torch.stack([group.mean(dim=1) for group in channel_groups], dim=1)

    def _expand_coarse_candidates(self, x: torch.Tensor, target_size):
        b, channels, coarse_h, coarse_w = x.shape
        k = self.candidate_kernel
        patches = F.unfold(x, kernel_size=k, padding=k // 2)
        patches = patches.view(b, channels * k * k, coarse_h, coarse_w)
        patches = F.interpolate(patches, size=target_size, mode="nearest")
        return patches.view(b, channels, k * k, *target_size)

    def forward(
        self,
        clip_logits_low: torch.Tensor,
        clip_margin_low: torch.Tensor,
        guidance: torch.Tensor,
    ):
        if guidance is None:
            return F.interpolate(
                clip_logits_low,
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            )

        b, _, coarse_h, coarse_w = clip_logits_low.shape
        target_size = guidance.shape[-2:]
        baseline_logits = F.interpolate(
            clip_logits_low,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        guidance_fine = self._reduce_guidance_channels(guidance)
        guidance_coarse = F.adaptive_avg_pool2d(guidance_fine, (coarse_h, coarse_w))
        guidance_fine = F.normalize(guidance_fine, dim=1)
        guidance_coarse = F.normalize(guidance_coarse, dim=1)
        guidance_candidates = self._expand_coarse_candidates(guidance_coarse, target_size)

        affinity = (guidance_fine.unsqueeze(2) * guidance_candidates).sum(dim=1)
        affinity = affinity / self.affinity_temperature
        affinity = affinity + self.spatial_bias.to(dtype=affinity.dtype)

        valid_coarse = torch.ones(
            (b, 1, coarse_h, coarse_w),
            device=clip_logits_low.device,
            dtype=clip_logits_low.dtype,
        )
        valid_candidates = F.unfold(
            valid_coarse,
            kernel_size=self.candidate_kernel,
            padding=self.candidate_kernel // 2,
        )
        valid_candidates = valid_candidates.view(
            b, self.candidate_kernel * self.candidate_kernel, coarse_h, coarse_w
        )
        valid_candidates = F.interpolate(
            valid_candidates,
            size=target_size,
            mode="nearest",
        ).bool()
        affinity = affinity.masked_fill(~valid_candidates, torch.finfo(affinity.dtype).min)
        affinity = affinity.softmax(dim=1)

        logit_candidates = self._expand_coarse_candidates(clip_logits_low, target_size)
        guided_logits = (logit_candidates * affinity.unsqueeze(1)).sum(dim=2)

        margin_source = clip_margin_low.detach() if self.detach_margin else clip_margin_low
        margin = F.interpolate(
            margin_source.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty_gate = torch.sigmoid(
            (self.margin_threshold - margin) * self.margin_sharpness
        )
        strength = self.max_strength * torch.sigmoid(self.strength_logit)
        mix = strength * uncertainty_gate
        return baseline_logits + mix * (guided_logits - baseline_logits)


class MultiScaleMASamGuidedSemanticUpsampler(MASamGuidedSemanticUpsampler):
    """Use native MASam 48x48 and 96x96 features for one cost-space transport."""

    expects_multiscale_guidance = True

    def _fuse_guidance(self, guidance, target_size):
        if isinstance(guidance, (list, tuple)):
            if not guidance:
                return None
            fine_guidance = guidance[-1]
            if len(guidance) == 1:
                return fine_guidance
            mid_guidance = guidance[0]
            mid_guidance = F.interpolate(
                mid_guidance,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            # Normalize each scale before concatenation so the 48x48 context
            # and 96x96 detail have comparable influence on affinity.
            fine_guidance = F.normalize(fine_guidance, dim=1)
            mid_guidance = F.normalize(mid_guidance, dim=1)
            return torch.cat((mid_guidance, fine_guidance), dim=1)
        return guidance

    def forward(
        self,
        clip_logits_low: torch.Tensor,
        clip_margin_low: torch.Tensor,
        guidance,
    ):
        guidance = self._fuse_guidance(guidance, guidance[-1].shape[-2:] if isinstance(guidance, (list, tuple)) and guidance else None)
        if guidance is None:
            return F.interpolate(
                clip_logits_low,
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            )

        b, _, coarse_h, coarse_w = clip_logits_low.shape
        target_size = guidance.shape[-2:]
        baseline_logits = F.interpolate(
            clip_logits_low,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        guidance_fine = self._reduce_guidance_channels(guidance)
        guidance_coarse = F.adaptive_avg_pool2d(guidance_fine, (coarse_h, coarse_w))
        guidance_fine = F.normalize(guidance_fine, dim=1)
        guidance_coarse = F.normalize(guidance_coarse, dim=1)
        guidance_candidates = self._expand_coarse_candidates(guidance_coarse, target_size)

        affinity = (guidance_fine.unsqueeze(2) * guidance_candidates).sum(dim=1)
        affinity = affinity / self.affinity_temperature
        affinity = affinity + self.spatial_bias.to(dtype=affinity.dtype)

        valid_coarse = torch.ones(
            (b, 1, coarse_h, coarse_w),
            device=clip_logits_low.device,
            dtype=clip_logits_low.dtype,
        )
        valid_candidates = F.unfold(
            valid_coarse,
            kernel_size=self.candidate_kernel,
            padding=self.candidate_kernel // 2,
        )
        valid_candidates = valid_candidates.view(
            b, self.candidate_kernel * self.candidate_kernel, coarse_h, coarse_w
        )
        valid_candidates = F.interpolate(
            valid_candidates,
            size=target_size,
            mode="nearest",
        ).bool()
        affinity = affinity.masked_fill(~valid_candidates, torch.finfo(affinity.dtype).min)
        affinity = affinity.softmax(dim=1)

        # All class logits are transported together; no source class is filtered.
        logit_candidates = self._expand_coarse_candidates(clip_logits_low, target_size)
        guided_logits = (logit_candidates * affinity.unsqueeze(1)).sum(dim=2)

        margin_source = clip_margin_low.detach() if self.detach_margin else clip_margin_low
        margin = F.interpolate(
            margin_source.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty_gate = torch.sigmoid(
            (self.margin_threshold - margin) * self.margin_sharpness
        )
        strength = self.max_strength * torch.sigmoid(self.strength_logit)
        mix = strength * uncertainty_gate
        return baseline_logits + mix * (guided_logits - baseline_logits)


class SharedSpatialResidualDecoder(nn.Module):
    """Small class-agnostic spatial correction applied to refined logits."""

    def __init__(
        self,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        kernel_size: int = 3,
        max_strength: float = 0.2,
        max_delta: float = 0.5,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        self.margin_threshold = float(margin_threshold)
        self.margin_sharpness = float(margin_sharpness)
        self.kernel_size = int(kernel_size)
        self.max_strength = float(max_strength)
        self.max_delta = float(max_delta)

        # One shared spatial kernel is applied independently to every class.
        self.shared_conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False)
        nn.init.constant_(self.shared_conv.weight, 1.0 / (kernel_size * kernel_size))
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor, clip_margin_low: torch.Tensor):
        b, num_classes, h, w = logits.shape
        margin = F.interpolate(
            clip_margin_low.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        low_margin_gate = torch.sigmoid(
            (self.margin_threshold - margin) * self.margin_sharpness
        )

        logits_per_class = logits.reshape(b * num_classes, 1, h, w)
        pad = self.kernel_size // 2
        padded = F.pad(logits_per_class, (pad, pad, pad, pad), mode="replicate")
        spatial_logits = self.shared_conv(padded)
        delta = spatial_logits - logits_per_class
        delta = self.max_delta * torch.tanh(delta / max(self.max_delta, 1e-6))
        delta = delta.reshape(b, num_classes, h, w)

        strength = self.max_strength * torch.tanh(self.strength)
        return logits + strength * low_margin_gate * delta


class MMOVPredictorV2(nn.Module):
    """Raw CLIP semantic logits refined by margin-gated MASam spatial affinity."""

    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_clip_guidance_dims: list,
        decoder_clip_guidance_proj_dims: list,
        decoder_dino_guidance_dims: list,
        decoder_dino_guidance_proj_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        use_clip_corr: bool,
        use_dino_corr: bool,
        fusion_type: str,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        affinity_kernel: int = 3,
        affinity_temperature: float = 0.07,
        max_guidance_channels: int = 32,
        refinement_mode: str = "local_affinity",
        guided_kernel: int = 3,
        guided_temperature: float = 0.07,
        guided_spatial_sigma: float = 1.0,
        guided_max_strength: float = 1.0,
        guided_initial_strength: float = 0.01,
        guided_detach_margin: bool = True,
        use_spatial_decoder: bool = False,
        decoder_kernel: int = 3,
        decoder_max_strength: float = 0.2,
        decoder_max_delta: float = 0.5,
    ):
        super().__init__()
        with open(train_class_json, "r", encoding="utf-8") as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, "r", encoding="utf-8") as f_in:
            self.test_class_texts = json.load(f_in)
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        path_to_remote_clip = "/media/zpp2/PHDD/RemoteCLIP/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
        if clip_pretrained in ("ViT-G", "ViT-H"):
            name, pretrain = (
                ("ViT-H-14", "laion2b_s32b_b79k")
                if clip_pretrained == "ViT-H"
                else ("ViT-bigG-14", "laion2b_s39b_b160k")
            )
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=device,
                force_image_size=336,
            )
            self.tokenizer = open_clip.get_tokenizer(name)
        elif clip_pretrained == "RemoteCLIP-ViT-B-32":
            clip_model, clip_preprocess = clip.load(
                "ViT-B/32",
                device=device,
                jit=False,
                prompt_depth=prompt_depth,
                prompt_length=prompt_length,
            )
            ckpt = torch.load(path_to_remote_clip)
            mapped_state_dict = {}
            for key, value in ckpt.items():
                if "in_proj_weight" in key:
                    qkv = torch.chunk(value, 3, dim=0)
                    base_key = key.replace("in_proj_weight", "")
                    mapped_state_dict[base_key + "q_proj_weight"] = qkv[0]
                    mapped_state_dict[base_key + "k_proj_weight"] = qkv[1]
                    mapped_state_dict[base_key + "v_proj_weight"] = qkv[2]
                else:
                    mapped_state_dict[key] = value
            clip_model.load_state_dict(mapped_state_dict, strict=True)
        else:
            clip_model, clip_preprocess = clip.load(
                clip_pretrained,
                device=device,
                jit=False,
                prompt_depth=prompt_depth,
                prompt_length=prompt_length,
            )

        if prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif prompt_ensemble_type in ("rs", "remote_sensing"):
            prompt_templates = imagenet_templates.RS_TEMPLATES
        elif prompt_ensemble_type in ("rs_all", "remote_sensing_all"):
            prompt_templates = imagenet_templates.RS_ALL_TEMPLATES
        elif prompt_ensemble_type == "single":
            prompt_templates = ["A photo of a {} in the scene"]
        else:
            raise NotImplementedError

        self.prompt_templates = prompt_templates
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        refinement_mode = str(refinement_mode).lower()
        if refinement_mode == "local_affinity":
            self.refiner = MarginGatedAffinityRefiner(
                margin_threshold=margin_threshold,
                margin_sharpness=margin_sharpness,
                affinity_kernel=affinity_kernel,
                affinity_temperature=affinity_temperature,
                max_guidance_channels=max_guidance_channels,
            )
        elif refinement_mode == "guided_upsampling":
            self.refiner = MASamGuidedSemanticUpsampler(
                margin_threshold=margin_threshold,
                margin_sharpness=margin_sharpness,
                candidate_kernel=guided_kernel,
                affinity_temperature=guided_temperature,
                spatial_sigma=guided_spatial_sigma,
                max_guidance_channels=max_guidance_channels,
                max_strength=guided_max_strength,
                initial_strength=guided_initial_strength,
                detach_margin=guided_detach_margin,
            )
        elif refinement_mode == "multiscale_guided_upsampling":
            self.refiner = MultiScaleMASamGuidedSemanticUpsampler(
                margin_threshold=margin_threshold,
                margin_sharpness=margin_sharpness,
                candidate_kernel=guided_kernel,
                affinity_temperature=guided_temperature,
                spatial_sigma=guided_spatial_sigma,
                max_guidance_channels=max_guidance_channels,
                max_strength=guided_max_strength,
                initial_strength=guided_initial_strength,
                detach_margin=guided_detach_margin,
            )
        else:
            raise ValueError(
                f"Unknown refinement_mode: {refinement_mode}. "
                "Available: local_affinity/guided_upsampling/"
                "multiscale_guided_upsampling."
            )
        self.spatial_decoder = (
            SharedSpatialResidualDecoder(
                margin_threshold=margin_threshold,
                margin_sharpness=margin_sharpness,
                kernel_size=decoder_kernel,
                max_strength=decoder_max_strength,
                max_delta=decoder_max_delta,
            )
            if use_spatial_decoder
            else None
        )
        self.cache = None
        self.cache_classnames = None

    def train(self, mode: bool = True):
        """Switch mode and invalidate text features after parameter updates.

        Text embeddings are cached during evaluation, but the CLIP text
        encoder can be fine-tuned during training.  Keeping an eval cache
        across a subsequent training epoch would therefore evaluate the new
        image encoder with stale text embeddings.  Clearing the cache when
        entering either mode keeps evaluation tied to the current weights.
        """
        self.cache = None
        self.cache_classnames = None
        return super().train(mode)

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)
        return torch.einsum("bchw,btpc->bpthw", img_feats, text_feats)

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ", " in classname:
                classname_splits = classname.split(", ")
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        return torch.stack(zeroshot_weights, dim=1).cuda()

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        classnames_key = tuple(classnames)
        if self.cache is not None and not self.training and self.cache_classnames == classnames_key:
            return self.cache

        class_embeddings = []
        for classname in classnames:
            if ", " in classname:
                classname_splits = classname.split(", ")
                texts = []
                for template in templates:
                    texts.extend(template.format(cls_split) for cls_split in classname_splits)
                alias_count = len(classname_splits)
            else:
                texts = [template.format(classname) for template in templates]
                alias_count = 1
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            embeds = clip_model.encode_text(texts, prompt)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            if alias_count > 1:
                embeds = embeds.reshape(len(templates), alias_count, -1).mean(dim=1)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            class_embeddings.append(embeds)

        class_embeddings = torch.stack(class_embeddings, dim=0)
        if not self.training:
            self.cache = class_embeddings
            self.cache_classnames = classnames_key
        return class_embeddings

    def forward(self, x, dino_feat, vis_guidance, dino_guidance, prompt=None, gt_cls=None):
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        text = text.repeat(x.shape[0], 1, 1, 1)

        clip_corr = self.correlation(x, text)
        clip_corr = clip_corr.mean(dim=1)
        top2 = clip_corr.topk(k=min(2, clip_corr.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            margin = torch.ones_like(top2[:, 0])
        else:
            margin = top2[:, 0] - top2[:, 1]

        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is not None:
            scale = logit_scale.exp().clamp(max=100.0).to(clip_corr.dtype)
            clip_logits = clip_corr * scale
        else:
            clip_logits = clip_corr

        guidance = None
        if dino_guidance is not None and len(dino_guidance) > 0:
            guidance = (
                dino_guidance
                if getattr(self.refiner, "expects_multiscale_guidance", False)
                else dino_guidance[-1]
            )
        refined_logits = self.refiner(clip_logits, margin, guidance)
        if self.spatial_decoder is not None:
            refined_logits = self.spatial_decoder(refined_logits, margin)
        return refined_logits
