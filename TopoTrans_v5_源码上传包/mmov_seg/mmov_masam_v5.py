"""MM-OVSeg-MASam V5 with margin-guided semantic graph refinement.

V5 keeps CLIP as the semantic source and uses MASam only to construct a
non-local relation matrix over the coarse semantic grid.  The relation matrix
propagates CLIP logits, while a single CLIP top-1/top-2 margin gate controls
where the residual correction is applied.  MAHR remains responsible for the
subsequent high-resolution spatial refinement.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmov_masam_v4 import MMOVMASamV4
from .modeling.transformer.MMOVPredictorV2 import (
    MultiScaleMASamGuidedSemanticUpsampler,
)


class MarginGuidedSemanticGraphRefiner(nn.Module):
    """Refine CLIP logits using MASam semantic relations only.

    The auxiliary features define the graph geometry; the graph values are
    always the original CLIP logits.  This preserves the CLIP image/text
    alignment while allowing uncertain semantic nodes to borrow evidence from
    MASam-related nodes.
    """

    def __init__(
        self,
        auxiliary_channels: int = 512 + 256,
        relation_dim: int = 64,
        temperature: float = 0.07,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        max_strength: float = 1.0,
        initial_strength: float = 0.1,
    ):
        super().__init__()
        if relation_dim <= 0:
            raise ValueError("relation_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if max_strength <= 0:
            raise ValueError("max_strength must be positive.")
        if not 0 < initial_strength < max_strength:
            raise ValueError("initial_strength must be between 0 and max_strength.")

        self.margin_threshold = float(margin_threshold)
        self.margin_sharpness = float(margin_sharpness)
        self.temperature = float(temperature)
        self.max_strength = float(max_strength)

        # The projection learns a relation space for MASam features without
        # changing CLIP features or text embeddings.
        self.relation_projection = nn.Sequential(
            nn.Conv2d(auxiliary_channels, relation_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, relation_dim),
            nn.GELU(),
        )

        initial_ratio = initial_strength / max_strength
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        self.strength_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def _semantic_guidance(self, dino_feat, dino_guidance):
        if dino_feat is None:
            return None

        target_size = dino_feat.shape[-2:]
        parts = [dino_feat]
        if isinstance(dino_guidance, (list, tuple)) and dino_guidance:
            # Use only the native 48x48 feature as semantic context.  The
            # native 96x96 feature remains exclusive to MAHR.
            mid = dino_guidance[0]
            mid = F.adaptive_avg_pool2d(mid, target_size)
            parts.append(mid)
        return torch.cat(parts, dim=1)

    def forward(self, clip_logits_low, clip_margin_low, dino_feat, dino_guidance):
        guidance = self._semantic_guidance(dino_feat, dino_guidance)
        if guidance is None:
            return clip_logits_low

        b, num_classes, height, width = clip_logits_low.shape
        relation = self.relation_projection(guidance)
        relation = relation.flatten(2).transpose(1, 2)
        relation = F.normalize(relation, dim=-1)

        # A dense semantic graph over 24x24 nodes.  This is non-local relation
        # reasoning, not fixed spatial-window smoothing.
        affinity = torch.bmm(relation, relation.transpose(1, 2))
        affinity = (affinity / self.temperature).softmax(dim=-1)

        logits_nodes = clip_logits_low.flatten(2).transpose(1, 2)
        graph_logits = torch.bmm(affinity, logits_nodes)
        graph_logits = graph_logits.transpose(1, 2).reshape(
            b, num_classes, height, width
        )

        # The only explicit gate: CLIP uncertainty controls the graph residual.
        margin = clip_margin_low.detach()
        uncertainty = torch.sigmoid(
            (self.margin_threshold - margin) * self.margin_sharpness
        ).unsqueeze(1)
        strength = self.max_strength * torch.sigmoid(self.strength_logit)
        return clip_logits_low + strength * uncertainty * (graph_logits - clip_logits_low)


class _V5PredictorAdapter(nn.Module):
    """Keep the production CLIP predictor and replace only its output path."""

    def __init__(
        self,
        production_predictor: nn.Module,
        relation_dim: int = 64,
        relation_temperature: float = 0.07,
        margin_threshold: float = 0.04,
        margin_sharpness: float = 40.0,
        relation_max_strength: float = 1.0,
        relation_initial_strength: float = 0.1,
        guided_kernel: int = 3,
        guided_temperature: float = 0.07,
        guided_spatial_sigma: float = 1.0,
        guided_max_strength: float = 1.0,
        guided_initial_strength: float = 0.01,
        guided_detach_margin: bool = True,
    ):
        super().__init__()
        self.production_predictor = production_predictor
        # The original v3 refiner is replaced by the isolated V5 path.
        self.production_predictor.refiner = nn.Identity()

        self.semantic_refiner = MarginGuidedSemanticGraphRefiner(
            auxiliary_channels=512 + 256,
            relation_dim=relation_dim,
            temperature=relation_temperature,
            margin_threshold=margin_threshold,
            margin_sharpness=margin_sharpness,
            max_strength=relation_max_strength,
            initial_strength=relation_initial_strength,
        )
        self.mahr = MultiScaleMASamGuidedSemanticUpsampler(
            margin_threshold=margin_threshold,
            margin_sharpness=margin_sharpness,
            candidate_kernel=guided_kernel,
            affinity_temperature=guided_temperature,
            spatial_sigma=guided_spatial_sigma,
            max_guidance_channels=32,
            max_strength=guided_max_strength,
            initial_strength=guided_initial_strength,
            detach_margin=guided_detach_margin,
        )

    @property
    def clip_model(self):
        return self.production_predictor.clip_model

    def _clip_logits_and_margin(self, x, gt_cls=None):
        base = self.production_predictor
        text = base.class_texts if self.training else base.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = base.get_text_embeds(text, base.prompt_templates, base.clip_model)
        text = text.repeat(x.shape[0], 1, 1, 1)

        clip_corr = base.correlation(x, text).mean(dim=1)
        top2 = clip_corr.topk(k=min(2, clip_corr.shape[1]), dim=1).values
        margin = (
            torch.ones_like(top2[:, 0])
            if top2.shape[1] == 1
            else top2[:, 0] - top2[:, 1]
        )

        logit_scale = getattr(base.clip_model, "logit_scale", None)
        if logit_scale is not None:
            scale = logit_scale.exp().clamp(max=100.0).to(clip_corr.dtype)
            clip_corr = clip_corr * scale
        return clip_corr, margin

    def forward(
        self,
        x,
        dino_feat,
        vis_guidance,
        dino_guidance,
        prompt=None,
        gt_cls=None,
    ):
        del vis_guidance, prompt
        clip_logits, margin = self._clip_logits_and_margin(x, gt_cls=gt_cls)
        semantic_logits = self.semantic_refiner(
            clip_logits,
            margin,
            dino_feat,
            dino_guidance,
        )
        return self.mahr(semantic_logits, margin, dino_guidance)


class MMOVMASamV5(MMOVMASamV4):
    """V5 = MG-SGR semantic refinement + V4 MAHR spatial refinement."""

    def __init__(
        self,
        *args,
        relation_dim: int = 64,
        relation_temperature: float = 0.07,
        relation_max_strength: float = 1.0,
        relation_initial_strength: float = 0.1,
        **kwargs,
    ):
        # Build the established v3/MAHR backbone first, then replace the
        # predictor output path inside this V5-only model instance.
        super().__init__(*args, predictor_version="v3", **kwargs)
        self.predictor = _V5PredictorAdapter(
            self.predictor,
            relation_dim=relation_dim,
            relation_temperature=relation_temperature,
            margin_threshold=float(kwargs.get("v2_margin_threshold", 0.04)),
            margin_sharpness=float(kwargs.get("v2_margin_sharpness", 40.0)),
            relation_max_strength=relation_max_strength,
            relation_initial_strength=relation_initial_strength,
            guided_kernel=int(kwargs.get("v2_guided_kernel", 3)),
            guided_temperature=float(kwargs.get("v2_guided_temperature", 0.07)),
            guided_spatial_sigma=float(kwargs.get("v2_guided_spatial_sigma", 1.0)),
            guided_max_strength=float(kwargs.get("v2_guided_max_strength", 1.0)),
            guided_initial_strength=float(kwargs.get("v2_guided_initial_strength", 0.01)),
            guided_detach_margin=bool(kwargs.get("v2_guided_detach_margin", True)),
        )
        self.v5_semantic_refinement = "margin_guided_masam_semantic_graph"

