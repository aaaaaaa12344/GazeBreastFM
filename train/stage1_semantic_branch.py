from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from breast_pretrain.models.concept_heads import ClinicalConceptHeads


class ConceptKeyModuleDict(nn.ModuleDict):
    """ModuleDict with canonical concept-ID lookup and safe storage keys."""

    @staticmethod
    def encode_concept_id(concept_id: str) -> str:
        value = str(concept_id)
        return "c_" + value.encode("utf-8").hex()

    @staticmethod
    def decode_module_key(module_key: str) -> str:
        key = str(module_key)
        if not key.startswith("c_"):
            raise ValueError(f"invalid concept module key: {module_key!r}")
        try:
            return bytes.fromhex(key[2:]).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid concept module key: {module_key!r}") from exc

    def __init__(self, modules: dict[str, nn.Module] | None = None) -> None:
        self.concept_to_module_key: dict[str, str] = {}
        self.module_key_to_concept: dict[str, str] = {}
        encoded_modules: dict[str, nn.Module] = {}
        for concept_id, module in (modules or {}).items():
            concept_id = str(concept_id)
            module_key = self.encode_concept_id(concept_id)
            if module_key in self.module_key_to_concept:
                raise ValueError(f"concept module key collision: {concept_id!r}")
            if self.decode_module_key(module_key) != concept_id:
                raise ValueError(f"concept module key round-trip failed: {concept_id!r}")
            self.concept_to_module_key[concept_id] = module_key
            self.module_key_to_concept[module_key] = concept_id
            encoded_modules[module_key] = module
        super().__init__(encoded_modules)

    def _resolve_key(self, key: str) -> str:
        return self.concept_to_module_key.get(str(key), str(key))

    def __getitem__(self, key: str) -> nn.Module:
        return super().__getitem__(self._resolve_key(key))

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._resolve_key(str(key)))

    def get(self, key: str, default: nn.Module | None = None) -> nn.Module | None:
        return super().get(self._resolve_key(key), default)


class Stage1SemanticBranch(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        align_dim: int,
        active_concept_heads: tuple[str, ...] = ("view", "laterality"),
        concept_head_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.align_dim = int(align_dim)
        self.visual_projector = nn.Linear(visual_dim, align_dim)
        self.case_projector = nn.Linear(text_dim, align_dim)
        self.active_concept_heads = tuple(active_concept_heads)
        self.concept_heads = ClinicalConceptHeads(
            align_dim,
            self.active_concept_heads,
            output_dims=concept_head_output_dims,
        )
        self.graph_prototype_projectors = ConceptKeyModuleDict()

    def configure_graph_prototype_projectors(
        self,
        graph_dim: int,
        prototype_head_names: tuple[str, ...] | None = None,
    ) -> None:
        self.graph_prototype_projectors = ConceptKeyModuleDict()
        graph_dim = int(graph_dim)
        if graph_dim == self.align_dim:
            return
        head_names = tuple(dict.fromkeys(prototype_head_names or self.active_concept_heads))
        self.graph_prototype_projectors = ConceptKeyModuleDict(
            {head_name: nn.Linear(self.align_dim, graph_dim) for head_name in head_names}
        )
        reference = next(self.parameters(), None)
        if reference is not None:
            self.graph_prototype_projectors.to(
                device=reference.device,
                dtype=reference.dtype,
            )

    def project_image(self, features: torch.Tensor) -> torch.Tensor:
        # Keep this small alignment path outside autocast so its FP32 gradient
        # does not overflow when the contrastive objective is GradScaler-scaled.
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = self.visual_projector(features.float())
            return F.normalize(projected, dim=-1, eps=1e-6)

    def project_case(self, features: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = self.case_projector(features.float())
            return F.normalize(projected, dim=-1, eps=1e-6)

    def diagnostic_projection_tensors(
        self,
        global_image_feature: torch.Tensor,
        visible_image_feature: torch.Tensor,
        case_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Expose semantic pre/post-normalization values for diagnostic-only replay."""
        with torch.autocast(device_type=global_image_feature.device.type, enabled=False):
            global_pre = self.visual_projector(global_image_feature.float())
            visible_pre = self.visual_projector(visible_image_feature.float())
            case_pre = self.case_projector(case_feature.float())
            return {
                "global_visual_projector_pre_normalize": global_pre,
                "global_visual_embedding_post_normalize": F.normalize(global_pre, dim=-1, eps=1e-6),
                "visible_visual_projector_pre_normalize": visible_pre,
                "visible_visual_embedding_post_normalize": F.normalize(visible_pre, dim=-1, eps=1e-6),
                "text_case_projector_pre_normalize": case_pre,
                "text_case_embedding_post_normalize": F.normalize(case_pre, dim=-1, eps=1e-6),
            }

    def forward(
        self,
        global_image_feature: torch.Tensor,
        visible_image_feature: torch.Tensor,
        case_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.build_outputs(
            global_image_feature=global_image_feature,
            visible_image_feature=visible_image_feature,
            case_feature=case_feature,
        )

    def build_outputs(
        self,
        global_image_feature: torch.Tensor,
        visible_image_feature: torch.Tensor,
        case_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_image_embedding = self.project_image(global_image_feature)
        visible_image_embedding = self.project_image(visible_image_feature)
        case_embedding = self.project_case(case_feature)
        concept_feature = 0.5 * (global_image_embedding + visible_image_embedding)
        concept_logits = {
            head_name: logits
            for head_name, logits in self.concept_heads(concept_feature).items()
        }
        outputs = {
            "global_image_embedding": global_image_embedding,
            "visible_image_embedding": visible_image_embedding,
            "case_embedding": case_embedding,
            "concept_feature": concept_feature,
            "concept_logits": concept_logits,
        }
        if "view" in concept_logits:
            outputs["view_logits"] = concept_logits["view"]
        if "laterality" in concept_logits:
            outputs["laterality_logits"] = concept_logits["laterality"]
        return {
            **outputs,
        }


def build_visible_region_weights(
    attention_tokens: torch.Tensor,
    high_conf_tokens: torch.Tensor,
    patch_mask: torch.Tensor,
    high_conf_weight_alpha: float,
) -> torch.Tensor:
    if attention_tokens.shape != high_conf_tokens.shape or attention_tokens.shape != patch_mask.shape:
        raise ValueError(
            "attention_tokens, high_conf_tokens, and patch_mask must share the same shape, got "
            f"{tuple(attention_tokens.shape)}, {tuple(high_conf_tokens.shape)}, and {tuple(patch_mask.shape)}."
        )

    visible_mask = (~patch_mask).to(dtype=attention_tokens.dtype)
    weights = attention_tokens.clamp_min(0.0) * visible_mask
    weights = weights * (1.0 + float(high_conf_weight_alpha) * high_conf_tokens.clamp(0.0, 1.0))
    zero_rows = weights.sum(dim=1, keepdim=True) <= 0.0
    if torch.any(zero_rows):
        fallback = visible_mask
        fallback_zero_rows = fallback.sum(dim=1, keepdim=True) <= 0.0
        if torch.any(fallback_zero_rows):
            fallback = torch.ones_like(fallback)
        weights = torch.where(zero_rows, fallback, weights)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
