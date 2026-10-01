from __future__ import annotations

from typing import Any

import torch
from torch import nn

from breast_pretrain.models.concept_heads import ClinicalConceptHeads
from breast_pretrain.models.mask_regressor import MaskRegressor
from breast_pretrain.models.semantic_projector import SemanticProjector
from breast_pretrain.models.minimal_student_encoder import MinimalPatchStudentEncoder


class JointPretrainModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        latent_dim: int,
        text_dim: int,
        align_dim: int,
        active_concept_heads: tuple[str, ...],
        vision_encoder: nn.Module | None = None,
        modality_vocab: tuple[str, ...] = ("mammography", "mri", "ultrasound"),
        modality_embedding_dim: int = 32,
        modality_embedding_strategy: str = "add_to_global",
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder or MinimalPatchStudentEncoder(
            image_size=image_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
        )
        self.mask_regressor = MaskRegressor(latent_dim)
        self.global_projector = SemanticProjector(latent_dim, align_dim)
        self.visible_projector = SemanticProjector(latent_dim, align_dim)
        self.case_projector = SemanticProjector(text_dim, align_dim)
        self.concept_heads = ClinicalConceptHeads(align_dim, active_concept_heads)

        self.modality_embedding_enabled = bool(modality_vocab) and len(modality_vocab) > 1
        self.modality_vocab = tuple(modality_vocab)
        self.modality_embedding_strategy = str(modality_embedding_strategy)
        self._modality_to_idx = {m: i for i, m in enumerate(self.modality_vocab)} if self.modality_embedding_enabled else {}
        if self.modality_embedding_enabled:
            self.modality_embed = nn.Embedding(len(self.modality_vocab), int(modality_embedding_dim))
            self.modality_proj_global = nn.Linear(int(modality_embedding_dim), latent_dim) if modality_embedding_strategy in {"add_to_global", "add_to_both"} else None
            self.modality_proj_visible = nn.Linear(int(modality_embedding_dim), latent_dim) if modality_embedding_strategy in {"add_to_both"} else None
        else:
            self.modality_embed = None
            self.modality_proj_global = None
            self.modality_proj_visible = None

    @property
    def num_patches(self) -> int:
        return int(self.vision_encoder.num_patches)

    def _visible_feature(
        self,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None,
        patch_gaze_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if patch_gaze_weight is not None:
            weights = patch_gaze_weight.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            if weights.ndim == 1:
                weights = weights.unsqueeze(0).expand(patch_tokens.shape[0], -1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
        if patch_mask is not None:
            visible = (~patch_mask).to(dtype=patch_tokens.dtype)
            visible = visible / visible.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (visible.unsqueeze(-1) * patch_tokens).sum(dim=1)
        return patch_tokens.mean(dim=1)

    def forward(
        self,
        *,
        image: torch.Tensor,
        attention_map: torch.Tensor | None = None,
        aggregated_attention_map: torch.Tensor | None = None,
        high_conf_mask: torch.Tensor | None = None,
        patch_gaze_weight: torch.Tensor | None = None,
        teacher_latent: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        prompt_embedding: torch.Tensor | None = None,
        semantic_soft_label_matrix: torch.Tensor | None = None,
        clinical_concept_labels: dict[str, torch.Tensor] | None = None,
        clinical_graph_prior: Any = None,
        patch_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        encoded = self.vision_encoder(image, return_dict=True)
        patch_tokens = encoded["patch_tokens"]
        global_image_feature = encoded["global_image_feature"]

        if self.modality_embedding_enabled and modality_ids is not None:
            mod_emb = self.modality_embed(modality_ids.to(dtype=torch.long, device=global_image_feature.device))
            if self.modality_embedding_strategy in {"add_to_global", "add_to_both"} and self.modality_proj_global is not None:
                global_image_feature = global_image_feature + self.modality_proj_global(mod_emb)
            if self.modality_embedding_strategy in {"add_to_both"} and self.modality_proj_visible is not None:
                pass
        elif self.modality_embedding_enabled and modality_ids is None:
            raise ValueError(
                "JointPretrainModel.modality_embedding is enabled but modality_ids was not provided. "
                "Stage1JointBatch must carry modality_ids when modality_embedding is active."
            )

        reconstructed_patch_tokens = self.mask_regressor(patch_tokens)
        case_feature = prompt_embedding if prompt_embedding is not None else text_embedding
        if case_feature is None:
            raise ValueError("JointPretrainModel.forward requires text_embedding or prompt_embedding.")
        visible_image_feature = self._visible_feature(
            patch_tokens=patch_tokens,
            patch_mask=patch_mask,
            patch_gaze_weight=patch_gaze_weight,
        )

        if self.modality_embedding_enabled and modality_ids is not None and self.modality_embedding_strategy in {"add_to_both"} and self.modality_proj_visible is not None:
            visible_image_feature = visible_image_feature + self.modality_proj_visible(
                self.modality_embed(modality_ids.to(dtype=torch.long, device=visible_image_feature.device))
            )

        global_image_embedding = self.global_projector(global_image_feature)
        visible_image_embedding = self.visible_projector(visible_image_feature)
        case_embedding = self.case_projector(case_feature)
        concept_feature = 0.5 * (global_image_embedding + visible_image_embedding)
        concept_logits = self.concept_heads(concept_feature)
        return {
            "patch_tokens": patch_tokens,
            "masked_prediction": reconstructed_patch_tokens,
            "global_image_feature": global_image_feature,
            "visible_image_feature": visible_image_feature,
            "global_image_embedding": global_image_embedding,
            "visible_image_embedding": visible_image_embedding,
            "case_embedding": case_embedding,
            "concept_feature": concept_feature,
            "concept_logits": concept_logits,
            "teacher_latent": teacher_latent,
            "attention_map": attention_map,
            "aggregated_attention_map": aggregated_attention_map,
            "high_conf_mask": high_conf_mask,
            "semantic_soft_label_matrix": semantic_soft_label_matrix,
            "clinical_concept_labels": clinical_concept_labels,
            "clinical_graph_prior": clinical_graph_prior,
        }
