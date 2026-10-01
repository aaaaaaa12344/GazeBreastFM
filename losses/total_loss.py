from __future__ import annotations

import torch


def total_joint_pretrain_loss(
    *,
    reconstruction_loss: torch.Tensor,
    global_align_loss: torch.Tensor,
    visible_align_loss: torch.Tensor,
    semantic_soft_loss: torch.Tensor,
    concept_loss: torch.Tensor,
    concept_consistency_loss: torch.Tensor,
    reconstruction_weight: float,
    global_align_weight: float,
    visible_align_weight: float,
    semantic_soft_weight: float,
    concept_loss_weight: float,
    concept_consistency_weight: float,
) -> torch.Tensor:
    return (
        float(reconstruction_weight) * reconstruction_loss
        + float(global_align_weight) * global_align_loss
        + float(visible_align_weight) * visible_align_loss
        + float(semantic_soft_weight) * semantic_soft_loss
        + float(concept_loss_weight) * concept_loss
        + float(concept_consistency_weight) * concept_consistency_loss
    )
