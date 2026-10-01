from __future__ import annotations

import torch

from breast_pretrain.semantics import (
    compute_concept_consistency_loss,
    compute_prior_head_consistency_loss,
)


def concept_consistency_regularization_loss(
    *,
    image_embeddings: torch.Tensor,
    semantic_targets: torch.Tensor,
    prior_records: list[object],
    concept_logits: dict[str, torch.Tensor],
    head_weights: dict[str, float],
) -> torch.Tensor:
    return compute_concept_consistency_loss(
        image_embeddings=image_embeddings,
        semantic_targets=semantic_targets,
        prior_records=prior_records,
    ) + compute_prior_head_consistency_loss(
        concept_logits=concept_logits,
        prior_records=prior_records,
        head_weights=head_weights,
    )
