from __future__ import annotations

import torch
from torch.nn import functional as F


def semantic_soft_contrastive_loss(
    *,
    image_embeddings: torch.Tensor,
    case_embeddings: torch.Tensor,
    semantic_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if image_embeddings.shape[0] < 2:
        return image_embeddings.new_zeros(())
    targets = semantic_targets.to(device=image_embeddings.device, dtype=image_embeddings.dtype)
    targets = torch.maximum(
        targets.clamp(0.0, 1.0),
        torch.eye(targets.shape[0], device=targets.device, dtype=targets.dtype),
    )
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-6)
    logits = (image_embeddings @ case_embeddings.transpose(0, 1)) / float(temperature)
    image_to_case = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    case_to_image = -(
        targets.transpose(0, 1) * F.log_softmax(logits.transpose(0, 1), dim=1)
    ).sum(dim=1).mean()
    return 0.5 * (image_to_case + case_to_image)
