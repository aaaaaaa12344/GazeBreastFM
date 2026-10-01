from __future__ import annotations

import torch
from torch.nn import functional as F


def visible_latent_alignment_loss(
    visible_image_embedding: torch.Tensor,
    case_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = (visible_image_embedding @ case_embedding.transpose(0, 1)) / float(temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
    )
