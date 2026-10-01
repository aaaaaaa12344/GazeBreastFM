from __future__ import annotations

import torch

from breast_pretrain.train.masked_latent_smoke import compute_masked_latent_loss


def masked_teacher_latent_reconstruction_loss(
    *,
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_weights: torch.Tensor,
) -> torch.Tensor:
    return compute_masked_latent_loss(
        student_latent=student_latent,
        teacher_latent=teacher_latent,
        patch_mask=patch_mask,
        patch_weights=patch_weights,
    )
