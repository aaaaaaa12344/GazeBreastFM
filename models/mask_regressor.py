from __future__ import annotations

import torch
from torch import nn


class MaskRegressor(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(patch_tokens)
