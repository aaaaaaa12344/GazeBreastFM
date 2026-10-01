from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SemanticProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.proj(features)
        # Keep the normalization numerically stable under FP16 autocast. The
        # projection geometry and unit-norm contract are unchanged.
        return F.normalize(projected.float(), dim=-1, eps=1e-6)
