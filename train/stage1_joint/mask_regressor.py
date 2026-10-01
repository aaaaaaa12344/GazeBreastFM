from __future__ import annotations

import torch
from torch import nn


class MaskRegressor(nn.Module):
    """Lightweight context-aware decoder for masked patch prediction.

    Takes context (visible) patch tokens and predicts the original features
    at masked positions. Simple architecture: LayerNorm -> Linear -> GELU -> Linear.

    The regressor receives ALL patches (both visible and masked positions) as
    context, but the reconstruction loss is only computed at masked positions.
    """

    def __init__(self, dim: int = 768, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        hidden_dim = max(64, int(dim * hidden_ratio))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self._output_dim = dim

    def forward(self, context_patch_tokens: torch.Tensor) -> torch.Tensor:
        """Predict patch tokens from context.

        Args:
            context_patch_tokens: [B, N, D] patch tokens from masked-view encoder

        Returns:
            predicted_patch_tokens: [B, N, D] predicted tokens at ALL positions
        """
        x = self.norm(context_patch_tokens)
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x


__all__ = ["MaskRegressor"]
