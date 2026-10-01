from __future__ import annotations

import torch
from torch import nn

from breast_pretrain.data.transforms.stage1_transform_spec import ImageSize, normalize_image_size


class MinimalPatchStudentEncoder(nn.Module):
    """Lightweight patch encoder for masked latent smoke training."""

    def __init__(
        self,
        image_size: ImageSize = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        image_height, image_width = normalize_image_size(image_size)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = (int(image_height), int(image_width))
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.grid_height = self.image_height // self.patch_size
        self.grid_width = self.image_width // self.patch_size
        self.grid_size = self.grid_height if self.grid_height == self.grid_width else None
        self.patch_grid = [self.grid_height, self.grid_width]
        self.num_patches = self.grid_height * self.grid_width

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.latent_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.pre_norm = nn.LayerNorm(self.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.post_norm = nn.LayerNorm(self.latent_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.latent_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _encode_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [batch, channels, height, width], got {tuple(image.shape)}"
            )

        tokens = self.patch_embed(image)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.pre_norm(tokens + self.position_embedding)
        tokens = tokens + self.mlp(tokens)
        return self.post_norm(tokens)

    def forward(
        self,
        image: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        patch_tokens = self._encode_patch_tokens(image)
        if not return_dict:
            return patch_tokens

        global_image_feature = patch_tokens.mean(dim=1)
        return {
            "patch_tokens": patch_tokens,
            "global_image_feature": global_image_feature,
            # The minimal smoke encoder predicts masked targets directly from patch tokens.
            "masked_prediction": patch_tokens,
        }
