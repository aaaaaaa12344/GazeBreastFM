from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def infer_patch_grid(patch_count: int) -> tuple[int, int]:
    if patch_count <= 0:
        raise ValueError("patch_count must be positive.")

    grid_size = int(round(patch_count ** 0.5))
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count must form a square grid, got {patch_count}.")
    return grid_size, grid_size


def reshape_teacher_tokens(
    array: np.ndarray,
    image_id: str,
    teacher_path: Path | None = None,
) -> torch.Tensor:
    squeezed = np.asarray(array, dtype=np.float32).squeeze()
    if squeezed.ndim == 2:
        tokens = squeezed
    elif squeezed.ndim == 3 and squeezed.shape[0] == 1:
        tokens = squeezed[0]
    elif squeezed.ndim == 3:
        height, width, dim = squeezed.shape
        tokens = squeezed.reshape(height * width, dim)
    else:
        location = str(teacher_path) if teacher_path is not None else "<memory>"
        raise ValueError(
            f"Unsupported teacher latent shape for {image_id} at {location}: {array.shape}"
        )

    if tokens.ndim != 2:
        location = str(teacher_path) if teacher_path is not None else "<memory>"
        raise ValueError(
            f"Teacher latent for {image_id} at {location} could not be reshaped to [N, D]."
        )
    return torch.from_numpy(np.asarray(tokens, dtype=np.float32))


def resize_teacher_tokens_2d(
    tokens: torch.Tensor,
    expected_patch_count: int,
    image_id: str,
    teacher_path: Path | None = None,
) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [N, D], got {tuple(tokens.shape)}")

    patch_count, latent_dim = tokens.shape
    if patch_count == expected_patch_count:
        return tokens.to(dtype=torch.float32).contiguous()

    source_grid = infer_patch_grid(patch_count)
    target_grid = infer_patch_grid(expected_patch_count)
    spatial = tokens.view(source_grid[0], source_grid[1], latent_dim)
    spatial = spatial.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        spatial,
        size=target_grid,
        mode="bilinear",
        align_corners=False,
    )
    flattened = resized.squeeze(0).permute(1, 2, 0).reshape(expected_patch_count, latent_dim)
    return flattened.to(dtype=torch.float32).contiguous()


def adapt_teacher_latent_dim(
    tokens: torch.Tensor,
    latent_dim: int,
    image_id: str,
    teacher_path: Path | None = None,
) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape [N, D], got {tuple(tokens.shape)}")

    current_dim = int(tokens.shape[1])
    location = str(teacher_path) if teacher_path is not None else "<memory>"
    if current_dim == latent_dim:
        return tokens.to(dtype=torch.float32).contiguous()

    if current_dim < latent_dim:
        padded = torch.zeros(
            (tokens.shape[0], latent_dim),
            dtype=torch.float32,
        )
        padded[:, :current_dim] = tokens.to(dtype=torch.float32)
        warnings.warn(
            f"Teacher latent dim for {image_id} at {location} is {current_dim}, padded to {latent_dim}.",
            stacklevel=2,
        )
        return padded.contiguous()

    warnings.warn(
        f"Teacher latent dim for {image_id} at {location} is {current_dim}, truncated to {latent_dim}.",
        stacklevel=2,
    )
    return tokens[:, :latent_dim].to(dtype=torch.float32).contiguous()
