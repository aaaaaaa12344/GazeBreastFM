from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from breast_pretrain.models.local_high_conf_branch import LocalHighConfBranchOutput


@dataclass(frozen=True)
class LocalBranchStepResult:
    local_loss: torch.Tensor
    metrics: dict[str, object]
    clean_output: LocalHighConfBranchOutput | None = None
    masked_output: LocalHighConfBranchOutput | None = None


def expand_patch_scores_to_spatial(
    patch_scores: torch.Tensor,
    patch_grid: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    batch_size, num_patches = patch_scores.shape[:2]
    grid_h, grid_w = int(patch_grid[0]), int(patch_grid[1])
    if int(num_patches) != grid_h * grid_w:
        raise ValueError(
            f"patch_scores has {int(num_patches)} patches but grid is {grid_h}x{grid_w}."
        )
    patch_scores_2d = patch_scores.reshape(batch_size, grid_h, grid_w)
    spatial = patch_scores_2d.repeat_interleave(int(patch_size), dim=1).repeat_interleave(
        int(patch_size), dim=2
    )
    return spatial.unsqueeze(1)


def run_local_high_conf_branch_step(
    *,
    local_branch: nn.Module | None,
    clean_image: torch.Tensor,
    masked_image: torch.Tensor,
    high_conf_patch_scores: torch.Tensor,
    patch_gaze_scores: torch.Tensor,
    patch_grid: tuple[int, int],
    patch_size: int,
    loss_weight: float,
) -> LocalBranchStepResult:
    zero_loss = clean_image.new_zeros(())
    if local_branch is None or float(loss_weight) <= 0.0:
        return LocalBranchStepResult(
            local_loss=zero_loss,
            metrics={
                "local_branch_used": False,
                "local_token_count": 0,
                "local_loss": 0.0,
                "local_high_conf_coverage": 0.0,
            },
        )

    high_conf_spatial = expand_patch_scores_to_spatial(
        high_conf_patch_scores.to(device=clean_image.device, dtype=clean_image.dtype),
        patch_grid,
        patch_size,
    )
    gaze_spatial = expand_patch_scores_to_spatial(
        patch_gaze_scores.to(device=clean_image.device, dtype=clean_image.dtype),
        patch_grid,
        patch_size,
    )
    high_conf_mask = (high_conf_spatial > 0.0).to(dtype=clean_image.dtype)

    clean_output = local_branch(
        image=clean_image,
        high_conf_mask=high_conf_mask,
        heatmap=gaze_spatial,
    )
    masked_output = local_branch(
        image=masked_image,
        high_conf_mask=high_conf_mask,
        heatmap=gaze_spatial,
    )
    per_sample_loss = (
        masked_output.local_patch_tokens - clean_output.local_patch_tokens.detach()
    ).pow(2).mean(dim=(1, 2))
    valid = clean_output.roi.roi_valid.to(device=per_sample_loss.device, dtype=torch.bool)
    if bool(valid.any()):
        local_loss = per_sample_loss[valid].mean()
    else:
        local_loss = zero_loss

    local_token_count = int(masked_output.local_patch_tokens.shape[1])
    coverage = float((high_conf_patch_scores > 0.0).to(dtype=torch.float32).mean().item())
    metrics = {
        "local_branch_used": True,
        "local_token_count": local_token_count,
        "local_loss": float(local_loss.detach().item()),
        "local_high_conf_coverage": coverage,
        "local_roi_valid_count": int(valid.sum().item()),
        "local_roi_fallback_reason": tuple(clean_output.roi.fallback_reason),
    }
    return LocalBranchStepResult(
        local_loss=local_loss,
        metrics=metrics,
        clean_output=clean_output,
        masked_output=masked_output,
    )


__all__ = [
    "LocalBranchStepResult",
    "expand_patch_scores_to_spatial",
    "run_local_high_conf_branch_step",
]
