from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from breast_pretrain.gaze.local_roi import LocalRoiOutput, build_local_high_conf_roi_batch


@dataclass(frozen=True)
class LocalHighConfBranchOutput:
    local_patch_tokens: torch.Tensor
    roi: LocalRoiOutput
    effective_stride: int
    local_patch_grid: tuple[int, int]
    local_branch_scaffold_ready: bool
    local_branch_training_ready: bool
    forbid_global_branch_replacement: bool
    forbid_graph_node_region_alignment: bool


class LocalHighConfBranch(nn.Module):
    def __init__(
        self,
        *,
        local_input_size: tuple[int, int] = (512, 512),
        effective_stride: int = 8,
        in_channels: int = 3,
        output_dim: int = 128,
        roi_margin: int = 32,
    ) -> None:
        super().__init__()
        self.local_input_size = (int(local_input_size[0]), int(local_input_size[1]))
        self.effective_stride = int(effective_stride)
        self.output_dim = int(output_dim)
        self.roi_margin = int(roi_margin)
        if self.effective_stride != 8:
            raise ValueError("LocalHighConfBranch scaffold currently supports effective_stride=8 only.")
        if self.local_input_size[0] % self.effective_stride != 0 or self.local_input_size[1] % self.effective_stride != 0:
            raise ValueError("local_input_size must be divisible by effective_stride.")
        self.local_patch_grid = (
            self.local_input_size[0] // self.effective_stride,
            self.local_input_size[1] // self.effective_stride,
        )
        self.patch_embed = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=self.output_dim,
            kernel_size=self.effective_stride,
            stride=self.effective_stride,
        )
        self.norm = nn.LayerNorm(self.output_dim)
        self.local_branch_scaffold_ready = True
        self.local_branch_training_ready = False
        self.forbid_global_branch_replacement = True
        self.forbid_graph_node_region_alignment = True

    def forward(
        self,
        *,
        image: torch.Tensor,
        high_conf_mask: torch.Tensor,
        heatmap: torch.Tensor,
    ) -> LocalHighConfBranchOutput:
        roi = build_local_high_conf_roi_batch(
            image=image,
            high_conf_mask=high_conf_mask,
            heatmap=heatmap,
            margin=self.roi_margin,
            local_input_size=self.local_input_size,
        )
        tokens = self.patch_embed(roi.local_image).flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return LocalHighConfBranchOutput(
            local_patch_tokens=tokens,
            roi=roi,
            effective_stride=self.effective_stride,
            local_patch_grid=self.local_patch_grid,
            local_branch_scaffold_ready=self.local_branch_scaffold_ready,
            local_branch_training_ready=self.local_branch_training_ready,
            forbid_global_branch_replacement=self.forbid_global_branch_replacement,
            forbid_graph_node_region_alignment=self.forbid_graph_node_region_alignment,
        )


__all__ = [
    "LocalHighConfBranch",
    "LocalHighConfBranchOutput",
]
