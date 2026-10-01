from __future__ import annotations

import torch

from breast_pretrain.models.local_high_conf_branch import LocalHighConfBranch


def attach_local_high_conf_branch(
    config: object,
    model: torch.nn.Module,
) -> dict[str, object]:
    if not getattr(config.model, "local_high_conf_branch_enabled", False):
        return {"local_high_conf_branch": {"enabled": False}}
    if not getattr(config.model, "local_high_conf_branch_training_ready", False):
        raise ValueError(
            "model.local_high_conf_branch.enabled=true requires "
            "local_branch_training_ready=true for formal Stage 1 training."
        )
    if float(getattr(config.model, "local_high_conf_branch_loss_weight", 0.0)) <= 0.0:
        raise ValueError(
            "model.local_high_conf_branch.enabled=true requires loss_weight > 0."
        )

    branch = LocalHighConfBranch(
        local_input_size=config.model.local_high_conf_branch_input_size,
        effective_stride=config.model.local_high_conf_branch_effective_stride,
        in_channels=3,
        output_dim=config.model.output_patch_dim or config.model.latent_dim,
        roi_margin=config.model.local_high_conf_branch_roi_margin,
    )
    branch.local_branch_training_ready = True
    try:
        device = next(model.parameters()).device
        branch = branch.to(device)
    except StopIteration:
        pass
    model.add_module("local_high_conf_branch", branch)
    return {
        "local_high_conf_branch": {
            "enabled": True,
            "local_branch_training_ready": True,
            "loss_weight": float(config.model.local_high_conf_branch_loss_weight),
            "local_input_size": list(config.model.local_high_conf_branch_input_size),
            "effective_stride": int(config.model.local_high_conf_branch_effective_stride),
        }
    }


__all__ = ["attach_local_high_conf_branch"]
