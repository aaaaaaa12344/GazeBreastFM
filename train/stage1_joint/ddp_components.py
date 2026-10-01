from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel


def unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """Return the owned module without changing non-DDP call sites."""
    return module.module if isinstance(module, DistributedDataParallel) else module


@dataclass(frozen=True)
class TrainableOwnership:
    module: str
    parameter_count: int
    requires_grad_count: int
    optimizer_parameter_count: int
    ddp_synchronized: bool


def wrap_independent_trainable_components(
    *,
    model: torch.nn.Module,
    semantic_branch: torch.nn.Module,
    mask_regressor: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    find_unused_parameters: bool = False,
    semantic_find_unused_parameters: bool | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Wrap the three disjoint formal trainable components exactly once."""
    if device.type != "cuda":
        return model, semantic_branch, mask_regressor
    ddp_kwargs = {
        "device_ids": [int(local_rank)],
        "output_device": int(local_rank),
        "find_unused_parameters": bool(find_unused_parameters),
        "static_graph": True,
        # Frozen BatchNorm buffers must not be rebroadcast between the clean
        # and masked forwards of one autograd graph.
        "broadcast_buffers": False,
    }
    semantic_kwargs = {
        **ddp_kwargs,
        "find_unused_parameters": bool(
            find_unused_parameters
            if semantic_find_unused_parameters is None
            else semantic_find_unused_parameters
        ),
    }
    return (
        DistributedDataParallel(model, **ddp_kwargs),
        DistributedDataParallel(semantic_branch, **semantic_kwargs),
        DistributedDataParallel(mask_regressor, **ddp_kwargs),
    )


def audit_trainable_ownership(
    *,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    ddp_wrapped_names: set[str],
) -> list[TrainableOwnership]:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }
    seen: set[int] = set()
    report: list[TrainableOwnership] = []
    for name, module in modules.items():
        params = list(module.parameters())
        trainable = [parameter for parameter in params if parameter.requires_grad]
        overlap = seen.intersection(id(parameter) for parameter in trainable)
        if overlap:
            raise ValueError(f"Duplicate trainable parameter ownership in module {name!r}.")
        seen.update(id(parameter) for parameter in trainable)
        report.append(
            TrainableOwnership(
                module=name,
                parameter_count=sum(parameter.numel() for parameter in params),
                requires_grad_count=sum(parameter.numel() for parameter in trainable),
                optimizer_parameter_count=sum(
                    parameter.numel() for parameter in trainable if id(parameter) in optimizer_ids
                ),
                ddp_synchronized=name in ddp_wrapped_names,
            )
        )
    return report


__all__ = [
    "TrainableOwnership",
    "audit_trainable_ownership",
    "unwrap_ddp",
    "wrap_independent_trainable_components",
]
