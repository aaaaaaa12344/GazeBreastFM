from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn


def _is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    """Parameters that should NOT receive weight decay."""
    if param.ndim < 2:
        return True
    lower = name.lower()
    if any(kw in lower for kw in ("bn", "bias", "layernorm", "ln", "norm")):
        return True
    return False


def audit_trainable_parameters(
    model: nn.Module,
    semantic_branch: nn.Module | None = None,
) -> dict[str, Any]:
    """Inventory all trainable parameters by module ownership.

    Returns a dict with per-module counts, total count, and total param count.
    Each trainable parameter is counted exactly once.
    """
    pretrained_ids: set[int] = set()
    if hasattr(model, "backbone"):
        pretrained_ids = {id(p) for p in model.backbone.parameters() if p.requires_grad}

    inventory: dict[str, dict[str, int]] = {
        "pretrained_backbone": {"count": 0, "params": 0},
        "visual_projections": {"count": 0, "params": 0},
        "joint_model_heads": {"count": 0, "params": 0},
        "semantic_branch": {"count": 0, "params": 0},
        "graph_encoder": {"count": 0, "params": 0},
        "graph_text_fusion": {"count": 0, "params": 0},
        "concept_heads": {"count": 0, "params": 0},
        "frozen": {"count": 0, "params": 0},
        "unclassified": {"count": 0, "params": 0},
    }

    all_modules: list[tuple[str, nn.Module]] = [("model", model)]
    if semantic_branch is not None:
        all_modules.append(("semantic_branch", semantic_branch))

    seen: set[int] = set()

    for module_prefix, module in all_modules:
        for name, param in module.named_parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            if not param.requires_grad:
                inventory["frozen"]["count"] += 1
                inventory["frozen"]["params"] += param.numel()
                continue

            full_name = f"{module_prefix}.{name}" if module_prefix else name

            if pid in pretrained_ids:
                inventory["pretrained_backbone"]["count"] += 1
                inventory["pretrained_backbone"]["params"] += param.numel()
            elif "proj" in full_name.lower() or "projection" in full_name.lower():
                inventory["visual_projections"]["count"] += 1
                inventory["visual_projections"]["params"] += param.numel()
            elif "graph_encoder" in full_name.lower():
                inventory["graph_encoder"]["count"] += 1
                inventory["graph_encoder"]["params"] += param.numel()
            elif "graph_text_fusion" in full_name.lower() or "text_fusion" in full_name.lower():
                inventory["graph_text_fusion"]["count"] += 1
                inventory["graph_text_fusion"]["params"] += param.numel()
            elif "concept_head" in full_name.lower() or "concept" in full_name.lower():
                inventory["concept_heads"]["count"] += 1
                inventory["concept_heads"]["params"] += param.numel()
            elif module_prefix == "semantic_branch":
                inventory["semantic_branch"]["count"] += 1
                inventory["semantic_branch"]["params"] += param.numel()
            else:
                inventory["joint_model_heads"]["count"] += 1
                inventory["joint_model_heads"]["params"] += param.numel()

    total_trainable = sum(
        v["params"] for k, v in inventory.items() if k not in ("frozen", "unclassified")
    )
    inventory["_total_trainable_params"] = int(total_trainable)
    inventory["_total_trainable_tensors"] = sum(
        v["count"] for k, v in inventory.items()
        if isinstance(v, dict) and k not in ("frozen", "unclassified")
    )

    return inventory


def build_optimizer_with_differential_lr(
    model: nn.Module,
    semantic_branch: nn.Module | None,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Build an AdamW optimizer with 4 differential-LR parameter groups.

    Groups:
      1. backbone_decay: pretrained backbone params with weight_decay
      2. backbone_no_decay: pretrained backbone bias/norm params (no decay)
      3. heads_decay: new head params with weight_decay
      4. heads_no_decay: new head bias/norm params (no decay)

    Every requires_grad parameter must enter exactly one group.
    """
    inventory = audit_trainable_parameters(model, semantic_branch)

    pretrained_ids: set[int] = set()
    if hasattr(model, "backbone"):
        pretrained_ids = {id(p) for p in model.backbone.parameters()}
    elif hasattr(model, "vision_encoder"):
        # Check if vision_encoder wraps a backbone attribute
        ve = model.vision_encoder
        if hasattr(ve, "backbone"):
            pretrained_ids = {id(p) for p in ve.backbone.parameters()}
        else:
            pretrained_ids = {id(p) for p in ve.parameters()}

    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    heads_decay: list[nn.Parameter] = []
    heads_no_decay: list[nn.Parameter] = []

    all_params: list[tuple[str, nn.Parameter]] = list(model.named_parameters())
    if semantic_branch is not None:
        all_params.extend(
            (f"semantic_branch.{n}", p) for n, p in semantic_branch.named_parameters()
        )

    assigned: set[int] = set()

    for name, param in all_params:
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in assigned:
            continue
        assigned.add(pid)

        is_backbone = pid in pretrained_ids
        no_decay = _is_no_decay_param(name, param)

        if is_backbone and no_decay:
            backbone_no_decay.append(param)
        elif is_backbone:
            backbone_decay.append(param)
        elif no_decay:
            heads_no_decay.append(param)
        else:
            heads_decay.append(param)

    # Verify coverage
    all_trainable_ids = {id(p) for n, p in all_params if p.requires_grad}
    assigned_ids = (
        {id(p) for p in backbone_decay}
        | {id(p) for p in backbone_no_decay}
        | {id(p) for p in heads_decay}
        | {id(p) for p in heads_no_decay}
    )
    missing = all_trainable_ids - assigned_ids
    if missing:
        missing_names = [n for n, p in all_params if id(p) in missing]
        raise ValueError(
            f"Parameters not assigned to any optimizer group: {missing_names}"
        )

    param_groups = []
    if backbone_decay:
        param_groups.append(
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay}
        )
    if backbone_no_decay:
        param_groups.append(
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0}
        )
    if heads_decay:
        param_groups.append(
            {"params": heads_decay, "lr": head_lr, "weight_decay": weight_decay}
        )
    if heads_no_decay:
        param_groups.append(
            {"params": heads_no_decay, "lr": head_lr, "weight_decay": 0.0}
        )

    optimizer = torch.optim.AdamW(param_groups)

    group_summary = {
        "backbone_decay": len(backbone_decay),
        "backbone_no_decay": len(backbone_no_decay),
        "heads_decay": len(heads_decay),
        "heads_no_decay": len(heads_no_decay),
        "total_params": (
            len(backbone_decay) + len(backbone_no_decay) + len(heads_decay) + len(heads_no_decay)
        ),
    }

    return optimizer, {"parameter_inventory": inventory, "group_summary": group_summary}


__all__ = [
    "audit_trainable_parameters",
    "build_optimizer_with_differential_lr",
]
