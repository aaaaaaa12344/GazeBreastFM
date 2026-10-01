from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from breast_pretrain.models import build_visual_encoder
from breast_pretrain.train.reproducibility import (
    build_initial_model_state,
    clone_state_dict_to_cpu,
    compute_state_dict_checksum,
)
from breast_pretrain.train.stage1_joint.types import (
    Stage1JointTrainerConfig,
    StudentForwardOutput,
)
from breast_pretrain.train.stage1_joint.ddp_components import unwrap_ddp


_MAMMO_FM_BACKEND = "mammo_fm_timm_efficientnet_b5"


def _compact_activation_summary(
    value: torch.Tensor,
    batch_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return diagnostic-only activation statistics without retaining the tensor."""
    detached = value.detach()
    numeric = detached.is_floating_point() or detached.is_complex()
    finite = torch.isfinite(detached) if numeric else torch.ones_like(detached, dtype=torch.bool)
    nonfinite = ~finite
    finite_values = detached[finite].float()
    nonfinite_count = int(nonfinite.sum().item())
    first_nonfinite = None
    if nonfinite_count:
        first_nonfinite = int(torch.nonzero(nonfinite.reshape(-1), as_tuple=False)[0].item())
    summary: dict[str, Any] = {
        "dtype": str(value.dtype),
        "shape": list(detached.shape),
        "numel": int(detached.numel()),
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": nonfinite_count,
        "nan_count": int(torch.isnan(detached).sum().item()) if numeric else 0,
        "posinf_count": int(torch.isposinf(detached).sum().item()) if numeric else 0,
        "neginf_count": int(torch.isneginf(detached).sum().item()) if numeric else 0,
        "first_nonfinite_flat_index": first_nonfinite,
        "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
        "finite_absmax": float(finite_values.abs().max().item()) if finite_values.numel() else None,
    }
    image_ids = (batch_context or {}).get("image_id", [])
    if detached.ndim >= 1 and int(detached.shape[0]) == len(image_ids):
        per_sample: list[dict[str, Any]] = []
        for batch_index in range(int(detached.shape[0])):
            sample = detached[batch_index]
            sample_numeric = sample.is_floating_point() or sample.is_complex()
            sample_finite = (
                torch.isfinite(sample)
                if sample_numeric
                else torch.ones_like(sample, dtype=torch.bool)
            )
            sample_values = sample[sample_finite].float()
            per_sample.append({
                "batch_index": batch_index,
                "image_id": image_ids[batch_index],
                "case_id": (batch_context or {}).get("case_id", [None] * len(image_ids))[batch_index],
                "dataset_id": (batch_context or {}).get("dataset_id", [None] * len(image_ids))[batch_index],
                "modality": (batch_context or {}).get("modality", [None] * len(image_ids))[batch_index],
                "finite_count": int(sample_finite.sum().item()),
                "nonfinite_count": int((~sample_finite).sum().item()),
                "nan_count": int(torch.isnan(sample).sum().item()) if sample_numeric else 0,
                "posinf_count": int(torch.isposinf(sample).sum().item()) if sample_numeric else 0,
                "neginf_count": int(torch.isneginf(sample).sum().item()) if sample_numeric else 0,
                "finite_absmax": (
                    float(sample_values.abs().max().item()) if sample_values.numel() else None
                ),
            })
        summary["per_sample"] = per_sample
    return summary


def _single_tensor_or_unsupported(value: Any) -> tuple[torch.Tensor | None, str | None]:
    if isinstance(value, torch.Tensor):
        return value, None
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0], None
    return None, type(value).__name__


def _module_parameter_dtypes(module: Any) -> list[str]:
    if not isinstance(module, nn.Module):
        return []
    return sorted({str(parameter.dtype) for parameter in module.parameters()})


def _compact_or_unsupported_summary(
    value: Any,
    batch_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep hook diagnostics total without retaining an activation tensor."""
    try:
        tensor, unsupported = _single_tensor_or_unsupported(value)
        return (
            _compact_activation_summary(tensor, batch_context)
            if tensor is not None
            else {"unsupported_output_type": unsupported}
        )
    except Exception as exc:
        return {"summary_error_type": type(exc).__name__}


def _record_internal_module_summary(
    summaries: dict[str, Any],
    name: str,
    value: Any,
    batch_context: dict[str, Any] | None,
) -> None:
    summaries[name] = _compact_or_unsupported_summary(value, batch_context)


def _first_nonfinite_internal_module(
    summaries: dict[str, Any], execution_order: list[str]
) -> str | None:
    for name in execution_order:
        summary = summaries.get(name)
        if isinstance(summary, dict) and int(summary.get("nonfinite_count", 0)) > 0:
            return name
    return None


def _spatial_projection_role(call_index: int) -> str:
    if call_index == 0:
        return "stride16"
    if call_index == 1:
        return "stride32_pooled"
    return f"unexpected_call_{call_index}"


def _set_spatial_projection_alias(
    summaries: dict[str, Any],
    call_index: int,
    component: str,
    summary: dict[str, Any],
) -> None:
    """Keep legacy keys stable aliases for the first, stride16 invocation."""
    if call_index == 0:
        summaries[f"spatial_proj_stride16_{component}"] = summary
        summaries[f"spatial_proj_{component}"] = summary
    elif call_index == 1:
        summaries[f"spatial_proj_stride32_pooled_{component}"] = summary


def _first_nonfinite_spatial_projection(summaries: dict[str, Any]) -> str | None:
    invocations = summaries.get("spatial_proj_invocations", [])
    if not isinstance(invocations, list):
        return None
    for invocation in invocations:
        if not isinstance(invocation, dict):
            continue
        role = invocation.get("role")
        if not isinstance(role, str):
            continue
        for component in ("input", "output"):
            summary = invocation.get(component)
            if isinstance(summary, dict) and int(summary.get("nonfinite_count", 0)) > 0:
                return f"{role}_{component}"
    return None


def _summary_has_nonfinite(summary: Any) -> bool:
    return isinstance(summary, dict) and int(summary.get("nonfinite_count", 0)) > 0


def _first_nonfinite_stride16_child(summaries: dict[str, Any]) -> tuple[str | None, int | None, str | None]:
    stage = summaries.get("stride16_stage")
    if not isinstance(stage, dict) or stage.get("status") == "unavailable":
        return None, None, None
    if _summary_has_nonfinite(stage.get("input")):
        return "stage_input", None, None
    children = stage.get("children", [])
    if not isinstance(children, list):
        return None, None, None
    for child in sorted(
        (item for item in children if isinstance(item, dict)),
        key=lambda item: int(item.get("actual_call_index", -1)),
    ):
        qualified_name = child.get("qualified_name")
        if not isinstance(qualified_name, str):
            continue
        child_index = child.get("child_index")
        child_type = child.get("module_type")
        for component in ("input", "output"):
            if _summary_has_nonfinite(child.get(component)):
                return (
                    f"{qualified_name}.{component}",
                    int(child_index) if isinstance(child_index, int) else None,
                    child_type if isinstance(child_type, str) else None,
                )
    return None, None, None


_BLOCK_OPERATOR_NAMES = (
    "conv_s2d", "bn_s2d", "conv_pw", "bn1", "conv_dw", "bn2", "aa", "se",
    "conv_pwl", "bn3", "drop_path",
)


def _record_operator_summary(
    value: Any,
    batch_context: dict[str, Any] | None,
) -> dict[str, Any]:
    return _compact_or_unsupported_summary(value, batch_context)


def _shadow_residual_add(
    shortcut: torch.Tensor | None,
    branch: torch.Tensor | None,
    batch_context: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_applicable",
        "shortcut_finite": False,
        "branch_finite": False,
        "post_add_finite": None,
        "shortcut_absmax": None,
        "branch_absmax": None,
        "post_add_absmax": None,
        "candidate_count": 0,
        "same_sign_overflow_candidate_count": 0,
        "positive_overflow_candidate_count": 0,
        "negative_overflow_candidate_count": 0,
        "per_sample": [],
    }
    if not isinstance(shortcut, torch.Tensor) or not isinstance(branch, torch.Tensor):
        result["reason"] = "operand_unavailable"
        return result
    if shortcut.shape != branch.shape or not shortcut.is_floating_point() or not branch.is_floating_point():
        result["reason"] = "operand_shape_or_dtype_incompatible"
        return result
    shortcut_finite = bool(torch.isfinite(shortcut).all().item())
    branch_finite = bool(torch.isfinite(branch).all().item())
    result["status"] = "computed"
    result["shortcut_finite"] = shortcut_finite
    result["branch_finite"] = branch_finite
    with torch.no_grad():
        shadow_sum = shortcut.float() + branch.float()
        finite_operands = torch.isfinite(shortcut) & torch.isfinite(branch)
        limit = float(torch.finfo(torch.float16).max)
        positive = finite_operands & (shadow_sum > limit)
        negative = finite_operands & (shadow_sum < -limit)
        candidates = positive | negative
        result["candidate_count"] = int(candidates.sum().item())
        result["same_sign_overflow_candidate_count"] = result["candidate_count"]
        result["positive_overflow_candidate_count"] = int(positive.sum().item())
        result["negative_overflow_candidate_count"] = int(negative.sum().item())
        image_ids = (batch_context or {}).get("image_id", [])
        if shadow_sum.ndim >= 1 and int(shadow_sum.shape[0]) == len(image_ids):
            for batch_index in range(int(shadow_sum.shape[0])):
                result["per_sample"].append({
                    "batch_index": batch_index,
                    "image_id": image_ids[batch_index],
                    "case_id": (batch_context or {}).get("case_id", [None] * len(image_ids))[batch_index],
                    "dataset_id": (batch_context or {}).get("dataset_id", [None] * len(image_ids))[batch_index],
                    "modality": (batch_context or {}).get("modality", [None] * len(image_ids))[batch_index],
                    "candidate_count": int(candidates[batch_index].sum().item()),
                    "positive_overflow_candidate_count": int(positive[batch_index].sum().item()),
                    "negative_overflow_candidate_count": int(negative[batch_index].sum().item()),
                })
    return result


def _finalize_block_operator_localization(
    localization: dict[str, Any],
    shortcut_ref: torch.Tensor | None,
    branch_ref: torch.Tensor | None,
) -> None:
    input_summary = localization.get("block_4_6_shortcut_input")
    post_summary = localization.get("residual_post_add")
    operator_records = localization.get("target_operator_summaries", [])
    if _summary_has_nonfinite(input_summary):
        first_bad = "block_input"
        first_type = None
    else:
        first_bad = None
        first_type = None
        for record in operator_records if isinstance(operator_records, list) else []:
            if not isinstance(record, dict):
                continue
            name = record.get("operator_name")
            module_type = record.get("module_type")
            if not isinstance(name, str):
                continue
            if _summary_has_nonfinite(record.get("input")):
                first_bad, first_type = f"{name}.input", module_type
                break
            if _summary_has_nonfinite(record.get("output")):
                first_bad, first_type = f"{name}.output", module_type
                break
        if first_bad is None and _summary_has_nonfinite(post_summary):
            first_bad = "residual_add" if localization.get("has_skip") is True else "block_output"
            first_type = "residual_add" if first_bad == "residual_add" else None
    localization["first_nonfinite_block_4_6_operator"] = first_bad
    localization["first_nonfinite_block_4_6_operator_type"] = first_type
    post_nonfinite = _summary_has_nonfinite(post_summary)
    shadow = _shadow_residual_add(shortcut_ref, branch_ref, localization.get("batch_context"))
    shadow["post_add_finite"] = not post_nonfinite
    shadow["shortcut_absmax"] = (
        localization.get("residual_shortcut", {}).get("finite_absmax")
        if isinstance(localization.get("residual_shortcut"), dict) else None
    )
    shadow["branch_absmax"] = (
        localization.get("residual_branch_pre_add", {}).get("finite_absmax")
        if isinstance(localization.get("residual_branch_pre_add"), dict) else None
    )
    shadow["post_add_absmax"] = (
        localization.get("residual_post_add", {}).get("finite_absmax")
        if isinstance(localization.get("residual_post_add"), dict) else None
    )
    localization["residual_add_shadow"] = shadow
    localization["residual_add_overflow_candidate_count"] = shadow.get("candidate_count", 0)
    localization["same_sign_overflow_candidate_count"] = shadow.get("candidate_count", 0)
    shortcut_finite = _summary_has_nonfinite(localization.get("residual_shortcut")) is False and bool(shadow.get("shortcut_finite"))
    branch_finite = _summary_has_nonfinite(localization.get("residual_branch_pre_add")) is False and bool(shadow.get("branch_finite"))
    localization["residual_add_overflow_observed"] = bool(
        localization.get("has_skip") is True
        and shortcut_finite
        and branch_finite
        and post_nonfinite
        and int(shadow.get("candidate_count", 0)) > 0
    )
    localization.pop("batch_context", None)


def _register_block_operator_hooks(
    model_module: nn.Module,
    summaries: dict[str, Any],
    batch_context: dict[str, Any] | None,
) -> tuple[list[Any], dict[str, Any] | None, dict[str, Any]]:
    target_name = os.environ.get("HSM_STAGE1_DIAGNOSTIC_TARGET_MODULE")
    if not target_name:
        return [], None, {"status": "inactive", "reason": "target_env_not_set"}
    module_map = dict(model_module.named_modules())
    target = module_map.get(target_name)
    if not isinstance(target, nn.Module):
        backbone = getattr(model_module, "backbone", None)
        if isinstance(backbone, nn.Module):
            target = dict(backbone.named_modules()).get(target_name)
    if not isinstance(target, nn.Module):
        localization = {
            "status": "target_unavailable",
            "target_module_name": target_name,
            "reason": "module_path_not_found",
        }
        summaries["target_block_operator_localization"] = localization
        return [], None, localization
    inventory = dict(target.named_children())
    localization: dict[str, Any] = {
        "status": "available",
        "target_module_name": target_name,
        "target_module_type": type(target).__name__,
        "has_skip": getattr(target, "has_skip", None) if isinstance(getattr(target, "has_skip", None), bool) else None,
        "conv_s2d_present": "conv_s2d" in inventory,
        "bn_s2d_present": "bn_s2d" in inventory,
        "drop_path_type": type(inventory["drop_path"]).__name__ if "drop_path" in inventory else None,
        "drop_prob": (
            float(getattr(inventory.get("drop_path"), "drop_prob"))
            if isinstance(getattr(inventory.get("drop_path"), "drop_prob", None), (int, float))
            else None
        ),
        "target_module_inventory": [
            {"name": name, "module_type": type(module).__name__}
            for name, module in inventory.items()
        ],
        "target_operator_summaries": [],
        "batch_context": batch_context,
    }
    summaries["target_block_operator_localization"] = localization
    handles: list[Any] = []
    shortcut_ref: torch.Tensor | None = None
    branch_ref: torch.Tensor | None = None
    active_by_module: dict[int, list[dict[str, Any]]] = {}
    next_call_index = 0

    def capture_block_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal shortcut_ref
        localization["block_4_6_shortcut_input"] = _record_operator_summary(inputs, batch_context)
        shortcut_ref, _ = _single_tensor_or_unsupported(inputs)
        localization["residual_shortcut"] = localization["block_4_6_shortcut_input"]

    def capture_block_output(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal branch_ref, shortcut_ref
        localization["residual_post_add"] = _record_operator_summary(output, batch_context)
        if isinstance(output, torch.Tensor):
            localization["block_4_6_output"] = localization["residual_post_add"]
        _finalize_block_operator_localization(localization, shortcut_ref, branch_ref)
        shortcut_ref = None
        branch_ref = None

    handles.append(target.register_forward_pre_hook(capture_block_input))
    handles.append(target.register_forward_hook(capture_block_output))
    for operator_name in _BLOCK_OPERATOR_NAMES:
        module = inventory.get(operator_name)
        if not isinstance(module, nn.Module):
            continue
        module_key = id(module)
        active_by_module[module_key] = []

        def capture_operator_input(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            captured_name: str = operator_name,
            captured_module: nn.Module = module,
            captured_key: int = module_key,
        ) -> None:
            nonlocal next_call_index
            record = {
                "call_index": next_call_index,
                "operator_name": captured_name,
                "module_type": type(captured_module).__name__,
                "input": _record_operator_summary(inputs, batch_context),
            }
            next_call_index += 1
            localization["target_operator_summaries"].append(record)
            active_by_module[captured_key].append(record)

        def capture_operator_output(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            captured_name: str = operator_name,
            captured_key: int = module_key,
        ) -> None:
            nonlocal branch_ref
            records = active_by_module[captured_key]
            if records:
                record = records.pop()
                record["output"] = _record_operator_summary(output, batch_context)
                if captured_name == "drop_path":
                    branch_ref, _ = _single_tensor_or_unsupported(output)
                    localization["residual_branch_pre_add"] = record["output"]

        handles.append(module.register_forward_pre_hook(capture_operator_input))
        handles.append(module.register_forward_hook(capture_operator_output))
    return handles, localization, {"shortcut_ref": shortcut_ref, "branch_ref": branch_ref}


def _register_stride16_stage_hooks(
    backbone: nn.Module,
    stride16_module_name: Any,
    summaries: dict[str, Any],
    batch_context: dict[str, Any] | None,
) -> list[Any]:
    """Capture only the resolved stride-16 stage and its immediate children."""
    if not isinstance(stride16_module_name, str) or not stride16_module_name:
        summaries["stride16_stage"] = {
            "status": "unavailable",
            "reason": "missing_stride16_module_name",
        }
        return []
    stage = dict(backbone.named_modules()).get(stride16_module_name)
    if not isinstance(stage, nn.Module):
        summaries["stride16_stage"] = {
            "status": "unavailable",
            "stage_name": stride16_module_name,
            "reason": "stage_not_found_in_backbone_named_modules",
        }
        return []

    declared_children = list(stage.named_children())
    stage_summary: dict[str, Any] = {
        "status": "available",
        "stage_name": stride16_module_name,
        "stage_type": type(stage).__name__,
        "stage_child_count": len(declared_children),
        "stage_children": [
            {
                "child_index": child_index,
                "child_name": child_name,
                "qualified_name": f"{stride16_module_name}.{child_name}",
                "module_type": type(child).__name__,
            }
            for child_index, (child_name, child) in enumerate(declared_children)
        ],
        "children": [],
    }
    summaries["stride16_stage"] = stage_summary
    handles: list[Any] = []
    def capture_stage_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if "input" not in stage_summary:
            stage_summary["input"] = _compact_or_unsupported_summary(inputs, batch_context)

    handles.append(stage.register_forward_pre_hook(capture_stage_input))

    active_invocations: dict[int, list[dict[str, Any]]] = {}
    next_call_index = 0
    for child_index, (child_name, child) in enumerate(declared_children):
        qualified_name = f"{stride16_module_name}.{child_name}"
        module_type = type(child).__name__
        child_key = id(child)
        active_invocations[child_key] = []

        def capture_child_input(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            *,
            captured_index: int = child_index,
            captured_name: str = child_name,
            captured_qualified_name: str = qualified_name,
            captured_type: str = module_type,
            captured_key: int = child_key,
        ) -> None:
            nonlocal next_call_index
            invocation = {
                "child_index": captured_index,
                "child_name": captured_name,
                "qualified_name": captured_qualified_name,
                "module_type": captured_type,
                "actual_call_index": next_call_index,
                "input": _compact_or_unsupported_summary(inputs, batch_context),
            }
            next_call_index += 1
            stage_summary["children"].append(invocation)
            active_invocations[captured_key].append(invocation)

        def capture_child_output(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            captured_key: int = child_key,
        ) -> None:
            invocations = active_invocations[captured_key]
            if invocations:
                invocations.pop()["output"] = _compact_or_unsupported_summary(output, batch_context)

        handles.append(child.register_forward_pre_hook(capture_child_input))
        handles.append(child.register_forward_hook(capture_child_output))
    return handles


def _register_mammo_internal_hooks(
    model_module: nn.Module,
    summaries: dict[str, Any],
    batch_context: dict[str, Any] | None,
) -> tuple[list[Any], list[str]]:
    """Install diagnostic-only major-stage hooks; callers always remove handles."""
    backbone = getattr(model_module, "backbone", None)
    if not isinstance(backbone, nn.Module):
        return [], []
    handles: list[Any] = []
    execution_order: list[str] = ["backbone_input"]
    handles.append(backbone.register_forward_pre_hook(
        lambda _module, inputs: _record_internal_module_summary(
            summaries, "backbone_input", inputs, batch_context
        )
    ))

    def add_output_hook(name: str, module: Any) -> None:
        if not isinstance(module, nn.Module):
            return
        execution_order.append(name)
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, captured_name=name: _record_internal_module_summary(
                summaries, captured_name, output, batch_context
            )
        ))

    add_output_hook("conv_stem", getattr(backbone, "conv_stem", None))
    add_output_hook("bn1", getattr(backbone, "bn1", None))
    blocks = getattr(backbone, "blocks", None)
    if isinstance(blocks, nn.Module):
        for block_name, block in blocks.named_children():
            add_output_hook(f"blocks.{block_name}", block)
    add_output_hook("conv_head", getattr(backbone, "conv_head", None))
    add_output_hook("bn2", getattr(backbone, "bn2", None))
    add_output_hook("global_pool", getattr(backbone, "global_pool", None))
    stride16_module_name = getattr(model_module, "_stride16_module_name", None)
    handles.extend(_register_stride16_stage_hooks(
        backbone,
        stride16_module_name,
        summaries,
        batch_context,
    ))
    block_operator_handles, _block_localization, _ = _register_block_operator_hooks(
        model_module,
        summaries,
        batch_context,
    )
    handles.extend(block_operator_handles)
    if isinstance(_block_localization, dict) and _block_localization.get("status") == "available":
        summaries["target_operator_summaries"] = _block_localization["target_operator_summaries"]
    spatial_proj = getattr(model_module, "spatial_proj", None)
    if isinstance(spatial_proj, nn.Module):
        invocations: list[dict[str, Any]] = []
        active_call_indices: list[int] = []
        next_call_index = 0
        summaries["spatial_proj_invocations"] = invocations

        def capture_spatial_proj_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            nonlocal next_call_index
            call_index = next_call_index
            next_call_index += 1
            input_summary = _compact_or_unsupported_summary(inputs, batch_context)
            invocation = {
                "call_index": call_index,
                "role": _spatial_projection_role(call_index),
                "input": input_summary,
            }
            invocations.append(invocation)
            active_call_indices.append(call_index)
            _set_spatial_projection_alias(summaries, call_index, "input", input_summary)

        def capture_spatial_proj_output(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not active_call_indices:
                return
            call_index = active_call_indices.pop()
            output_summary = _compact_or_unsupported_summary(output, batch_context)
            invocations[call_index]["output"] = output_summary
            _set_spatial_projection_alias(summaries, call_index, "output", output_summary)

        handles.append(spatial_proj.register_forward_pre_hook(capture_spatial_proj_input))
        handles.append(spatial_proj.register_forward_hook(capture_spatial_proj_output))
    summaries["stride16_module_name"] = stride16_module_name
    summaries["internal_execution_order"] = execution_order
    return handles, execution_order


def _enforce_patch_gaze_weight_alignment(
    *,
    patch_gaze_weight: torch.Tensor | None,
    patch_grid: tuple[int, int] | None,
    encoder_backend_name: str,
) -> None:
    """R4: Require patch_gaze_weight length == patch_grid[0] * patch_grid[1].

    Silent broadcast, truncation, or interpolation is not allowed without an
    explicit projection contract and audit evidence.
    """
    if patch_gaze_weight is None or patch_grid is None:
        return
    expected_n = int(patch_grid[0]) * int(patch_grid[1])
    prior_n = int(patch_gaze_weight.shape[1])
    if prior_n == expected_n:
        return
    raise RuntimeError(
        f"patch_gaze_weight shape mismatch: prior has {prior_n} patches but "
        f"patch_grid {patch_grid} expects {expected_n} patches. "
        f"encoder_backend={encoder_backend_name}. "
        f"Silent broadcast/truncation/interpolation is forbidden (R4). "
        f"Either apply an explicit projection contract with audit evidence, "
        f"or regenerate the gaze prior at the correct resolution."
    )


def build_student_encoder(
    config: Stage1JointTrainerConfig,
    device: torch.device,
) -> tuple[nn.Module, str]:
    initial_model_state = None
    initial_model_checksum = None
    if config.reproducibility.reuse_initial_model:
        output_patch_dim = config.model.output_patch_dim or config.model.latent_dim
        initial_model_state, initial_model_checksum = build_initial_model_state(
            image_size=config.data.image_size,
            patch_size=config.model.patch_size,
            latent_dim=output_patch_dim,
            seed=config.reproducibility.seed,
            deterministic_ablation=config.reproducibility.deterministic_ablation,
        )

    model = build_visual_encoder(
        model_config=config.model,
        image_size=config.data.image_size,
    )
    if initial_model_state is not None:
        model.load_state_dict(initial_model_state)
        model_init_checksum = initial_model_checksum or compute_state_dict_checksum(
            initial_model_state
        )
    else:
        model_init_checksum = compute_state_dict_checksum(
            clone_state_dict_to_cpu(model.state_dict())
        )

    if config.model.modality_embedding:
        num_modalities = len(config.model.modality_vocab)
        embed_dim = config.model.modality_embedding_dim
        global_dim = config.model.latent_dim
        model._modality_embed = nn.Embedding(num_modalities, embed_dim).to(device)
        model._modality_proj_global = nn.Linear(embed_dim, global_dim).to(device)
        model._modality_vocab = list(config.model.modality_vocab)
        model._modality_embedding_strategy = config.model.modality_embedding_strategy

    return model.to(device), model_init_checksum


def set_bn_policy(model: nn.Module, policy: str, train_affine: bool) -> dict[str, object]:
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_count += 1
            if policy == "freeze_running_stats":
                m.eval()
            if m.weight is not None:
                m.weight.requires_grad = bool(train_affine)
            if m.bias is not None:
                m.bias.requires_grad = bool(train_affine)
    return {"bn_policy": policy, "train_affine": train_affine, "bn_layer_count": bn_count}


def get_bn_running_stats_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            snapshot[f"{name}.running_mean"] = m.running_mean.clone().detach().cpu()
            snapshot[f"{name}.running_var"] = m.running_var.clone().detach().cpu()
    return snapshot


def forward_student(
    model: nn.Module,
    image: torch.Tensor,
    modality_ids: torch.Tensor | None = None,
    patch_gaze_weight: torch.Tensor | None = None,
    diagnostic_tensors: dict[str, Any] | None = None,
    diagnostic_internal_summaries: dict[str, Any] | None = None,
    diagnostic_batch_context: dict[str, Any] | None = None,
) -> StudentForwardOutput:
    diagnostic_active = diagnostic_tensors is not None or diagnostic_internal_summaries is not None
    safety_requested = os.environ.get("HSM_STAGE1_MAMMO_ENCODER_FP32_SAFETY") == "1"
    model_module = unwrap_ddp(model) if (diagnostic_active or safety_requested) else None
    mammo_safety_island = bool(
        safety_requested
        and getattr(model_module, "backend_name", None) == _MAMMO_FM_BACKEND
    )
    global_proj_hook: Any | None = None
    internal_handles: list[Any] = []
    internal_execution_order: list[str] = []
    if (
        diagnostic_tensors is not None
        and getattr(model_module, "backend_name", None) == _MAMMO_FM_BACKEND
        and hasattr(model_module, "global_proj")
    ):
        def _capture_global_proj(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                diagnostic_tensors["mammo_global_2048_pre_projection"] = inputs[0]
            if isinstance(output, torch.Tensor):
                diagnostic_tensors["mammo_global_768_post_projection"] = output

        global_proj_hook = model_module.global_proj.register_forward_hook(_capture_global_proj)
    if (
        diagnostic_internal_summaries is not None
        and getattr(model_module, "backend_name", None) == _MAMMO_FM_BACKEND
    ):
        internal_handles, internal_execution_order = _register_mammo_internal_hooks(
            model_module,
            diagnostic_internal_summaries,
            diagnostic_batch_context,
        )

    if mammo_safety_island:
        encoder_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if image.device.type == "cuda"
            else nullcontext()
        )
        encoder_input = image.float()
    else:
        encoder_context = nullcontext()
        encoder_input = image

    if diagnostic_internal_summaries is not None:
        diagnostic_internal_summaries["mammo_encoder_precision_policy"] = (
            "fp32_safety_island" if mammo_safety_island else "amp_default_fp16"
        )
        diagnostic_internal_summaries["mammo_encoder_input_dtype"] = str(encoder_input.dtype)
        if getattr(model_module, "backend_name", None) == _MAMMO_FM_BACKEND:
            diagnostic_internal_summaries["mammo_backbone_parameter_dtypes"] = _module_parameter_dtypes(
                getattr(model_module, "backbone", None)
            )
            diagnostic_internal_summaries["mammo_global_proj_parameter_dtypes"] = _module_parameter_dtypes(
                getattr(model_module, "global_proj", None)
            )
            diagnostic_internal_summaries["mammo_spatial_proj_parameter_dtypes"] = _module_parameter_dtypes(
                getattr(model_module, "spatial_proj", None)
            )

    try:
        with encoder_context:
            encoded = model(encoder_input, return_dict=True)
    finally:
        if global_proj_hook is not None:
            global_proj_hook.remove()
        for handle in internal_handles:
            handle.remove()

    if diagnostic_internal_summaries is not None:
        diagnostic_internal_summaries["first_nonfinite_internal_module"] = (
            _first_nonfinite_internal_module(
                diagnostic_internal_summaries,
                internal_execution_order,
            )
        )
        diagnostic_internal_summaries["spatial_proj_invocation_count"] = len(
            diagnostic_internal_summaries.get("spatial_proj_invocations", [])
        )
        diagnostic_internal_summaries["first_nonfinite_spatial_projection"] = (
            _first_nonfinite_spatial_projection(diagnostic_internal_summaries)
        )
        (
            diagnostic_internal_summaries["first_nonfinite_stride16_child"],
            diagnostic_internal_summaries["first_nonfinite_stride16_child_index"],
            diagnostic_internal_summaries["first_nonfinite_stride16_child_type"],
        ) = _first_nonfinite_stride16_child(diagnostic_internal_summaries)
        block_localization = diagnostic_internal_summaries.get(
            "target_block_operator_localization"
        )
        if isinstance(block_localization, dict):
            for field in (
                "first_nonfinite_block_4_6_operator",
                "first_nonfinite_block_4_6_operator_type",
                "residual_add_overflow_observed",
                "residual_add_overflow_candidate_count",
                "same_sign_overflow_candidate_count",
            ):
                if field in block_localization:
                    diagnostic_internal_summaries[field] = block_localization[field]
            for field in (
                "target_module_name",
                "target_module_type",
                "has_skip",
                "target_module_inventory",
                "block_4_6_shortcut_input",
                "residual_shortcut",
                "residual_branch_pre_add",
                "residual_post_add",
                "residual_add_shadow",
            ):
                if field in block_localization:
                    diagnostic_internal_summaries[field] = block_localization[field]

    global_feature = encoded.get("global_feature", encoded["global_image_feature"])
    patch_tokens = encoded["patch_tokens"]

    if diagnostic_internal_summaries is not None:
        diagnostic_internal_summaries["mammo_patch_tokens_dtype"] = str(patch_tokens.dtype)
        diagnostic_internal_summaries["mammo_global_feature_dtype"] = str(global_feature.dtype)

    if model_module is None:
        model_module = unwrap_ddp(model)
    if diagnostic_tensors is not None:
        diagnostic_tensors["backbone_global_feature_pre_modality"] = global_feature

    if modality_ids is not None and hasattr(model_module, "_modality_embed"):
        embed = model_module._modality_embed
        proj = model_module._modality_proj_global
        strategy = getattr(model_module, "_modality_embedding_strategy", "add_to_global")

        modality_ids_dev = modality_ids.to(device=global_feature.device, dtype=torch.long)
        mod_embed = embed(modality_ids_dev)
        mod_proj = proj(mod_embed)
        if diagnostic_tensors is not None:
            diagnostic_tensors["modality_embedding_applied"] = True
            diagnostic_tensors["modality_embedding"] = mod_embed
            diagnostic_tensors["modality_projected_global"] = mod_proj

        if strategy == "add_to_global":
            global_feature = global_feature + mod_proj
        elif strategy == "concat_to_global":
            global_feature = torch.cat([global_feature, mod_proj], dim=-1)
        else:
            global_feature = global_feature + mod_proj
        if diagnostic_tensors is not None:
            diagnostic_tensors["global_feature_post_modality"] = global_feature
    elif diagnostic_tensors is not None:
        diagnostic_tensors["modality_embedding_applied"] = False

    patch_grid = encoded.get("patch_grid")
    num_patches_val = encoded.get("num_patches")

    # R4: mandatory patch_gaze_weight shape alignment
    _enforce_patch_gaze_weight_alignment(
        patch_gaze_weight=patch_gaze_weight,
        patch_grid=patch_grid,
        encoder_backend_name=str(
            encoded.get("encoder_backend_name", encoded.get("backend_name", ""))
        ),
    )

    return StudentForwardOutput(
        patch_tokens=patch_tokens,
        global_image_feature=global_feature,
        patch_grid=patch_grid,
        num_patches=num_patches_val,
    )
