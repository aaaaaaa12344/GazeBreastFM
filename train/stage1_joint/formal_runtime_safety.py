from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


class FormalNumericalFailure(RuntimeError):
    """Raised after every rank has observed a formal numerical failure."""


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _json_write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _iter_values(value: Any, name: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_values(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_values(item, f"{name}[{index}]")
    else:
        yield name, value


def tensor_summary(name: str, value: torch.Tensor | float | int, *, reason: str = "nonfinite") -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        detached = value.detach().float()
        finite = torch.isfinite(detached)
        nonfinite = ~finite
        finite_values = detached[finite]
        nonfinite_count = int(nonfinite.sum().item())
        first_nonfinite_flat_index = None
        if nonfinite_count:
            first_nonfinite_flat_index = int(
                torch.nonzero(nonfinite.reshape(-1), as_tuple=False)[0].item()
            )
        summary: dict[str, Any] = {
            "tensor_name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "finite_count": int(finite.sum().item()),
            "nonfinite_count": nonfinite_count,
            "nan_count": int(torch.isnan(detached).sum().item()),
            "posinf_count": int(torch.isposinf(detached).sum().item()),
            "neginf_count": int(torch.isneginf(detached).sum().item()),
            "first_nonfinite_flat_index": first_nonfinite_flat_index,
            "reason": reason,
        }
        if finite_values.numel():
            summary.update({
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "absmax": float(finite_values.abs().max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
            })
        else:
            summary.update({"min": None, "max": None, "absmax": None, "mean": None, "std": None})
        return summary
    scalar = float(value)
    return {
        "tensor_name": name,
        "dtype": type(value).__name__,
        "shape": [],
        "numel": 1,
        "finite_count": int(math.isfinite(scalar)),
        "nonfinite_count": int(not math.isfinite(scalar)),
        "nan_count": int(math.isnan(scalar)),
        "posinf_count": int(scalar == float("inf")),
        "neginf_count": int(scalar == float("-inf")),
        "first_nonfinite_flat_index": None if math.isfinite(scalar) else 0,
        "min": scalar if math.isfinite(scalar) else None,
        "max": scalar if math.isfinite(scalar) else None,
        "absmax": abs(scalar) if math.isfinite(scalar) else None,
        "mean": scalar if math.isfinite(scalar) else None,
        "std": 0.0 if math.isfinite(scalar) else None,
        "reason": reason,
    }


def first_invalid_value(
    named_values: dict[str, Any], *, positive_names: set[str] | None = None
) -> dict[str, Any] | None:
    positive = positive_names or set()
    for root_name, root_value in named_values.items():
        for name, value in _iter_values(root_value, root_name):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() or value.is_complex():
                    if not bool(torch.isfinite(value).all().item()):
                        return tensor_summary(name, value)
            elif isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    return tensor_summary(name, value)
            if name in positive:
                if isinstance(value, torch.Tensor):
                    if not bool((value > 0).all().item()):
                        return tensor_summary(name, value, reason="non_positive")
                elif not isinstance(value, (float, int)) or isinstance(value, bool) or float(value) <= 0.0:
                    return tensor_summary(name, value if isinstance(value, (float, int)) else 0.0, reason="non_positive")
    return None


def aggregate_has_invalid_value(
    named_values: dict[str, Any], *, positive_names: set[str] | None = None
) -> bool:
    """Return one aggregate non-finite decision without per-tensor host syncs.

    This is the normal formal-step gate.  It deliberately constructs GPU scalar
    predicates for every tensor and transfers only their final reduction to the
    host.  ``first_invalid_value`` remains forensic-only and is called only
    after this aggregate reports a failure (or while validating checkpoints).
    """
    positive = positive_names or set()
    tensor_flags: list[torch.Tensor] = []
    scalar_invalid = False
    for root_name, root_value in named_values.items():
        for name, value in _iter_values(root_value, root_name):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() or value.is_complex():
                    tensor_flags.append(torch.isfinite(value).all())
                if name in positive:
                    tensor_flags.append((value > 0).all())
            elif isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    scalar_invalid = True
                if name in positive and float(value) <= 0.0:
                    scalar_invalid = True
            elif name in positive:
                scalar_invalid = True
    if scalar_invalid:
        return True
    if not tensor_flags:
        return False
    # torch.stack preserves device execution and .item() synchronizes exactly
    # once for this whole gate, regardless of parameter/state tensor count.
    return not bool(torch.stack(tensor_flags).all().item())


def module_parameter_values(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            values[f"{module_name}.parameter.{name}"] = parameter
    return values


def module_gradient_values(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                values[f"{module_name}.gradient.{name}"] = parameter.grad
    return values


def optimizer_state_values(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {"optimizer_state": optimizer.state_dict().get("state", {})}


def scaler_values(scaler: torch.cuda.amp.GradScaler) -> dict[str, Any]:
    values: dict[str, Any] = {"scaler_state": scaler.state_dict()}
    if scaler.is_enabled():
        values["scaler_scale"] = float(scaler.get_scale())
    return values


def scaler_found_inf_after_unscale(
    scaler: torch.cuda.amp.GradScaler, optimizer: torch.optim.Optimizer
) -> bool:
    """Read GradScaler's current optimizer found-inf flags after ``unscale_``.

    PyTorch keeps these tensors in the scaler's per-optimizer state.  They are
    authoritative for the imminent ``scaler.step`` decision, so formal DDP can
    coordinate before any rank executes AdamW.
    """
    if not scaler.is_enabled():
        return False
    per_optimizer = getattr(scaler, "_per_optimizer_states", {}).get(id(optimizer), {})
    flags = list(per_optimizer.get("found_inf_per_device", {}).values())
    if not flags:
        return False
    return bool(torch.stack([flag.detach().to(dtype=torch.bool) for flag in flags]).any().item())


def should_recover_amp_overflow(
    *,
    effective_amp: bool,
    scaler_enabled: bool,
    local_found_inf: bool,
    found_inf_all_rank: bool,
    local_nonfinite_grad: bool,
    failure_reason: str,
    consecutive_recoveries: int,
) -> bool:
    """Allow exactly one coordinated post-unscale GradScaler overflow recovery."""
    return bool(
        effective_amp
        and scaler_enabled
        and local_found_inf
        and found_inf_all_rank
        and local_nonfinite_grad
        and failure_reason in {"nonfinite", "found_inf"}
        and int(consecutive_recoveries) == 0
    )


def _state_collection_summary(named_values: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    tensor_count = 0
    nonfinite_tensor_count = 0
    first_invalid = None
    global_absmax = None
    top_absmax: list[dict[str, Any]] = []
    for name, value in named_values:
        if not (value.is_floating_point() or value.is_complex()):
            continue
        tensor_count += 1
        detached = value.detach()
        finite = torch.isfinite(detached)
        if not bool(finite.all().item()):
            nonfinite_tensor_count += 1
            if first_invalid is None:
                first_invalid = tensor_summary(name, detached)
        finite_values = detached[finite].float()
        if finite_values.numel():
            absmax = float(finite_values.abs().max().item())
            global_absmax = absmax if global_absmax is None else max(global_absmax, absmax)
            top_absmax.append({"name": name, "absmax": absmax})
    top_absmax.sort(key=lambda item: item["absmax"], reverse=True)
    return {
        "tensor_count": tensor_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "first_invalid": first_invalid,
        "global_absmax": global_absmax,
        "top_absmax": top_absmax[:5],
    }


def model_state_at_failure(modules: dict[str, torch.nn.Module]) -> dict[str, Any]:
    """Failure-only, read-only state summary before backward or optimizer update."""
    parameters: list[tuple[str, torch.Tensor]] = []
    buffers: list[tuple[str, torch.Tensor]] = []
    for module_name, module in modules.items():
        parameters.extend(
            (f"{module_name}.parameter.{name}", parameter)
            for name, parameter in module.named_parameters()
        )
        buffers.extend(
            (f"{module_name}.buffer.{name}", buffer)
            for name, buffer in module.named_buffers()
        )
    parameter_summary = _state_collection_summary(parameters)
    buffer_summary = _state_collection_summary(buffers)
    return {
        "parameters": parameter_summary,
        "buffers": buffer_summary,
        "MODEL_PARAMETERS_FINITE": parameter_summary["nonfinite_tensor_count"] == 0,
        "MODEL_BUFFERS_FINITE": buffer_summary["nonfinite_tensor_count"] == 0,
    }


def _with_rank_failures(gathered: list[dict[str, Any] | None]) -> dict[str, Any]:
    rank_failures = [{"rank": rank, "failure": failure} for rank, failure in enumerate(gathered)]
    for source_rank, failure in enumerate(gathered):
        if failure is not None:
            return {"source_rank": source_rank, **failure, "rank_failures": rank_failures}
    raise RuntimeError("DDP numerical failure reduction lost the source-rank failure.")


def coordinate_failure(local_failure: dict[str, Any] | None, device: torch.device) -> dict[str, Any] | None:
    rank, world_size = _rank_world()
    if world_size == 1:
        return _with_rank_failures([local_failure]) if local_failure is not None else None
    flag = torch.tensor(int(local_failure is not None), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()) == 0:
        return None
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_failure)
    return _with_rank_failures(gathered)


def write_failure_receipt(
    *, output_dir: Path, context: dict[str, Any], failure: dict[str, Any]
) -> Path | None:
    rank, world_size = _rank_world()
    if rank != 0:
        return None
    attempt_step = int(context.get("attempt_step", -1))
    source_context = failure.get("source_context", {})
    path = output_dir / "numerical_failures" / f"numerical_failure_attempt_{attempt_step:06d}.json"
    payload = {
        "schema_version": "formal_stage1_numerical_failure_receipt_v2",
        "status": "FAIL",
        "rank": rank,
        "world_size": world_size,
        "failure_rank": failure.get("source_rank", rank),
        "failure_image_id": source_context.get("image_id", context.get("image_id", [])),
        "failure_case_id": source_context.get("case_id", context.get("case_id", [])),
        "failure_dataset_id": source_context.get("dataset_id", context.get("dataset_id", [])),
        "failure_modality": source_context.get("modality", context.get("modality", [])),
        "rank_failures": failure.get("rank_failures", []),
        **context,
        "failure": failure,
    }
    _json_write_atomic(path, payload)
    return path


def raise_coordinated_failure(
    *, output_dir: Path, context: dict[str, Any], failure: dict[str, Any]
) -> None:
    receipt = write_failure_receipt(output_dir=output_dir, context=context, failure=failure)
    raise FormalNumericalFailure(
        "FORMAL_NUMERICAL_FAIL_FAST: "
        f"boundary={context.get('failure_boundary')} tensor={failure.get('tensor_name')} "
        f"attempt_step={context.get('attempt_step')} receipt={receipt}"
    )


def batch_failure_context(batch: Any, *, attempt_step: int, global_step: int, epoch: int) -> dict[str, Any]:
    rank, world_size = _rank_world()
    return {
        "run_id": os.environ.get("HSM_FORMAL_RUN_ID") or os.environ.get("HSM_FORMAL_LAUNCH_ID"),
        "rank": rank,
        "world_size": world_size,
        "attempt_step": int(attempt_step),
        "last_committed_global_step": int(global_step),
        "epoch": int(epoch),
        "image_id": list(getattr(batch, "image_ids", [])),
        "case_id": list(getattr(batch, "case_ids", [])),
        "dataset_id": list(getattr(batch, "dataset_ids", [])),
        "modality": list(getattr(batch, "modalities", [])),
    }


def checkpoint_numerical_validation(payload: dict[str, Any], *, require_scaler: bool) -> dict[str, Any]:
    named_values: dict[str, Any] = {
        "model_state": payload.get("model_state_dict"),
        "semantic_branch_state": payload.get("semantic_branch_state_dict"),
        "mask_regressor_state": payload.get("mask_regressor_state_dict"),
        "optimizer_state": payload.get("optimizer_state_dict"),
        "scaler_state": payload.get("scaler_state_dict"),
    }
    scaler_state = payload.get("scaler_state_dict")
    if isinstance(scaler_state, dict) and "scale" in scaler_state:
        named_values["scaler_scale"] = scaler_state["scale"]
    failures: list[dict[str, Any]] = []
    for root_name, root_value in named_values.items():
        if root_value is None:
            if root_name in {"mask_regressor_state", "scaler_state"} and not require_scaler:
                continue
            failures.append({"tensor_name": root_name, "reason": "missing"})
            continue
        failure = first_invalid_value({root_name: root_value})
        if failure is not None:
            failures.append(failure)
    scale = scaler_state.get("scale") if isinstance(scaler_state, dict) else None
    if require_scaler and (scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0):
        failures.append({"tensor_name": "scaler_scale", "reason": "missing_or_non_positive"})
    sampler = payload.get("sampler_state")
    step = int(payload.get("step", -1))
    successful_optimizer_steps = int(payload.get("successful_optimizer_steps", -1))
    attempt_step = int(payload.get("attempt_step", -1))
    step_semantics_consistent = successful_optimizer_steps == step and attempt_step >= successful_optimizer_steps
    if not step_semantics_consistent:
        failures.append({
            "tensor_name": "step_state",
            "reason": "successful_optimizer_steps_or_attempt_step_inconsistent",
            "step": step,
            "successful_optimizer_steps": successful_optimizer_steps,
            "attempt_step": attempt_step,
        })
    sampler_consistent = (
        isinstance(sampler, dict)
        and "next_batch_index" in sampler
        and int(sampler.get("next_global_step", -1)) == step + 1
    )
    if not sampler_consistent:
        failures.append({"tensor_name": "sampler_state", "reason": "global_step_cursor_inconsistent"})
    checks = {
        "MODEL_FINITE": not any(item.get("tensor_name", "").startswith("model_state") for item in failures),
        "SEMANTIC_BRANCH_FINITE": not any(item.get("tensor_name", "").startswith("semantic_branch_state") for item in failures),
        "MASK_REGRESSOR_FINITE": not any(item.get("tensor_name", "").startswith("mask_regressor_state") for item in failures),
        "OPTIMIZER_STATE_FINITE": not any(item.get("tensor_name", "").startswith("optimizer_state") for item in failures),
        "SCALER_STATE_FINITE": not any(item.get("tensor_name", "").startswith("scaler_state") for item in failures),
        "SCALER_SCALE_GT_ZERO": not any(item.get("tensor_name", "") == "scaler_scale" for item in failures),
        "STEP_SEMANTICS_CONSISTENT": step_semantics_consistent,
        "GLOBAL_STEP_CURSOR_CONSISTENT": sampler_consistent,
    }
    state_contract_failed = not step_semantics_consistent or not sampler_consistent
    status = (
        "PASS"
        if not failures
        else "INVALID_STATE_CONTRACT" if state_contract_failed else "INVALID_NUMERICAL"
    )
    return {
        "passed": not failures,
        "status": status,
        "checks": checks,
        "failures": failures,
    }


__all__ = [
    "FormalNumericalFailure",
    "batch_failure_context",
    "aggregate_has_invalid_value",
    "checkpoint_numerical_validation",
    "coordinate_failure",
    "first_invalid_value",
    "module_gradient_values",
    "module_parameter_values",
    "model_state_at_failure",
    "optimizer_state_values",
    "raise_coordinated_failure",
    "scaler_values",
    "scaler_found_inf_after_unscale",
    "should_recover_amp_overflow",
]
