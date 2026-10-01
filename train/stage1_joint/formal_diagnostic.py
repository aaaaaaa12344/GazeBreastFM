from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch


_PER_SAMPLE_BATCH_TENSORS = {
    "input_image", "input_patch_gaze_weight", "input_high_conf_patch_prior",
    "input_valid_content_patch_mask", "dynamic_gaze", "dynamic_high_conf", "patch_mask",
    "visible_patch_mask", "visible_spatial_mask", "clean_encoder_output",
    "clean_pooled_global_feature", "masked_image", "masked_encoder_output", "context_patches",
    "predicted_patch_tokens", "reconstruction_prediction", "reconstruction_target",
}


def diagnostic_start_attempt() -> int | None:
    value = os.environ.get("HSM_STAGE1_DIAGNOSTIC_START_ATTEMPT", "").strip()
    if not value:
        return None
    start = int(value)
    if start < 1:
        raise ValueError("HSM_STAGE1_DIAGNOSTIC_START_ATTEMPT must be a positive integer.")
    return start


def diagnostic_enabled(attempt_step: int) -> bool:
    start = diagnostic_start_attempt()
    return start is not None and int(attempt_step) >= start


def gradient_aggregate_stats(named_gradients: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Diagnostic-only aggregate gradient observability without tensor dumps."""
    gradients = [value.detach().float() for value in named_gradients.values()]
    if not gradients:
        return {"gradient_finite": True, "gradient_absmax": 0.0, "gradient_norm": 0.0}
    finite_flags = [torch.isfinite(value).all() for value in gradients]
    if not bool(torch.stack(finite_flags).all().item()):
        return {"gradient_finite": False, "gradient_absmax": None, "gradient_norm": None}
    absmaxes = [value.abs().amax() for value in gradients]
    squared_sums = [(value * value).sum() for value in gradients]
    return {
        "gradient_finite": True,
        "gradient_absmax": float(torch.stack(absmaxes).amax().item()),
        "gradient_norm": float(torch.stack(squared_sums).sum().sqrt().item()),
    }


def _multi_index(flat_index: int, shape: list[int]) -> list[int]:
    result: list[int] = []
    remaining = int(flat_index)
    for size in reversed(shape):
        result.append(remaining % size)
        remaining //= size
    return list(reversed(result))


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    numeric = detached.is_floating_point() or detached.is_complex()
    finite = torch.isfinite(detached) if numeric else torch.ones_like(detached, dtype=torch.bool)
    nonfinite = ~finite
    finite_values = detached[finite].float()
    nonfinite_count = int(nonfinite.sum().item())
    shape = list(detached.shape)
    first_flat = None
    first_multi = None
    if nonfinite_count:
        first_flat = int(torch.nonzero(nonfinite.reshape(-1), as_tuple=False)[0].item())
        first_multi = _multi_index(first_flat, shape)
    result: dict[str, Any] = {
        "dtype": str(value.dtype), "shape": shape, "numel": int(detached.numel()),
        "isfinite": nonfinite_count == 0, "finite_count": int(finite.sum().item()),
        "nonfinite_count": nonfinite_count,
        "nan_count": int(torch.isnan(detached).sum().item()) if numeric else 0,
        "posinf_count": int(torch.isposinf(detached).sum().item()) if numeric else 0,
        "neginf_count": int(torch.isneginf(detached).sum().item()) if numeric else 0,
        "first_nonfinite_flat_index": first_flat,
        "first_nonfinite_multi_index": first_multi,
    }
    if finite_values.numel():
        result.update({
            "min": float(finite_values.min().item()), "max": float(finite_values.max().item()),
            "absmax": float(finite_values.abs().max().item()), "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
        })
    else:
        result.update({"min": None, "max": None, "absmax": None, "mean": None, "std": None})
    return result


def _per_sample_summary(value: torch.Tensor, context: dict[str, Any]) -> list[dict[str, Any]] | None:
    image_ids = context.get("image_id", [])
    if value.ndim < 1 or int(value.shape[0]) != len(image_ids):
        return None
    result: list[dict[str, Any]] = []
    for batch_index in range(int(value.shape[0])):
        tensor = value[batch_index].detach()
        numeric = tensor.is_floating_point() or tensor.is_complex()
        finite = torch.isfinite(tensor) if numeric else torch.ones_like(tensor, dtype=torch.bool)
        finite_values = tensor[finite].float()
        result.append({
            "batch_index": batch_index, "image_id": image_ids[batch_index],
            "case_id": context.get("case_id", [None] * len(image_ids))[batch_index],
            "dataset_id": context.get("dataset_id", [None] * len(image_ids))[batch_index],
            "modality": context.get("modality", [None] * len(image_ids))[batch_index],
            "finite_count": int(finite.sum().item()), "nonfinite_count": int((~finite).sum().item()),
            "nan_count": int(torch.isnan(tensor).sum().item()) if numeric else 0,
            "posinf_count": int(torch.isposinf(tensor).sum().item()) if numeric else 0,
            "neginf_count": int(torch.isneginf(tensor).sum().item()) if numeric else 0,
            "finite_absmax": float(finite_values.abs().max().item()) if finite_values.numel() else None,
        })
    return result


def _summary(name: str, value: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        scalar = float(value)
        finite = bool(torch.isfinite(torch.tensor(scalar)))
        return {
            "dtype": type(value).__name__, "shape": [], "numel": 1, "isfinite": finite,
            "finite_count": int(finite), "nonfinite_count": int(not finite),
            "nan_count": int(scalar != scalar), "posinf_count": int(scalar == float("inf")),
            "neginf_count": int(scalar == float("-inf")),
            "first_nonfinite_flat_index": None if finite else 0,
            "first_nonfinite_multi_index": None if finite else [],
            "min": scalar if finite else None, "max": scalar if finite else None,
            "absmax": abs(scalar) if finite else None, "mean": scalar if finite else None,
            "std": 0.0 if finite else None,
        }
    result = _tensor_summary(value)
    if name in _PER_SAMPLE_BATCH_TENSORS:
        per_sample = _per_sample_summary(value, context)
        if per_sample is not None:
            result["per_sample"] = per_sample
    return result


def _iter_diagnostic_leaves(name: str, value: Any) -> Iterable[tuple[str, Any, bool]]:
    """Flatten nested diagnostic values without coercing metadata to numerics."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_diagnostic_leaves(f"{name}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_diagnostic_leaves(f"{name}[{index}]", item)
    elif isinstance(value, torch.Tensor) or (
        isinstance(value, (float, int)) and not isinstance(value, bool)
    ):
        yield name, value, True
    else:
        yield name, value, False


def runtime_fingerprint() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda_available else None
    properties = torch.cuda.get_device_properties(device_index) if cuda_available else None
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_enabled = bool(torch.is_autocast_enabled())
    try:
        autocast_dtype = str(torch.get_autocast_dtype("cuda"))
    except (AttributeError, TypeError):
        autocast_dtype = str(torch.get_autocast_gpu_dtype()) if cuda_available else None
    try:
        sdpa = {"status": "available", "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math": bool(torch.backends.cuda.math_sdp_enabled())}
    except (AttributeError, RuntimeError):
        sdpa = {"status": "unavailable"}
    return {
        "rank": int(os.environ.get("RANK", "0")), "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")), "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda, "cuda_device_index": device_index,
        "gpu_name": properties.name if properties is not None else None,
        "gpu_total_memory": int(properties.total_memory) if properties is not None else None,
        "gpu_uuid": None, "driver_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "autocast_enabled": autocast_enabled, "autocast_dtype": autocast_dtype,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if cuda_available else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if cuda_available else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "sdpa": sdpa,
    }


def _json_safe_internal_summary(value: Any) -> Any:
    """Prevent internal-hook diagnostics from retaining or serializing tensors."""
    if isinstance(value, torch.Tensor):
        return {"unsupported_internal_value_type": "Tensor"}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_internal_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_internal_summary(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"unsupported_internal_value_type": type(value).__name__}


def write_diagnostic_summary(
    *,
    output_dir: Path,
    context: dict[str, Any],
    tensors: dict[str, Any],
    internal_module_summaries: dict[str, Any] | None = None,
) -> Path | None:
    if not diagnostic_enabled(int(context["attempt_step"])):
        return None
    rank = int(context["rank"])
    path = output_dir / "numerical_diagnostics" / f"attempt_{int(context['attempt_step']):06d}_rank_{rank:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_summaries: dict[str, dict[str, Any]] = {}
    non_numeric_diagnostics: list[dict[str, str]] = []
    for name, value in tensors.items():
        for leaf_name, leaf_value, is_numeric in _iter_diagnostic_leaves(name, value):
            if is_numeric:
                tensor_summaries[leaf_name] = _summary(leaf_name, leaf_value, context)
            else:
                non_numeric_diagnostics.append({
                    "path": leaf_name,
                    "python_type": type(leaf_value).__name__,
                })
    payload = {
        "schema_version": "formal_stage1_bounded_replay_diagnostic_v2",
        "serializer_revision": "v2.1_nested_numeric_internal_modules",
        "diagnostic_only": True, **context, "runtime_fingerprint": runtime_fingerprint(),
        "tensors": tensor_summaries,
        "non_numeric_diagnostics": non_numeric_diagnostics,
        "internal_module_summaries": _json_safe_internal_summary(internal_module_summaries or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


__all__ = ["diagnostic_enabled", "diagnostic_start_attempt", "gradient_aggregate_stats", "runtime_fingerprint", "write_diagnostic_summary"]
