from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND = "mammo_fm_timm_efficientnet_b5"
DEFAULT_MAMMO_FM_TIMM_MODEL_NAME = "tf_efficientnet_b5"

_SOURCE_PREFIX = "image_encoder."
_BLOCK_RE = re.compile(r"^_blocks\.(\d+)\.(.+)$")
_BN_SUFFIXES = {"weight", "bias", "running_mean", "running_var", "num_batches_tracked"}


@dataclass(frozen=True)
class MammoFmTimmTranslationResult:
    translated_state_dict: dict[str, torch.Tensor]
    mapping_report: dict[str, object]


def _flatten_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        state = (
            payload.get("vision_encoder_state_dict")
            or payload.get("model_state_dict")
            or payload.get("state_dict")
            or payload.get("model")
            or payload
        )
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError("Mammo-FM checkpoint payload must contain a state_dict mapping.")
    return {str(key): value for key, value in state.items() if isinstance(value, torch.Tensor)}


def extract_mammo_fm_image_encoder_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    tensors = _flatten_state_dict(payload)
    image_encoder: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        normalized = key[len("module.") :] if key.startswith("module.") else key
        if normalized.startswith(_SOURCE_PREFIX):
            image_encoder[normalized[len(_SOURCE_PREFIX) :]] = value
    if not image_encoder:
        raise ValueError("Mammo-FM checkpoint contains no image_encoder.* tensor weights.")
    return image_encoder


def extract_mammo_fm_image_projection_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    tensors = _flatten_state_dict(payload)
    projection: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        normalized = key[len("module.") :] if key.startswith("module.") else key
        if normalized.startswith("image_projection."):
            projection[normalized[len("image_projection.") :]] = value
    return projection


def build_timm_efficientnet_flat_block_map(timm_model: nn.Module) -> dict[int, str]:
    if not hasattr(timm_model, "blocks"):
        raise ValueError("Target timm model has no EfficientNet-style blocks module.")
    block_map: dict[int, str] = {}
    flat_index = 0
    for stage_index, stage in enumerate(timm_model.blocks):
        for block_index, _block in enumerate(stage):
            block_map[flat_index] = f"blocks.{stage_index}.{block_index}"
            flat_index += 1
    if not block_map:
        raise ValueError("Target timm model has no traversable EfficientNet blocks.")
    return block_map


def _map_top_level_key(src_key: str) -> str | None:
    replacements = (
        ("_conv_stem.", "conv_stem."),
        ("_conv_head.", "conv_head."),
        ("_bn0.", "bn1."),
        ("_bn1.", "bn2."),
    )
    for src_prefix, dst_prefix in replacements:
        if src_key.startswith(src_prefix):
            return dst_prefix + src_key[len(src_prefix) :]
    return None


def _is_inverted_residual_block(dst_block_prefix: str, target_state_dict: dict[str, torch.Tensor]) -> bool:
    return f"{dst_block_prefix}.conv_pwl.weight" in target_state_dict


def _map_block_bn_key(
    inner_key: str,
    *,
    dst_block_prefix: str,
    target_state_dict: dict[str, torch.Tensor],
) -> str | None:
    if "." not in inner_key:
        return None
    src_bn_name, suffix = inner_key.split(".", 1)
    if suffix not in _BN_SUFFIXES:
        return None
    inverted_residual = _is_inverted_residual_block(dst_block_prefix, target_state_dict)
    if inverted_residual:
        bn_map = {"_bn0": "bn1", "_bn1": "bn2", "_bn2": "bn3"}
    else:
        bn_map = {"_bn0": "bn1", "_bn1": "bn1", "_bn2": "bn2"}
    dst_bn_name = bn_map.get(src_bn_name)
    if dst_bn_name is None:
        return None
    candidate = f"{dst_block_prefix}.{dst_bn_name}.{suffix}"
    if candidate not in target_state_dict:
        return None
    return candidate


def _map_block_inner_key(
    inner_key: str,
    *,
    dst_block_prefix: str,
    target_state_dict: dict[str, torch.Tensor],
) -> str | None:
    bn_key = _map_block_bn_key(
        inner_key,
        dst_block_prefix=dst_block_prefix,
        target_state_dict=target_state_dict,
    )
    if bn_key is not None:
        return bn_key
    replacements = (
        ("_depthwise_conv.", "conv_dw."),
        ("_expand_conv.", "conv_pw."),
        ("_se_reduce.", "se.conv_reduce."),
        ("_se_expand.", "se.conv_expand."),
    )
    for src_prefix, dst_prefix in replacements:
        if inner_key.startswith(src_prefix):
            return f"{dst_block_prefix}.{dst_prefix}{inner_key[len(src_prefix):]}"
    if inner_key.startswith("_project_conv."):
        return None
    return None


def _resolve_project_conv_key(
    *,
    inner_key: str,
    dst_block_prefix: str,
    target_state_dict: dict[str, torch.Tensor],
) -> str | None:
    suffix = inner_key[len("_project_conv.") :]
    for dst_name in ("conv_pwl", "conv_pw"):
        candidate = f"{dst_block_prefix}.{dst_name}.{suffix}"
        if candidate in target_state_dict:
            return candidate
    return None


def _map_mammo_fm_key(
    src_key: str,
    *,
    flat_block_map: dict[int, str],
    target_state_dict: dict[str, torch.Tensor],
) -> str | None:
    top_level = _map_top_level_key(src_key)
    if top_level is not None:
        return top_level

    block_match = _BLOCK_RE.match(src_key)
    if block_match is None:
        return None

    flat_index = int(block_match.group(1))
    inner_key = block_match.group(2)
    dst_block_prefix = flat_block_map.get(flat_index)
    if dst_block_prefix is None:
        return None
    if inner_key.startswith("_project_conv."):
        return _resolve_project_conv_key(
            inner_key=inner_key,
            dst_block_prefix=dst_block_prefix,
            target_state_dict=target_state_dict,
        )
    return _map_block_inner_key(
        inner_key,
        dst_block_prefix=dst_block_prefix,
        target_state_dict=target_state_dict,
    )


def _load_key_report(
    *,
    translated_state_dict: dict[str, torch.Tensor],
    target_model: nn.Module,
) -> tuple[list[str], list[str]]:
    target_state = target_model.state_dict()
    missing = sorted(key for key in target_state if key not in translated_state_dict)
    unexpected = sorted(key for key in translated_state_dict if key not in target_state)
    return missing, unexpected


def _is_allowed_num_batches_tracked_only(keys: list[str]) -> bool:
    return all(key.endswith(".num_batches_tracked") for key in keys)


def translate_mammo_fm_image_encoder_to_timm_efficientnet(
    mammo_fm_state_dict: Any,
    timm_model: nn.Module,
) -> MammoFmTimmTranslationResult:
    source_state = extract_mammo_fm_image_encoder_state_dict(mammo_fm_state_dict)
    target_state = timm_model.state_dict()
    flat_block_map = build_timm_efficientnet_flat_block_map(timm_model)

    translated: dict[str, torch.Tensor] = {}
    untranslated_src: list[str] = []
    duplicate_dst: list[str] = []
    shape_mismatch: list[dict[str, object]] = []
    no_shape_match: list[dict[str, object]] = []

    for src_key, tensor in sorted(source_state.items()):
        dst_key = _map_mammo_fm_key(
            src_key,
            flat_block_map=flat_block_map,
            target_state_dict=target_state,
        )
        if dst_key is None:
            untranslated_src.append(src_key)
            continue
        if dst_key in translated:
            duplicate_dst.append(dst_key)
            continue
        dst_tensor = target_state.get(dst_key)
        if dst_tensor is None:
            untranslated_src.append(src_key)
            no_shape_match.append(
                {
                    "src_key": src_key,
                    "dst_key": dst_key,
                    "src_shape": list(tensor.shape),
                    "dst_shape": None,
                }
            )
            continue
        if tuple(tensor.shape) != tuple(dst_tensor.shape):
            shape_mismatch.append(
                {
                    "src_key": src_key,
                    "dst_key": dst_key,
                    "src_shape": list(tensor.shape),
                    "dst_shape": list(dst_tensor.shape),
                }
            )
            continue
        translated[dst_key] = tensor

    missing_dst, unexpected_after_load = _load_key_report(
        translated_state_dict=translated,
        target_model=timm_model,
    )
    loaded_ratio = float(len(translated) / len(target_state)) if target_state else 0.0
    report = {
        "src_count": int(len(source_state)),
        "dst_count": int(len(target_state)),
        "translated_count": int(len(translated)),
        "untranslated_src": untranslated_src,
        "missing_dst": missing_dst,
        "missing_after_load": missing_dst,
        "unexpected_after_load": unexpected_after_load,
        "shape_mismatch": shape_mismatch,
        "no_shape_match": no_shape_match,
        "duplicate_dst": sorted(set(duplicate_dst)),
        "loaded_ratio": loaded_ratio,
        "flat_block_count": int(len(flat_block_map)),
        "flat_block_map": {str(key): value for key, value in flat_block_map.items()},
        "allowed_missing_num_batches_tracked_only": _is_allowed_num_batches_tracked_only(missing_dst),
    }
    return MammoFmTimmTranslationResult(translated_state_dict=translated, mapping_report=report)


def load_translated_mammo_fm_into_timm_efficientnet(
    *,
    mammo_fm_state_dict: Any,
    timm_model: nn.Module,
    strict: bool = True,
    min_loaded_ratio: float = 0.99,
) -> MammoFmTimmTranslationResult:
    result = translate_mammo_fm_image_encoder_to_timm_efficientnet(mammo_fm_state_dict, timm_model)
    report = result.mapping_report
    fatal_reasons = []
    if report["no_shape_match"]:
        fatal_reasons.append("no_shape_match is not empty")
    if report["shape_mismatch"]:
        fatal_reasons.append("shape_mismatch is not empty")
    if report["unexpected_after_load"]:
        fatal_reasons.append("unexpected_after_load is not empty")
    missing_after_load = list(report["missing_after_load"])
    if missing_after_load and not _is_allowed_num_batches_tracked_only(missing_after_load):
        fatal_reasons.append("missing_after_load contains keys other than num_batches_tracked")
    if float(report["loaded_ratio"]) < float(min_loaded_ratio):
        fatal_reasons.append(f"loaded_ratio {report['loaded_ratio']:.6f} < {min_loaded_ratio:.6f}")
    if strict and fatal_reasons:
        raise ValueError("Mammo-FM to timm EfficientNet key translation failed: " + "; ".join(fatal_reasons))

    load_result = timm_model.load_state_dict(result.translated_state_dict, strict=False)
    after_missing = list(load_result.missing_keys)
    after_unexpected = list(load_result.unexpected_keys)
    report["missing_after_load"] = after_missing
    report["unexpected_after_load"] = after_unexpected
    report["strict_load_compatible"] = not after_missing and not after_unexpected
    report["near_strict_load_compatible"] = (
        not after_unexpected and (not after_missing or _is_allowed_num_batches_tracked_only(after_missing))
    )
    if strict and (after_unexpected or (after_missing and not _is_allowed_num_batches_tracked_only(after_missing))):
        raise ValueError(
            "Translated Mammo-FM state_dict did not load near-strictly: "
            f"missing={after_missing}, unexpected={after_unexpected}"
        )
    return result


def build_mammo_fm_projection(payload: Any) -> tuple[nn.Module | None, dict[str, object]]:
    projection_state = extract_mammo_fm_image_projection_state_dict(payload)
    if not projection_state:
        return None, {"available": False, "loaded": False, "reason": "no image_projection.* tensors"}
    weight = projection_state.get("weight")
    bias = projection_state.get("bias")
    if weight is None or weight.ndim != 2:
        return None, {"available": True, "loaded": False, "reason": "unsupported image_projection state_dict"}
    projection = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
    load_result = projection.load_state_dict(projection_state, strict=False)
    return projection, {
        "available": True,
        "loaded": not load_result.missing_keys and not load_result.unexpected_keys,
        "input_dim": int(weight.shape[1]),
        "output_dim": int(weight.shape[0]),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


__all__ = [
    "DEFAULT_MAMMO_FM_TIMM_MODEL_NAME",
    "MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND",
    "MammoFmTimmTranslationResult",
    "build_mammo_fm_projection",
    "build_timm_efficientnet_flat_block_map",
    "extract_mammo_fm_image_encoder_state_dict",
    "extract_mammo_fm_image_projection_state_dict",
    "load_translated_mammo_fm_into_timm_efficientnet",
    "translate_mammo_fm_image_encoder_to_timm_efficientnet",
]
