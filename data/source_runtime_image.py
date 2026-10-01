"""Frozen-contract reconstruction of E1 canonical tensors from source images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breast_pretrain.data.image_loading import load_image_to_uint8_rgb
from breast_pretrain.data.runtime_canonical_cache import RuntimeCanonicalCache
from breast_pretrain.data.transforms.stage1_transform_spec import (
    apply_stage1_spatial_transform_with_geometry,
    build_stage1_transform_spec,
    stage1_transform_geometry_checksum,
    stage1_transform_spec_checksum,
)
from breast_pretrain.data_entry.release_common import sha256_file


CANONICAL_IMAGE_MODE_MATERIALIZED = "materialized_npy"
CANONICAL_IMAGE_MODE_SOURCE_RUNTIME = "source_runtime"
CANONICAL_IMAGE_MODES = {CANONICAL_IMAGE_MODE_MATERIALIZED, CANONICAL_IMAGE_MODE_SOURCE_RUNTIME}
DECODE_CONTRACT_VERSION = "stage1_image_loading_rgb_uint8_v1"
RECONSTRUCTION_IMPLEMENTATION_VERSION = "source_runtime_canonical_reconstruction_v1"


class SourceRuntimeContractError(ValueError):
    """A formal source/runtime lineage or canonical-output contract failure."""


@dataclass(frozen=True)
class SourceIntegrityMemoEntry:
    """Worker-local proof that a specific source file already passed SHA verification."""

    resolved_source_path: str
    file_size: int
    mtime_ns: int
    inode: int | None
    verified_source_sha256: str


SourceIntegrityMemo = dict[str, SourceIntegrityMemoEntry]


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def decode_contract() -> dict[str, str]:
    payload = {
        "version": DECODE_CONTRACT_VERSION,
        "dicom": "image_loading.load_dicom_pil_image:voi_lut_uint8_monochrome1_rgb",
        "raster": "image_loading.load_image_to_uint8_rgb:PIL_RGB",
        "output": "uint8_hwc_rgb_then_float32_div255_chw",
    }
    return {"decode_contract_version": DECODE_CONTRACT_VERSION, "decode_contract_sha256": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()}


def resolve_canonical_image_mode(value: Any) -> str:
    mode = str(value or CANONICAL_IMAGE_MODE_MATERIALIZED).strip().lower()
    if mode not in CANONICAL_IMAGE_MODES:
        raise SourceRuntimeContractError(f"Unsupported canonical_image_mode={value!r}.")
    return mode


def canonical_output_contract() -> dict[str, Any]:
    return {
        "canonical_output_dtype": "float32", "canonical_output_layout": "CHW",
        "canonical_output_channels": 3, "canonical_output_value_range": [0.0, 1.0],
    }


def canonical_reconstruction_key(*, source_image_sha256: str, decode_contract_sha256: str, transform_spec_checksum: str, transform_geometry_checksum: str, output_contract: dict[str, Any] | None = None, implementation_version: str = RECONSTRUCTION_IMPLEMENTATION_VERSION) -> str:
    output = output_contract or canonical_output_contract()
    payload = {
        "source_image_sha256": str(source_image_sha256), "decode_contract_sha256": str(decode_contract_sha256),
        "transform_spec_checksum": str(transform_spec_checksum), "transform_geometry_checksum": str(transform_geometry_checksum),
        **output, "reconstruction_implementation_version": str(implementation_version),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_source_runtime_row(row: dict[str, Any], *, expected_mode: str = "auto") -> None:
    mode = resolve_canonical_image_mode(row.get("canonical_image_mode"))
    if expected_mode != "auto" and mode != resolve_canonical_image_mode(expected_mode):
        raise SourceRuntimeContractError(f"E1 canonical mode {mode} does not match expected_mode={expected_mode}.")
    if mode != CANONICAL_IMAGE_MODE_SOURCE_RUNTIME:
        return
    required = (
        "source_image_path", "source_image_sha256", "source_height", "source_width", "decode_contract_version",
        "decode_contract_sha256", "target_height", "target_width", "transform_policy", "transform_spec_checksum",
        "resize_geometry", "transform_geometry_checksum", "canonical_reconstruction_key", "patch_size_h",
        "patch_size_w", "grid_h", "grid_w", "patch_count", "patch_token_order_version", "valid_content_mask_ref",
        "image_preprocessing_contract_sha256",
    )
    missing = [field for field in required if row.get(field) in (None, "")]
    if missing:
        raise SourceRuntimeContractError("source_runtime row is missing required fields: " + ", ".join(missing))
    if row.get("canonical_image_path") not in (None, "") or row.get("canonical_image_sha256") not in (None, ""):
        raise SourceRuntimeContractError("source_runtime row must not declare a canonical NPY path or hash.")
    expected_decode = decode_contract()
    if row.get("decode_contract_version") != expected_decode["decode_contract_version"] or row.get("decode_contract_sha256") != expected_decode["decode_contract_sha256"]:
        raise SourceRuntimeContractError("source_runtime decode contract differs from the registered image_loading contract.")
    output = canonical_output_contract()
    for field, value in output.items():
        observed = row.get(field)
        if field == "canonical_output_channels":
            observed = int(observed)
        elif field == "canonical_output_value_range" and isinstance(observed, str):
            observed = json.loads(observed)
        if observed != value:
            raise SourceRuntimeContractError(f"source_runtime canonical output contract differs on {field}.")
    if str(row.get("patch_token_order_version")) != "row_major_h_w_v1":
        raise SourceRuntimeContractError("source_runtime requires patch_token_order_version=row_major_h_w_v1.")
    if int(row["patch_count"]) != int(row["grid_h"]) * int(row["grid_w"]):
        raise SourceRuntimeContractError("source_runtime patch_count does not equal grid_h * grid_w.")
    expected_key = canonical_reconstruction_key(
        source_image_sha256=str(row["source_image_sha256"]), decode_contract_sha256=str(row["decode_contract_sha256"]),
        transform_spec_checksum=str(row["transform_spec_checksum"]), transform_geometry_checksum=str(row["transform_geometry_checksum"]),
    )
    if str(row["canonical_reconstruction_key"]) != expected_key:
        raise SourceRuntimeContractError("source_runtime canonical_reconstruction_key is invalid.")


def validate_source_image_integrity(row: dict[str, Any], *, memo: SourceIntegrityMemo | None = None) -> Path:
    path = Path(str(row.get("source_image_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source_runtime source image is missing: {path}")
    stat = path.stat()
    file_size = int(stat.st_size)
    mtime_ns = int(stat.st_mtime_ns)
    inode = getattr(stat, "st_ino", None)
    key = str(row.get("canonical_reconstruction_key") or "")
    cached = memo.get(key) if memo is not None else None
    cache_hit = (
        cached is not None
        and cached.resolved_source_path == str(path)
        and cached.file_size == file_size
        and cached.mtime_ns == mtime_ns
        and cached.inode == inode
    )
    observed = cached.verified_source_sha256 if cache_hit else sha256_file(path)
    if observed != str(row.get("source_image_sha256") or "").strip().lower():
        raise SourceRuntimeContractError("source_runtime source_image_sha256 does not match the source file.")
    if memo is not None:
        memo[key] = SourceIntegrityMemoEntry(
            resolved_source_path=str(path),
            file_size=file_size,
            mtime_ns=mtime_ns,
            inode=inode,
            verified_source_sha256=observed,
        )
    return path


def _raw_tensor_from_source(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    array = load_image_to_uint8_rgb(path)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise SourceRuntimeContractError(f"Canonical image loader returned invalid RGB shape {array.shape} for {path}.")
    height, width = int(array.shape[0]), int(array.shape[1])
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(dtype=torch.float32).div_(255.0)
    return tensor, (height, width)


def _spec_and_geometry(row: dict[str, Any]) -> tuple[Any, dict[str, object]]:
    spec = build_stage1_transform_spec(
        modality=str(row.get("modality") or ""), image_size=(int(row["target_height"]), int(row["target_width"])),
        patch_size=int(row["patch_size_h"]), policy=str(row["transform_policy"]),
    )
    geometry = row["resize_geometry"]
    if isinstance(geometry, str):
        try:
            geometry = json.loads(geometry)
        except json.JSONDecodeError as exc:
            raise SourceRuntimeContractError("source_runtime resize_geometry is not valid frozen JSON.") from exc
    if not isinstance(geometry, dict):
        raise SourceRuntimeContractError("source_runtime resize_geometry must be an E1-frozen object.")
    if stage1_transform_spec_checksum(spec) != str(row["transform_spec_checksum"]):
        raise SourceRuntimeContractError("source_runtime transform_spec_checksum differs from the frozen transform spec.")
    if stage1_transform_geometry_checksum(spec, geometry) != str(row["transform_geometry_checksum"]):
        raise SourceRuntimeContractError("source_runtime transform_geometry_checksum differs from E1 frozen geometry.")
    return spec, geometry


def validate_canonical_tensor(tensor: torch.Tensor, row: dict[str, Any]) -> None:
    if tensor.dtype != torch.float32 or tensor.ndim != 3:
        raise SourceRuntimeContractError("Canonical tensor must be float32 CHW.")
    expected_shape = (3, int(row["target_height"]), int(row["target_width"]))
    if tuple(tensor.shape) != expected_shape:
        raise SourceRuntimeContractError(f"Canonical tensor shape {tuple(tensor.shape)} differs from {expected_shape}.")
    if not torch.isfinite(tensor).all() or float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise SourceRuntimeContractError("Canonical tensor values must be finite float32 in [0, 1].")


def reconstruct_source_runtime_canonical(row: dict[str, Any], *, expected_mode: str = "auto", verified_source_path: Path | None = None, source_integrity_memo: SourceIntegrityMemo | None = None) -> tuple[torch.Tensor, dict[str, object]]:
    validate_source_runtime_row(row, expected_mode=expected_mode)
    source_path = verified_source_path or validate_source_image_integrity(row, memo=source_integrity_memo)
    raw, source_hw = _raw_tensor_from_source(source_path)
    if source_hw != (int(row["source_height"]), int(row["source_width"])):
        raise SourceRuntimeContractError("source_runtime decoded source H/W differs from E1 frozen source dimensions.")
    spec, geometry = _spec_and_geometry(row)
    tensor = apply_stage1_spatial_transform_with_geometry(raw.unsqueeze(0), spec, geometry).squeeze(0).contiguous()
    validate_canonical_tensor(tensor, row)
    return tensor, {"transform_policy": str(row["transform_policy"]), "transform_spec_checksum": str(row["transform_spec_checksum"]), "transform_geometry": geometry, "transform_geometry_checksum": str(row["transform_geometry_checksum"])}


def load_source_runtime_canonical(row: dict[str, Any], *, cache: RuntimeCanonicalCache | None, expected_mode: str = "auto", source_integrity_memo: SourceIntegrityMemo | None = None) -> tuple[torch.Tensor, dict[str, object]]:
    validate_source_runtime_row(row, expected_mode=expected_mode)
    source_path = validate_source_image_integrity(row, memo=source_integrity_memo)
    spec, geometry = _spec_and_geometry(row)
    metadata = {"transform_policy": str(row["transform_policy"]), "transform_spec_checksum": str(row["transform_spec_checksum"]), "transform_geometry": geometry, "transform_geometry_checksum": str(row["transform_geometry_checksum"])}
    expected = {"canonical_reconstruction_key": str(row["canonical_reconstruction_key"]), "source_image_sha256": str(row["source_image_sha256"]), "decode_contract_sha256": str(row["decode_contract_sha256"]), "transform_spec_checksum": str(row["transform_spec_checksum"]), "transform_geometry_checksum": str(row["transform_geometry_checksum"]), "canonical_output_contract": canonical_output_contract(), "implementation_version": RECONSTRUCTION_IMPLEMENTATION_VERSION}
    if cache is not None:
        cached = cache.get(str(row["canonical_reconstruction_key"]), expected)
        if cached is not None:
            validate_canonical_tensor(cached, row)
            return cached, metadata
        cache.stats.source_reconstructions += 1
    tensor, _ = reconstruct_source_runtime_canonical(row, expected_mode=expected_mode, verified_source_path=source_path)
    if cache is not None:
        cache.put(str(row["canonical_reconstruction_key"]), tensor, expected)
    return tensor, metadata


__all__ = [
    "CANONICAL_IMAGE_MODE_MATERIALIZED", "CANONICAL_IMAGE_MODE_SOURCE_RUNTIME", "CANONICAL_IMAGE_MODES",
    "DECODE_CONTRACT_VERSION", "RECONSTRUCTION_IMPLEMENTATION_VERSION", "SourceIntegrityMemo",
    "SourceIntegrityMemoEntry", "SourceRuntimeContractError",
    "canonical_output_contract", "canonical_reconstruction_key", "decode_contract", "load_source_runtime_canonical",
    "reconstruct_source_runtime_canonical", "resolve_canonical_image_mode", "validate_canonical_tensor",
    "validate_source_image_integrity", "validate_source_runtime_row",
]
