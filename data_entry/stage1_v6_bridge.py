"""Read-only Dataset Entry V2 sidecar bridge for an existing V6 Stage 1 bundle.

It deliberately does not replace the V6 bundle/materializer.  The bridge
copies the already-frozen bundle into a new wrapper and adds immutable E1/E2
references that are checked before a V2-enabled formal entry can proceed.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from breast_pretrain.data.stage1_v6_contract import validate_stage1_v6_bundle
from breast_pretrain.data.source_runtime_image import (
    CANONICAL_IMAGE_MODE_SOURCE_RUNTIME,
    reconstruct_source_runtime_canonical,
    resolve_canonical_image_mode,
    validate_source_runtime_row,
)
from breast_pretrain.data_entry.gaze_shards import GazeShardReader
from breast_pretrain.data_entry.mask_shards import MaskShardReader
from breast_pretrain.data_entry.release_common import (
    atomic_write_json,
    atomic_write_jsonl,
    load_pass_receipt,
    read_jsonl,
    sha256_file,
    write_release_receipt,
)


SCHEMA = "stage1_v6_dataset_entry_v2_wrapper_v1"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _image_rows(root: Path, name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / "manifests" / name)
    result = {str(row.get("image_id") or ""): row for row in rows}
    if not result or "" in result or len(result) != len(rows):
        raise ValueError(f"Invalid unique image manifest {root / 'manifests' / name}")
    return result


def materialize_stage1_v6_dataset_entry_v2_wrapper(
    *,
    v6_bundle_root: Path,
    image_release_root: Path,
    gaze_release_root: Path,
    release_root: Path,
    release_id: str,
) -> dict[str, Any]:
    """Create a new wrapper without changing any pre-existing V6 artefact."""
    image_receipt, _ = load_pass_receipt(image_release_root)
    gaze_receipt, _ = load_pass_receipt(gaze_release_root)
    if gaze_receipt.get("upstream_release_ids", {}).get("image") != image_receipt.get("release_id"):
        raise ValueError("E2 receipt is not derived from the provided E1 release.")
    manifest = v6_bundle_root / "manifest_stage1_v6.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Existing V6 bundle lacks manifest_stage1_v6.csv: {v6_bundle_root}")
    if release_root.exists():
        raise FileExistsError(f"Refusing to overwrite Dataset Entry V2 wrapper: {release_root}")
    e1 = _image_rows(image_release_root, "image_preprocessing_manifest.jsonl")
    e2 = _image_rows(gaze_release_root, "gaze_to_patch_projection_manifest.jsonl")
    v6_rows = _csv_rows(manifest)
    sidecars: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for row in v6_rows:
        image_id = str(row.get("image_id") or "").strip()
        image = e1.get(image_id)
        gaze = e2.get(image_id)
        if image is None or gaze is None:
            raise ValueError(f"V6 bundle image_id={image_id!r} has no closed E1/E2 sidecars.")
        if str(row.get("case_id") or "") != str(image.get("case_id") or ""):
            raise ValueError(f"V6/E1 case mismatch for image_id={image_id}")
        valid_ref = image["valid_content_mask_ref"]
        sidecar = {
                "image_id": image_id,
                "canonical_image_mode": image.get("canonical_image_mode", "materialized_npy"),
                "canonical_image_path": image["canonical_image_path"],
                "canonical_image_sha256": image["canonical_image_sha256"],
                "canonical_reconstruction_key": image.get("canonical_reconstruction_key", ""),
                "source_image_path": image.get("source_image_path", ""),
                "source_image_sha256": image.get("source_image_sha256", ""),
                "source_height": image.get("source_height", ""), "source_width": image.get("source_width", ""),
                "decode_contract_version": image.get("decode_contract_version", ""),
                "decode_contract_sha256": image.get("decode_contract_sha256", ""),
                "target_height": image.get("target_height", ""), "target_width": image.get("target_width", ""),
                "transform_policy": image.get("transform_policy", ""),
                "transform_spec_checksum": image.get("transform_spec_checksum", ""),
                "resize_geometry": json.dumps(image.get("resize_geometry", {}), sort_keys=True, separators=(",", ":")),
                "transform_geometry_checksum": image.get("transform_geometry_checksum", ""),
                "canonical_output_dtype": image.get("canonical_output_dtype", "float32"),
                "canonical_output_layout": image.get("canonical_output_layout", "CHW"),
                "canonical_output_channels": image.get("canonical_output_channels", 3),
                "canonical_output_value_range": json.dumps(image.get("canonical_output_value_range", [0.0, 1.0])),
                "image_preprocessing_contract_sha256": image["image_preprocessing_contract_sha256"],
                "image_geometry_key": image["image_geometry_key"],
                "image_geometry_sha256": image["image_geometry_sha256"],
                "patch_geometry_key": image["patch_geometry_key"],
                "patch_geometry_sha256": image["geometry_contract_sha256"],
                "patch_size_h": int(image["patch_size_h"]),
                "patch_size_w": int(image["patch_size_w"]),
                "patch_count": int(image["patch_count"]),
                "patch_token_order_version": image["patch_token_order_version"],
                "grid_h": int(image["grid_h"]),
                "grid_w": int(image["grid_w"]),
                "valid_content_mask_ref": valid_ref,
                "valid_content_mask_shard_key": valid_ref["shard_key"],
                "valid_content_mask_shard_sha256": valid_ref["sha256"],
                "valid_content_mask_offset": int(valid_ref["offset"]),
                "valid_content_mask_length": int(valid_ref["length"]),
                "gaze_training_enabled": int(gaze["gaze_training_enabled"]),
                "gaze_status": gaze["gaze_status"],
                "gaze_weight_shard_key": gaze["gaze_weight_shard_key"],
                "gaze_shard_sha256": gaze["shard_sha256"],
                "gaze_slice_sha256": gaze["slice_sha256"],
                "gaze_offset": gaze["offset"],
                "gaze_length": gaze["length"],
                "projection_version": gaze["projection_version"],
            }
        sidecars.append(sidecar)
        # Keep the legacy V6 row and add only frozen sidecars.  The runtime
        # manifest is deliberately a wrapper asset, never an in-place V6 edit.
        runtime_rows.append({**row, **sidecar, "dataset_entry_v2_enabled": "true"})
    release_root.mkdir(parents=True)
    # A copied wrapper keeps the original V6 freeze byte-for-byte intact and
    # makes a transportable release even when source paths disappear later.
    shutil.copytree(v6_bundle_root, release_root / "v6_bundle")
    atomic_write_jsonl(release_root / "manifests" / "dataset_entry_v2_sidecars.jsonl", sidecars)
    runtime_manifest = release_root / "manifests" / "manifest_stage1_v6_dataset_entry_v2.csv"
    with runtime_manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in runtime_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(runtime_rows)
    atomic_write_json(
        release_root / "release_manifest.json",
        {
            "schema_version": SCHEMA,
            "release_id": release_id,
            "dataset_id": image_receipt["dataset_id"],
            "upstream_release_ids": {"image": image_receipt["release_id"], "gaze": gaze_receipt["release_id"]},
            "v6_bundle_source_sha256": sha256_file(manifest),
        },
    )
    receipt = write_release_receipt(
        release_root=release_root,
        schema_version=SCHEMA,
        release_id=release_id,
        dataset_id=image_receipt["dataset_id"],
        upstream_release_ids={"image": image_receipt["release_id"], "gaze": gaze_receipt["release_id"]},
        resolved_config={"v6_bundle_source_manifest_sha256": sha256_file(manifest), "patch_asset_mode": "deterministic_runtime", "canonical_image_mode": resolve_canonical_image_mode(next(iter(e1.values())).get("canonical_image_mode"))},
        terminal_status="PASS_STAGE1_V6_DATASET_ENTRY_V2_WRAPPER",
        counts={"image_count": len(sidecars)},
        extra={"formal_authorization": False},
    )
    return {"status": "PASS", "release_root": str(release_root), "receipt": str(receipt)}


def validate_stage1_v6_dataset_entry_v2_wrapper(
    *, wrapper_root: Path, image_release_root: Path, gaze_release_root: Path, resolved_formal_config: Path, audit_root: Path
) -> dict[str, Any]:
    """Open actual E1/E2 assets and the copied V6 bundle in an audit root."""
    receipt, _ = load_pass_receipt(wrapper_root)
    image_receipt, _ = load_pass_receipt(image_release_root)
    gaze_receipt, _ = load_pass_receipt(gaze_release_root)
    if receipt.get("upstream_release_ids") != {"image": image_receipt["release_id"], "gaze": gaze_receipt["release_id"]}:
        raise ValueError("V2 wrapper upstream E1/E2 lineage mismatch.")
    v6_result = validate_stage1_v6_bundle(wrapper_root / "v6_bundle", resolved_formal_config)
    if v6_result.get("status") != "PASS_FORMAL":
        raise ValueError(f"Existing V6 bundle validation failed: {v6_result.get('errors', [])[:3]}")
    e1 = _image_rows(image_release_root, "image_preprocessing_manifest.jsonl")
    sidecars = _image_rows(wrapper_root, "dataset_entry_v2_sidecars.jsonl")
    runtime_rows = {
        str(row.get("image_id") or "").strip(): row
        for row in _csv_rows(wrapper_root / "manifests" / "manifest_stage1_v6_dataset_entry_v2.csv")
    }
    if not runtime_rows or "" in runtime_rows or set(runtime_rows) != set(sidecars):
        raise ValueError("V2 wrapper runtime manifest image_ids do not exactly match sidecars.")
    masks = MaskShardReader(image_release_root)
    gaze = GazeShardReader(gaze_release_root)
    for image_id, row in sidecars.items():
        source = e1.get(image_id)
        runtime = runtime_rows[image_id]
        if source is None:
            raise ValueError(f"V2 wrapper sidecar image is absent from E1: {image_id}")
        for key in (
            "canonical_image_mode", "canonical_image_sha256", "canonical_reconstruction_key",
            "image_geometry_sha256", "patch_geometry_key", "patch_token_order_version",
        ):
            if row.get(key) != source.get(key):
                raise ValueError(f"V2 wrapper/E1 mismatch for image_id={image_id} on {key}")
        for key in ("patch_size_h", "patch_size_w", "grid_h", "grid_w", "patch_count"):
            try:
                sidecar_value = int(row[key])
                source_value = int(source[key])
                runtime_value = int(runtime[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"V2 wrapper missing or invalid {key} for image_id={image_id}") from exc
            if sidecar_value != source_value or runtime_value != source_value:
                raise ValueError(f"V2 wrapper/E1 mismatch for image_id={image_id} on {key}")
            if key in {"patch_size_h", "patch_size_w"} and source_value <= 0:
                raise ValueError(f"V2 wrapper patch size must be positive for image_id={image_id}")
        if int(source["patch_count"]) != int(source["grid_h"]) * int(source["grid_w"]):
            raise ValueError(f"V2 wrapper patch_count/grid mismatch for image_id={image_id}")
        if runtime.get("patch_token_order_version") != source.get("patch_token_order_version"):
            raise ValueError(f"V2 wrapper/E1 mismatch for image_id={image_id} on patch_token_order_version")
        if source["patch_token_order_version"] != "row_major_h_w_v1":
            raise ValueError(f"Unsupported frozen patch token order for image_id={image_id}")
        if int(runtime.get("valid_content_mask_length", -1)) != int(source["patch_count"]):
            raise ValueError(f"V2 wrapper valid-content length mismatch for image_id={image_id}")
        if resolve_canonical_image_mode(source.get("canonical_image_mode")) == CANONICAL_IMAGE_MODE_SOURCE_RUNTIME:
            validate_source_runtime_row(source)
            reconstruct_source_runtime_canonical(source)
        else:
            image_path = image_release_root / str(source["canonical_image_path"])
            if not image_path.is_file() or sha256_file(image_path) != str(source["canonical_image_sha256"]):
                raise ValueError(f"Canonical E1 image is missing or tampered: {image_id}")
        masks.read(source["valid_content_mask_ref"], expected_length=int(source["patch_count"]))
        if row["gaze_status"] == "NO_GAZE":
            if int(row["gaze_training_enabled"]) != 0:
                raise ValueError(f"Invalid no-gaze sidecar: {image_id}")
        else:
            gaze.read(
                {
                    "gaze_weight_shard_key": row["gaze_weight_shard_key"], "shard_sha256": row["gaze_shard_sha256"],
                    "offset": row["gaze_offset"], "length": row["gaze_length"], "dtype": "float32",
                    "shape": [int(row["gaze_length"])], "slice_sha256": row["gaze_slice_sha256"],
                    "gaze_weight_sha256": row["gaze_slice_sha256"], "storage_version": "gaze_weight_npy_shard_v1",
                }, expected_length=int(source["patch_count"]),
            )
    if audit_root.exists():
        # Formal resume is allowed only when this exact immutable lineage was
        # already audited successfully.  A different audit never overwrites it.
        prior, _ = load_pass_receipt(audit_root)
        expected = {"wrapper": receipt["release_id"], "image": image_receipt["release_id"], "gaze": gaze_receipt["release_id"]}
        if prior.get("upstream_release_ids") != expected:
            raise ValueError("Existing V6/V2 audit root belongs to different immutable inputs.")
        return {"status": "PASS_FORMAL", "wrapper_release_id": receipt["release_id"], "image_count": len(sidecars), "audit_root": str(audit_root), "receipt": str(audit_root / "receipts" / "terminal_receipt.json"), "reused": True}
    audit_root.mkdir(parents=True)
    result = {"status": "PASS_FORMAL", "wrapper_release_id": receipt["release_id"], "image_count": len(sidecars), "v6_validation": v6_result}
    atomic_write_json(
        audit_root / "release_manifest.json",
        {
            "schema_version": "stage1_v6_dataset_entry_v2_audit_v1",
            "release_id": f"audit:{receipt['release_id']}",
            "dataset_id": receipt["dataset_id"],
            "upstream_release_ids": {
                "wrapper": receipt["release_id"],
                "image": image_receipt["release_id"],
                "gaze": gaze_receipt["release_id"],
            },
        },
    )
    atomic_write_json(audit_root / "stage1_v6_dataset_entry_v2_validation.json", result)
    audit_receipt = write_release_receipt(
        release_root=audit_root,
        schema_version="stage1_v6_dataset_entry_v2_audit_v1",
        release_id=f"audit:{receipt['release_id']}", dataset_id=receipt["dataset_id"],
        upstream_release_ids={"wrapper": receipt["release_id"], "image": image_receipt["release_id"], "gaze": gaze_receipt["release_id"]},
        resolved_config={"validator": "open_e1_e2_sidecars"}, terminal_status="PASS_STAGE1_V6_DATASET_ENTRY_V2_AUDIT",
        counts={"image_count": len(sidecars)}, extra={"formal_authorization": False},
    )
    return {**result, "audit_root": str(audit_root), "receipt": str(audit_receipt)}


__all__ = ["materialize_stage1_v6_dataset_entry_v2_wrapper", "validate_stage1_v6_dataset_entry_v2_wrapper"]
