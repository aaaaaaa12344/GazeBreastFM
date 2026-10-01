from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from breast_pretrain.data.image_loading import pil_to_raw_tensor
from breast_pretrain.clinical_graph_sidecar.runtime_loader import ClinicalGraphV2SidecarLoader
from breast_pretrain.datasets.breast_image_dataset import BreastImageDataset
from breast_pretrain.gaze.prior_io import resolve_prior_path
from breast_pretrain.data_entry.gaze_shards import GazeShardReader

_HIGH_CONF_PATCH_PRIOR_PATH_FIELDS = (
    "high_conf_patch_prior_path",
    "high_conf_patch_weight_path",
)


def _apply_source_path_rebind(records: list[dict[str, Any]]) -> None:
    config_value = os.environ.get("HSM_STAGE1_SOURCE_PATH_REBIND", "").strip()
    if not config_value:
        return
    config_path = Path(config_value).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Source path rebind config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema_version = config.get("schema_version")
    if schema_version not in {
        "formal_stage1_source_path_prefix_rebind_v1",
        "formal_stage1_source_path_prefix_rebind_v2",
    }:
        raise ValueError(f"Unsupported source path rebind schema: {config_path}")
    mappings = [config] if schema_version.endswith("_v1") else config.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError(f"Source path rebind config has no mappings: {config_path}")
    seen_datasets: set[str] = set()
    for mapping in mappings:
        dataset_id = str(mapping.get("dataset_id") or "").strip()
        old_prefix = str(mapping.get("old_prefix") or "").strip()
        new_prefix = str(mapping.get("new_prefix") or "").strip()
        expected_rows = int(mapping.get("expected_manifest_rows", 0))
        if not dataset_id or not old_prefix or not new_prefix or expected_rows <= 0:
            raise ValueError(f"Incomplete source path rebind mapping: {config_path}")
        if dataset_id in seen_datasets:
            raise ValueError(f"Duplicate source path rebind dataset_id: {dataset_id}")
        seen_datasets.add(dataset_id)
        if not old_prefix.endswith("/") or not new_prefix.endswith("/"):
            raise ValueError("Source path rebind prefixes must end with '/'.")
        new_root = Path(new_prefix).expanduser().resolve()
        if not new_root.is_dir():
            raise FileNotFoundError(f"Source path rebind target root not found: {new_root}")

        if schema_version.endswith("_v2"):
            receipt_path = Path(str(mapping.get("mirror_receipt_path") or "")).expanduser().resolve()
            expected_receipt_sha = str(mapping.get("mirror_receipt_sha256") or "").lower()
            expected_authority_rows = int(mapping.get("expected_authority_rows", 0))
            expected_authority_sha = str(mapping.get("authority_manifest_sha256") or "").lower()
            if (
                not receipt_path.is_file()
                or len(expected_receipt_sha) != 64
                or expected_authority_rows <= 0
                or len(expected_authority_sha) != 64
            ):
                raise ValueError(f"Missing mirror PASS receipt binding for dataset_id={dataset_id}")
            observed_receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt_target = str(Path(str(receipt.get("target_root") or "")).resolve()).rstrip("/") + "/"
            receipt_rows = int(receipt.get("validated_rows", -1))
            if (
                observed_receipt_sha != expected_receipt_sha
                or receipt.get("PASS") is not True
                or receipt_target != str(new_root).rstrip("/") + "/"
                or receipt_rows != expected_authority_rows
                or int(receipt.get("expected_rows", -1)) != expected_authority_rows
                or str(receipt.get("authority_manifest_sha256") or "").lower() != expected_authority_sha
            ):
                raise ValueError(f"Mirror receipt validation failed for dataset_id={dataset_id}")

        matched = 0
        for record in records:
            if str(record.get("dataset_id") or "").strip() != dataset_id:
                continue
            source_path = str(record.get("image_path") or "").strip()
            if not source_path.startswith(old_prefix):
                raise ValueError(
                    f"Source path rebind exact-prefix mismatch for {_record_label(record)}: {source_path}"
                )
            suffix = source_path[len(old_prefix):]
            if not suffix or suffix.startswith("/"):
                raise ValueError(f"Source path rebind invalid suffix for {_record_label(record)}")
            record["image_path"] = new_prefix + suffix
            matched += 1
        if matched != expected_rows:
            raise ValueError(
                f"Source path rebind matched {matched} rows for dataset_id={dataset_id}; expected {expected_rows}."
            )


def _load_gaze_membership_authority(
    path: Path,
    *,
    expected_available: int,
    expected_disabled: int,
) -> set[str]:
    """Read the frozen exact-image gaze authority and return enabled IDs only."""
    try:
        receipt = json.loads(path.with_name("gaze_available_membership_audit_receipt.json").read_text(encoding="utf-8"))
        expected_roster = int(receipt["formal_roster"]["expected_rows"])
        global_membership = receipt["global_membership"]
        details_path = Path(str(receipt["details_file"])).expanduser()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid gaze membership receipt beside {path}: {exc}") from exc
    if receipt.get("status") != "PASS_ALL_RESOLVED":
        raise ValueError(f"Gaze membership authority is not PASS_ALL_RESOLVED: {path}")
    if int(global_membership.get("GAZE_AVAILABLE", -1)) != expected_available:
        raise ValueError("Frozen gaze authority available count does not match formal configuration.")
    if int(global_membership.get("GAZE_DISABLED", -1)) != expected_disabled:
        raise ValueError("Frozen gaze authority disabled count does not match formal configuration.")
    if not details_path.is_file():
        raise FileNotFoundError(f"Gaze membership details file not found: {details_path}")

    enabled: set[str] = set()
    seen: set[str] = set()
    try:
        with details_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid gaze membership JSON at line {line_number}: {details_path}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Gaze membership line {line_number} is not an object: {details_path}")
                image_id = str(row.get("image_id", "")).strip()
                membership = str(row.get("membership", "")).strip()
                if not image_id or image_id in seen:
                    raise ValueError(f"Missing or duplicate gaze membership image_id at line {line_number}.")
                if membership not in {"GAZE_AVAILABLE", "GAZE_DISABLED"}:
                    raise ValueError(f"Unsupported gaze membership {membership!r} at line {line_number}.")
                seen.add(image_id)
                if membership == "GAZE_AVAILABLE":
                    enabled.add(image_id)
    except UnicodeDecodeError as exc:
        raise ValueError(f"Gaze membership details are not UTF-8: {details_path}") from exc
    if len(seen) != expected_roster or len(enabled) != expected_available:
        raise ValueError(
            f"Gaze membership counts do not close frozen roster: rows={len(seen)} expected={expected_roster}, "
            f"available={len(enabled)} expected={expected_available}."
        )
    return enabled


def _is_no_gaze(record: dict[str, Any]) -> bool:
    return str(record.get("gaze_supervision_source", "")).strip().lower() == "no_gaze"


def _resolve_bundle_or_manifest_relative_path(
    raw_value: Any,
    manifest_dir: Path,
) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = (
        (manifest_dir / path).resolve(),
        (manifest_dir.parent / path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _record_label(record: dict[str, Any]) -> str:
    row_number = record.get("__manifest_row_number__", "?")
    image_id = str(record.get("image_id", "")).strip() or "<missing>"
    return f"row {row_number} image_id={image_id}"


def _patch_grid_for_dataset(dataset: BreastImageDataset, record: dict[str, Any]) -> tuple[int, int]:
    spec = dataset._transform_spec_for_record(record)
    return int(spec.patch_grid[0]), int(spec.patch_grid[1])


def _load_patch_grid_array(
    path: Path,
    *,
    expected_grid: tuple[int, int],
    field_name: str,
    record: dict[str, Any],
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 {field_name} file not found for {_record_label(record)}: {path}"
        )
    array = np.load(path, allow_pickle=False)
    if not np.isfinite(array).all():
        raise ValueError(
            f"Stage 1 {field_name} contains non-finite values for {_record_label(record)}: {path}"
        )
    expected_h, expected_w = expected_grid
    expected_numel = expected_h * expected_w
    if array.ndim == 1 and int(array.shape[0]) == expected_numel:
        flat = array
    elif array.ndim == 2 and tuple(int(item) for item in array.shape) == expected_grid:
        flat = array.reshape(-1)
    elif array.ndim == 3 and int(array.shape[0]) == 1 and tuple(int(item) for item in array.shape[1:]) == expected_grid:
        flat = array[0].reshape(-1)
    elif array.ndim == 3 and int(array.shape[-1]) == 1 and tuple(int(item) for item in array.shape[:2]) == expected_grid:
        flat = array[..., 0].reshape(-1)
    else:
        raise ValueError(
            f"Stage 1 {field_name} shape mismatch for {_record_label(record)}: "
            f"got {tuple(array.shape)}, expected {expected_grid} or ({expected_numel},)."
        )
    return torch.as_tensor(flat, dtype=torch.float32)


def _load_patch_gaze_weight(
    dataset: BreastImageDataset,
    record: dict[str, Any],
) -> torch.Tensor | None:
    raw_path = str(record.get("patch_gaze_weight_path", "")).strip()
    if not raw_path:
        shard_key = str(record.get("gaze_weight_shard_key", "")).strip()
        shard_root = str(record.get("gaze_shard_root", "")).strip()
        if shard_key and shard_root:
            values = GazeShardReader(Path(shard_root)).read(
                {
                    "gaze_weight_shard_key": shard_key,
                    "shard_sha256": record.get("gaze_shard_sha256"),
                    "offset": int(record.get("gaze_offset", -1)),
                    "length": int(record.get("gaze_length", -1)),
                    "slice_sha256": record.get("gaze_slice_sha256"),
                    "gaze_weight_sha256": record.get("gaze_slice_sha256"),
                    "storage_version": "gaze_weight_npy_shard_v1",
                },
                expected_length=int(_patch_grid_for_dataset(dataset, record)[0] * _patch_grid_for_dataset(dataset, record)[1]),
            )
            return torch.from_numpy(values.copy()).to(dtype=torch.float32)
        if _is_no_gaze(record):
            return None
        return None
    path = _resolve_bundle_or_manifest_relative_path(raw_path, dataset.manifest_dir)
    if path is None:
        return None
    return _load_patch_grid_array(
        path,
        expected_grid=_patch_grid_for_dataset(dataset, record),
        field_name="patch_gaze_weight",
        record=record,
    )


def _load_explicit_high_conf_patch_prior(
    dataset: BreastImageDataset,
    record: dict[str, Any],
) -> torch.Tensor | None:
    for field_name in _HIGH_CONF_PATCH_PRIOR_PATH_FIELDS:
        raw_path = str(record.get(field_name, "")).strip()
        if not raw_path:
            continue
        path = _resolve_bundle_or_manifest_relative_path(raw_path, dataset.manifest_dir)
        if path is None:
            continue
        return _load_patch_grid_array(
            path,
            expected_grid=_patch_grid_for_dataset(dataset, record),
            field_name=field_name,
            record=record,
        )
    return None


def _load_single_channel_mask_tensor(path: Path, record: dict[str, Any]) -> torch.Tensor:
    if path.suffix.lower() != ".npy":
        with Image.open(path) as image:
            return pil_to_raw_tensor(image, mode="gray")

    array = np.load(path, allow_pickle=False)
    if not np.isfinite(array).all():
        raise ValueError(
            f"Stage 1 high_conf_mask contains non-finite values for {_record_label(record)}: {path}"
        )
    if array.ndim == 2:
        spatial = array
    elif array.ndim == 3 and int(array.shape[0]) == 1:
        spatial = array[0]
    elif array.ndim == 3 and int(array.shape[-1]) == 1:
        spatial = array[..., 0]
    else:
        raise ValueError(
            f"Stage 1 high_conf_mask shape mismatch for {_record_label(record)}: "
            f"got {tuple(array.shape)}, expected (H, W), (1, H, W), or (H, W, 1)."
        )
    return torch.as_tensor(spatial, dtype=torch.float32).unsqueeze(0)


def _project_high_conf_mask_to_patch_prior(
    dataset: BreastImageDataset,
    record: dict[str, Any],
) -> torch.Tensor | None:
    raw_path = str(record.get("high_conf_mask_path", "")).strip()
    if not raw_path:
        if _is_no_gaze(record):
            return None
        return None
    path = _resolve_bundle_or_manifest_relative_path(raw_path, dataset.manifest_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 high_conf_mask file not found for {_record_label(record)}: {path}"
        )
    raw_mask = _load_single_channel_mask_tensor(path, record)

    expected_grid = _patch_grid_for_dataset(dataset, record)
    spec = dataset._transform_spec_for_record(record)
    if str(record.get("gaze_sidecar_coordinate_space", "")).strip() == "canonical_target_canvas":
        if tuple(raw_mask.shape[-2:]) != tuple(spec.image_size):
            raise ValueError(
                f"Canonical Stage 1 high_conf_mask shape mismatch for {_record_label(record)}: "
                f"got {tuple(raw_mask.shape[-2:])}, expected {tuple(spec.image_size)}."
            )
        high_conf_mask = raw_mask
    else:
        high_conf_mask, _metadata = dataset._apply_record_spatial_transform(
            raw_mask,
            record,
            is_mask=True,
        )
    pooled = F.max_pool2d(
        (high_conf_mask >= 0.5).to(dtype=torch.float32).unsqueeze(0),
        kernel_size=spec.patch_size,
        stride=spec.patch_size,
    )
    prior = pooled[0, 0].reshape(-1).to(dtype=torch.float32)
    if int(prior.numel()) != expected_grid[0] * expected_grid[1]:
        raise ValueError(
            f"Stage 1 high_conf_patch_prior length mismatch for {_record_label(record)}: "
            f"got {int(prior.numel())}, expected {expected_grid[0] * expected_grid[1]}."
        )
    return prior


class JointPretrainDataset(BreastImageDataset):
    """Production wrapper around the existing manifest-backed dataset."""

    def __init__(
        self,
        *args: Any,
        clinical_graph_v2_sidecar_loader: ClinicalGraphV2SidecarLoader | None = None,
        gaze_membership_path: str | Path | None = None,
        gaze_membership_expected_available: int | None = None,
        gaze_membership_expected_disabled: int | None = None,
        **kwargs: Any,
    ) -> None:
        membership_requested = gaze_membership_path is not None
        if membership_requested and kwargs.get("require_attention_prior_paths"):
            # Excluded rows are not formal inputs; validate required priors only after authority filtering.
            kwargs = dict(kwargs)
            kwargs["require_attention_prior_paths"] = False
        super().__init__(*args, **kwargs)
        _apply_source_path_rebind(self.records)
        self.clinical_graph_v2_sidecar_loader = clinical_graph_v2_sidecar_loader
        if gaze_membership_path is not None:
            if gaze_membership_expected_available is None or gaze_membership_expected_disabled is None:
                raise ValueError(
                    "Gaze membership authority requires explicit expected available and disabled counts."
                )
            membership_path = Path(gaze_membership_path).expanduser().resolve()
            if not membership_path.is_file():
                raise FileNotFoundError(f"Gaze membership authority not found: {membership_path}")
            enabled_ids = _load_gaze_membership_authority(
                membership_path,
                expected_available=int(gaze_membership_expected_available),
                expected_disabled=int(gaze_membership_expected_disabled),
            )
            manifest_ids = {str(record.get("image_id", "")).strip() for record in self.records}
            if len(manifest_ids) != len(self.records):
                raise ValueError("Stage 1 manifest contains duplicate or empty image_id values.")
            missing = enabled_ids - manifest_ids
            if missing:
                raise ValueError(f"Gaze-enabled authority IDs missing from Stage 1 manifest: {sorted(missing)[:5]}")
            self.records = [record for record in self.records if str(record.get("image_id", "")).strip() in enabled_ids]
            if len(self.records) != int(gaze_membership_expected_available):
                raise ValueError(
                    f"Filtered Stage 1 manifest count={len(self.records)} does not equal "
                    f"gaze-qualified count={int(gaze_membership_expected_available)}."
                )
            if self.require_attention_prior_paths:
                self._validate_required_prior_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(super().__getitem__(index))
        record = self.records[index]
        sample["attention_map_path"] = str(record.get("attention_map_path", "")).strip()
        sample["high_conf_mask_path"] = str(record.get("high_conf_mask_path", "")).strip()
        sample["patch_gaze_weight_path"] = str(record.get("patch_gaze_weight_path", "")).strip()
        sample["dataset_id"] = str(record.get("dataset_id", "")).strip()
        sample["case_id"] = str(record.get("case_id", "")).strip()
        sample["trajectory_consensus_summary_path"] = str(
            record.get("trajectory_consensus_summary_path", "")
        ).strip()
        sample["prior_version"] = str(record.get("prior_version", "")).strip()
        sample["coverage_ratio"] = str(record.get("coverage_ratio", "")).strip()
        sample["high_conf_area_ratio"] = str(record.get("high_conf_area_ratio", "")).strip()
        sample["inside_ratio"] = str(record.get("inside_ratio", "")).strip()
        sample["clinical_graph_prior_path"] = str(
            record.get("clinical_graph_prior_path", "")
        ).strip()
        if not self.dataset_entry_v2_enabled:
            patch_gaze_weight = _load_patch_gaze_weight(self, record)
            if str(getattr(self, "manifest_path", "")).find("formal") >= 0:
                if patch_gaze_weight is None or patch_gaze_weight.numel() == 0:
                    raise ValueError(f"Formal patch_gaze_weight missing for image_id={record.get('image_id')}")
                if not torch.isfinite(patch_gaze_weight).all() or float(patch_gaze_weight.sum()) <= 0.0:
                    raise ValueError(f"Formal patch_gaze_weight is non-finite or zero-mass for image_id={record.get('image_id')}")
            if patch_gaze_weight is not None:
                sample["patch_gaze_weight"] = patch_gaze_weight
            high_conf_patch_prior = _load_explicit_high_conf_patch_prior(self, record)
            if high_conf_patch_prior is None:
                high_conf_patch_prior = _project_high_conf_mask_to_patch_prior(
                    self,
                    record,
                )
            if high_conf_patch_prior is not None:
                sample["high_conf_patch_prior"] = high_conf_patch_prior
        if self.clinical_graph_v2_sidecar_loader is not None:
            graph_sample = self.clinical_graph_v2_sidecar_loader.lookup(record)
            sample["clinical_graph_v2_node_values"] = graph_sample.node_values.clone()
            sample["clinical_graph_v2_observed_mask"] = graph_sample.observed_mask.clone()
            sample["clinical_graph_v2_node_index"] = graph_sample.node_index.clone()
            sample["clinical_graph_v2_node_ids"] = self.clinical_graph_v2_sidecar_loader.node_ids
        return sample

    def resolve_attention_map_path(self, record: dict[str, Any]) -> Path | None:
        attention_path = resolve_prior_path(record.get("attention_map_path"), self.manifest_dir)
        if attention_path is None and self.attention_map_dir is not None:
            return (self.attention_map_dir / f"{record['image_id']}.png").resolve()
        return attention_path

    def resolve_high_conf_mask_path(self, record: dict[str, Any]) -> Path | None:
        return resolve_prior_path(record.get("high_conf_mask_path"), self.manifest_dir)


def build_joint_pretrain_dataset(**kwargs: Any) -> JointPretrainDataset:
    return JointPretrainDataset(**kwargs)
