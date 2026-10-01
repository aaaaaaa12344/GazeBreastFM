from __future__ import annotations

import csv
import json
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from breast_pretrain.data.image_loading import (
    DICOM_SUFFIXES,
    load_dicom_pil_image,
    pil_to_raw_tensor,
)
from breast_pretrain.data.transforms.stage1_transform_spec import (
    ImageSize,
    Stage1TransformSpec,
    apply_stage1_spatial_transform,
    build_valid_content_patch_mask_from_geometry,
    build_stage1_transform_spec,
    normalize_image_size,
    stage1_transform_geometry_checksum,
    stage1_transform_spec_checksum,
)
from breast_pretrain.data.stage1_gaze_enabled_bundle import (
    CONCEPT_TARGET_FIELDS,
    CONCEPT_TARGET_METADATA_SUFFIXES,
)
from breast_pretrain.teachers import (
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TEACHER_SOURCE_MISSING,
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
)
from breast_pretrain.data.runtime_canonical_cache import RuntimeCanonicalCache, SourceRuntimeCacheConfig
from breast_pretrain.data.source_runtime_image import (
    CANONICAL_IMAGE_MODE_SOURCE_RUNTIME,
    SourceIntegrityMemo,
    load_source_runtime_canonical,
    resolve_canonical_image_mode,
)
from breast_pretrain.data_entry.gaze_shards import GazeShardReader
from breast_pretrain.data_entry.mask_shards import MaskShardReader
from breast_pretrain.data_entry.release_common import sha256_file
from breast_pretrain.train.stage1_joint.dynamic_patch_utils import assert_frozen_dataset_entry_v2_geometry


REQUIRED_SAMPLE_FIELDS = {"image_id", "image_path", "modality"}
SUPPORTED_MODALITIES = {"mri", "mammo", "mammography", "ultrasound"}
REQUIRED_PRIOR_FIELDS = ("attention_map_path", "high_conf_mask_path")
_FORMAL_GAZE_RUNTIME_BINDING_VALUE = os.environ.get("HSM_FORMAL_GAZE_RUNTIME_BINDING", "").strip()
FORMAL_GAZE_RUNTIME_BINDING = (
    Path(_FORMAL_GAZE_RUNTIME_BINDING_VALUE).expanduser().resolve()
    if _FORMAL_GAZE_RUNTIME_BINDING_VALUE
    else None
)
PROMPT_SOURCE_TEXT_PROMPT_PATH = "text_prompt_path"


def _record_identity(record: dict[str, Any]) -> str:
    row_number = int(record.get("__manifest_row_number__", -1))
    image_id = str(record.get("image_id", "")).strip() or "<missing>"
    return f"row {row_number} (image_id={image_id})"


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"JSONL row {line_number} in {manifest_path} is not an object."
                    )
                rows.append(item)
    else:
        raise ValueError(
            f"Unsupported manifest format: {manifest_path}. Expected .csv or .jsonl."
        )

    for index, row in enumerate(rows, start=1):
        row["__manifest_row_number__"] = index
        raw_row_index = str(row.get("row_index", "")).strip()
        row["__manifest_row_index__"] = int(raw_row_index) if raw_row_index else index - 1
        missing_fields = [
            field for field in REQUIRED_SAMPLE_FIELDS if not str(row.get(field, "")).strip()
        ]
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Manifest {_record_identity(row)} is missing required fields: {missing}"
            )

        modality = str(row.get("modality", "")).strip().lower()
        if modality not in SUPPORTED_MODALITIES:
            supported = ", ".join(sorted(SUPPORTED_MODALITIES))
            raise ValueError(
                f"Manifest {_record_identity(row)} has unsupported modality '{modality}'. "
                f"Expected one of: {supported}."
            )

    return rows


def _load_prompt_lookup(prompt_path: Path | None) -> dict[str, str]:
    if prompt_path is None or not prompt_path.exists():
        return {}

    suffix = prompt_path.suffix.lower()
    prompts: dict[str, str] = {}

    if suffix == ".jsonl":
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                image_id = str(item.get("image_id", "")).strip()
                text_prompt = str(item.get("text_prompt", "")).strip()
                if not image_id:
                    raise ValueError(
                        f"Prompt row {line_number} in {prompt_path} must include image_id."
                    )
                if not text_prompt:
                    continue
                prompts[image_id] = text_prompt
        return prompts

    if suffix == ".csv":
        with prompt_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                image_id = str(row.get("image_id", "")).strip()
                text_prompt = str(row.get("text_prompt", "")).strip()
                if not image_id:
                    raise ValueError(
                        f"Prompt row {index} in {prompt_path} must include image_id."
                    )
                if not text_prompt:
                    continue
                prompts[image_id] = text_prompt
        return prompts

    raise ValueError(f"Unsupported prompt format: {prompt_path}. Expected .csv or .jsonl.")


def _load_teacher_manifest_lookup(teacher_latent_dir: Path | None) -> dict[str, dict[str, Any]]:
    if teacher_latent_dir is None:
        return {}

    manifest_path = teacher_latent_dir / "teacher_latent_manifest.json"
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"teacher_latent_manifest.json entries must be a list: {manifest_path}"
        )

    lookup: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"teacher_latent_manifest.json entry must be an object: {manifest_path}"
            )
        image_id = str(entry.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(
                f"teacher_latent_manifest.json entry is missing image_id: {manifest_path}"
            )
        lookup[image_id] = entry
    return lookup


def _resolve_optional_path(raw_value: Any, base_dir: Path) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path





def _load_single_channel_npy(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 2:
        spatial = array
    elif array.ndim == 3 and array.shape[0] == 1:
        spatial = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        spatial = array[..., 0]
    else:
        raise ValueError(
            f"Unsupported single-channel .npy shape in {path}: {array.shape}. "
            "Expected (H, W), (1, H, W), or (H, W, 1)."
        )

    return torch.as_tensor(spatial, dtype=torch.float32).unsqueeze(0)



def _build_default_prompt(record: dict[str, Any]) -> str:
    modality = str(record.get("modality", "unknown")).strip().lower() or "unknown"
    laterality = str(record.get("laterality", "unknown")).strip() or "unknown"
    view = str(record.get("view", "unknown")).strip() or "unknown"
    study = str(record.get("study_description", "unknown")).strip() or "unknown"

    if modality == "mri":
        prefix = "Structured breast MRI prompt"
    elif modality == "ultrasound":
        prefix = "Structured breast ultrasound prompt"
    else:
        prefix = "Structured mammography prompt"

    return (
        f"{prefix}: "
        f"modality={modality}; laterality={laterality}; view={view}; study={study}."
    )


class BreastImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: ImageSize,
        attention_map_dir: str | Path | None = None,
        teacher_latent_dir: str | Path | None = None,
        text_prompt_path: str | Path | None = None,
        max_samples: int | None = None,
        require_attention_prior_paths: bool = False,
        image_size_by_modality: dict[str, tuple[int, int]] | None = None,
        transform_policy_by_modality: dict[str, str] | None = None,
        patch_size: int = 16,
        dataset_entry_v2_enabled: bool = False,
        image_release_root: str | Path | None = None,
        gaze_release_root: str | Path | None = None,
        canonical_runtime: dict[str, Any] | None = None,
        source_runtime_cache: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {self.manifest_path}")

        self.image_size = normalize_image_size(image_size)
        self.image_size_by_modality: dict[str, tuple[int, int]] = {}
        for key, value in (image_size_by_modality or {}).items():
            modality_key = str(key).strip().lower()
            if modality_key == "mammo":
                modality_key = "mammography"
            if modality_key == "us":
                modality_key = "ultrasound"
            self.image_size_by_modality[modality_key] = normalize_image_size(value)
        self.transform_policy_by_modality: dict[str, str] = {}
        for key, value in (transform_policy_by_modality or {}).items():
            modality_key = str(key).strip().lower()
            if modality_key == "mammo":
                modality_key = "mammography"
            if modality_key == "us":
                modality_key = "ultrasound"
            policy = str(value).strip()
            if modality_key and policy:
                self.transform_policy_by_modality[modality_key] = policy
        self.patch_size = int(patch_size)
        self.dataset_entry_v2_enabled = bool(dataset_entry_v2_enabled)
        if self.dataset_entry_v2_enabled:
            if image_release_root is None or gaze_release_root is None:
                raise ValueError("Dataset Entry V2 runtime requires image_release_root and gaze_release_root.")
            self.image_release_root = Path(image_release_root).expanduser().resolve()
            self.gaze_release_root = Path(gaze_release_root).expanduser().resolve()
            self._v2_masks = MaskShardReader(self.image_release_root)
            self._v2_gaze = GazeShardReader(self.gaze_release_root)
            runtime = canonical_runtime or {}
            self._canonical_expected_mode = str(runtime.get("expected_mode", "auto")).strip().lower() or "auto"
            if self._canonical_expected_mode not in {"auto", "materialized_npy", "source_runtime"}:
                raise ValueError("canonical_runtime.expected_mode must be auto, materialized_npy, or source_runtime.")
            cache_config = SourceRuntimeCacheConfig.from_mapping(source_runtime_cache)
            if cache_config.root is not None:
                try:
                    cache_config.root.relative_to(self.image_release_root)
                except ValueError:
                    pass
                else:
                    raise ValueError("source_runtime_cache.root must not be inside the formal E1 release root.")
            self._source_runtime_cache = RuntimeCanonicalCache(cache_config)
            self._source_integrity_memo: SourceIntegrityMemo = {}
        self.manifest_dir = self.manifest_path.parent
        self.attention_map_dir = (
            Path(attention_map_dir).expanduser().resolve()
            if attention_map_dir is not None
            else None
        )
        self.teacher_latent_dir = (
            Path(teacher_latent_dir).expanduser().resolve()
            if teacher_latent_dir is not None
            else None
        )
        self.text_prompt_path = (
            Path(text_prompt_path).expanduser().resolve()
            if text_prompt_path is not None
            else None
        )
        self.require_attention_prior_paths = bool(require_attention_prior_paths)

        self.records = _load_manifest(self.manifest_path)
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided.")
            self.records = self.records[: int(max_samples)]
        if self.require_attention_prior_paths and any(not str(r.get("attention_map_path", "")).strip() for r in self.records):
            binding = FORMAL_GAZE_RUNTIME_BINDING
            if binding is None or not binding.is_file():
                raise FileNotFoundError(
                    "Formal gaze runtime binding is unavailable; set HSM_FORMAL_GAZE_RUNTIME_BINDING."
                )
            index_path = Path(
                os.environ.get(
                    "HSM_GAZE_RUNTIME_INDEX_PATH",
                    str(binding.parent / "gaze_runtime_binding_index.sqlite"),
                )
            ).expanduser()
            if not index_path.is_file():
                raise FileNotFoundError(f"Gaze runtime binding index does not exist: {index_path}")
            with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as db:
                columns = {str(row[1]) for row in db.execute("PRAGMA table_info(gaze)").fetchall()}
                select_fields = ["image_id", "resolved_root", "stage0_prior_key"]
                if "attention_map_path" in columns:
                    select_fields.append("attention_map_path")
                if "high_conf_mask_path" in columns:
                    select_fields.append("high_conf_mask_path")
                if "patch_gaze_weight_path" in columns:
                    select_fields.append("patch_gaze_weight_path")
                for field in (
                    "gaze_weight_shard_key", "gaze_shard_sha256", "gaze_slice_sha256",
                    "gaze_offset", "gaze_length", "gaze_shard_root",
                    "gaze_supervision_source", "prior_status",
                ):
                    if field in columns:
                        select_fields.append(field)
                binding_rows = {
                    str(row[0]).strip(): row[1:]
                    for row in db.execute(f"SELECT {', '.join(select_fields)} FROM gaze")
                }
                for record in self.records:
                    image_id = str(record.get("image_id", "")).strip()
                    item = binding_rows.get(image_id)
                    if item is None:
                        raise ValueError(f"Gaze runtime binding missing image_id={image_id}")
                    root = Path(str(item[0]).strip()).expanduser(); key = str(item[1]).strip()
                    attention_index = select_fields.index("attention_map_path") - 1 if "attention_map_path" in select_fields else -1
                    high_conf_index = select_fields.index("high_conf_mask_path") - 1 if "high_conf_mask_path" in select_fields else -1
                    patch_weight_index = select_fields.index("patch_gaze_weight_path") - 1 if "patch_gaze_weight_path" in select_fields else -1
                    attention_locator = str(item[attention_index]).strip() if attention_index >= 0 and item[attention_index] else ""
                    high_conf_locator = str(item[high_conf_index]).strip() if high_conf_index >= 0 and item[high_conf_index] else ""
                    patch_weight_locator = str(item[patch_weight_index]).strip() if patch_weight_index >= 0 and item[patch_weight_index] else ""
                    if not root.is_absolute() or (not key and not attention_locator and not high_conf_locator):
                        raise ValueError(f"Incomplete Gaze runtime binding for image_id={image_id}")
                    # Avoid per-row realpath/stat calls during formal initialization. Physical
                    # readability and identity are checked when the representative/sample is consumed.
                    record["attention_map_path"] = str(Path(attention_locator).expanduser()) if attention_locator else str(root / key)
                    record["high_conf_mask_path"] = str(Path(high_conf_locator).expanduser()) if high_conf_locator else str(root / "masks" / Path(key).name.replace("_soft_attention.npy", "_high_conf_mask.npy"))
                    if patch_weight_locator:
                        record["patch_gaze_weight_path"] = str(Path(patch_weight_locator).expanduser())
                    elif str(self.manifest_path).find("formal") >= 0:
                        def bound(field: str, default: Any = "") -> Any:
                            return item[select_fields.index(field) - 1] if field in select_fields else default
                        shard_key = str(bound("gaze_weight_shard_key") or "").strip()
                        if shard_key:
                            record.update({
                                "gaze_weight_shard_key": shard_key,
                                "gaze_shard_sha256": bound("gaze_shard_sha256"),
                                "gaze_slice_sha256": bound("gaze_slice_sha256"),
                                "gaze_offset": int(bound("gaze_offset", -1)),
                                "gaze_length": int(bound("gaze_length", -1)),
                                "gaze_shard_root": str(bound("gaze_shard_root")),
                            })
                        else:
                            raise ValueError(f"Formal Gaze runtime binding missing patch_gaze_weight_path for image_id={image_id}")
                    if "gaze_supervision_source" in select_fields:
                        record["gaze_supervision_source"] = str(
                            item[select_fields.index("gaze_supervision_source") - 1] or ""
                        ).strip()
                    if "prior_status" in select_fields:
                        record["prior_status"] = str(
                            item[select_fields.index("prior_status") - 1] or ""
                        ).strip()
        if self.dataset_entry_v2_enabled and self._source_runtime_cache.config.root is not None:
            forbidden = [self.image_release_root, self.gaze_release_root, *self._source_runtime_cache.config.forbidden_roots]
            if self.manifest_path.parent.name == "manifests":
                forbidden.extend([self.manifest_path.parent, self.manifest_path.parent.parent])
            for record in self.records:
                if resolve_canonical_image_mode(record.get("canonical_image_mode")) == CANONICAL_IMAGE_MODE_SOURCE_RUNTIME:
                    forbidden.append(Path(str(record.get("source_image_path") or "")).expanduser().resolve().parent)
            for root in forbidden:
                try:
                    self._source_runtime_cache.config.root.relative_to(root.resolve())
                except ValueError:
                    continue
                raise ValueError("source_runtime_cache.root is inside a forbidden formal, bundle, or source-dataset root.")
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided.")
            self.records = self.records[: int(max_samples)]
        if self.require_attention_prior_paths:
            self._validate_required_prior_paths()
        self.prompt_lookup = _load_prompt_lookup(self.text_prompt_path)
        self.teacher_manifest_lookup = _load_teacher_manifest_lookup(self.teacher_latent_dir)

    def __len__(self) -> int:
        return len(self.records)

    def _target_size_for_record(self, record: dict[str, Any]) -> tuple[int, int]:
        modality = str(record.get("modality", "")).strip().lower()
        if modality == "mammo":
            modality = "mammography"
        if modality == "us":
            modality = "ultrasound"
        return self.image_size_by_modality.get(modality, self.image_size)

    def _transform_spec_for_record(self, record: dict[str, Any]) -> Stage1TransformSpec:
        modality = str(record.get("modality", "")).strip().lower()
        if modality == "mammo":
            modality = "mammography"
        if modality == "us":
            modality = "ultrasound"
        return build_stage1_transform_spec(
            modality=modality,
            image_size=self._target_size_for_record(record),
            patch_size=self.patch_size,
            policy=self.transform_policy_by_modality.get(modality),
        )

    def _apply_record_spatial_transform(
        self,
        tensor: torch.Tensor,
        record: dict[str, Any],
        *,
        is_mask: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        spec = self._transform_spec_for_record(record)
        transformed, geometry = apply_stage1_spatial_transform(
            tensor.unsqueeze(0),
            spec,
            is_mask=is_mask,
        )
        metadata = self._transform_metadata(spec, geometry)
        return transformed.squeeze(0), metadata

    @staticmethod
    def _transform_metadata(
        spec: Stage1TransformSpec,
        geometry: dict[str, object],
    ) -> dict[str, object]:
        return {
            **spec.to_metadata(),
            "transform_spec_checksum": stage1_transform_spec_checksum(spec),
            "transform_geometry": geometry,
            "transform_geometry_checksum": stage1_transform_geometry_checksum(spec, geometry),
        }

    def _valid_content_patch_mask(
        self,
        *,
        image_transform: dict[str, object],
        record: dict[str, Any],
    ) -> torch.Tensor:
        spec = self._transform_spec_for_record(record)
        mask, _overlap = build_valid_content_patch_mask_from_geometry(
            transform_policy=spec.policy,
            geometry=image_transform["transform_geometry"],
            patch_grid=spec.patch_grid,
            patch_size=spec.patch_size,
        )
        return mask

    def _load_image_tensor(self, image_path: Path, record: dict[str, Any]) -> torch.Tensor:
        image, _metadata, _valid = self._load_image_tensor_with_metadata(image_path, record)
        return image

    def _load_image_tensor_with_metadata(
        self,
        image_path: Path,
        record: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
        target_size = self._target_size_for_record(record)
        if self.dataset_entry_v2_enabled:
            mode = resolve_canonical_image_mode(record.get("canonical_image_mode"))
            if mode == CANONICAL_IMAGE_MODE_SOURCE_RUNTIME:
                tensor, metadata = load_source_runtime_canonical(
                    record, cache=self._source_runtime_cache, expected_mode=self._canonical_expected_mode,
                    source_integrity_memo=self._source_integrity_memo,
                )
                patch_count = int(record.get("patch_count", 0))
                valid = self._v2_masks.read({"shard_key":record.get("valid_content_mask_shard_key"),"offset":int(record.get("valid_content_mask_offset",-1)),"length":patch_count,"storage_length":(patch_count+7)//8,"dtype":"uint8","shape":[patch_count],"sha256":record.get("valid_content_mask_shard_sha256"),"storage_version":"patch_mask_packbits_npy_shard_v2","packing":"np.packbits","bit_order":"little"}, expected_length=patch_count)
                grid = (int(record.get("grid_h", 0)), int(record.get("grid_w", 0)))
                assert_frozen_dataset_entry_v2_geometry(patch_count=patch_count, patch_grid=grid, patch_token_order_version=str(record.get("patch_token_order_version")), valid_content_mask=torch.from_numpy(valid.astype(bool)))
                return tensor, metadata, torch.from_numpy(valid.astype(bool))
            if self._canonical_expected_mode not in {"auto", mode}:
                raise ValueError(f"E1 canonical mode {mode} differs from expected_mode={self._canonical_expected_mode}.")
            canonical = self.image_release_root / str(record.get("canonical_image_path") or "")
            if not canonical.is_file() or sha256_file(canonical) != str(record.get("canonical_image_sha256") or ""):
                raise ValueError(f"Frozen canonical image is missing or hash-mismatched for {_record_identity(record)}.")
            tensor = torch.from_numpy(np.asarray(np.load(canonical, allow_pickle=False), dtype=np.float32))
            if tuple(tensor.shape) != (3, *target_size):
                raise ValueError("Frozen canonical image shape differs from formal target geometry.")
            patch_count = int(record.get("patch_count", 0))
            valid = self._v2_masks.read({"shard_key":record.get("valid_content_mask_shard_key"),"offset":int(record.get("valid_content_mask_offset",-1)),"length":patch_count,"storage_length":(patch_count+7)//8,"dtype":"uint8","shape":[patch_count],"sha256":record.get("valid_content_mask_shard_sha256"),"storage_version":"patch_mask_packbits_npy_shard_v2","packing":"np.packbits","bit_order":"little"}, expected_length=patch_count)
            grid = (int(record.get("grid_h", 0)), int(record.get("grid_w", 0)))
            assert_frozen_dataset_entry_v2_geometry(patch_count=patch_count, patch_grid=grid, patch_token_order_version=str(record.get("patch_token_order_version")), valid_content_mask=torch.from_numpy(valid.astype(bool)))
            return tensor, {"transform_policy":"frozen_dataset_entry_v2","transform_spec_checksum":str(record.get("image_geometry_sha256")),"transform_geometry":{},"transform_geometry_checksum":str(record.get("image_geometry_sha256"))}, torch.from_numpy(valid.astype(bool))
        if image_path.suffix.lower() in DICOM_SUFFIXES:
            image = load_dicom_pil_image(image_path)
            raw = pil_to_raw_tensor(image, mode="rgb")
            transformed, metadata = self._apply_record_spatial_transform(raw, record)
            valid_mask = self._valid_content_patch_mask(
                image_transform=metadata,
                record=record,
            )
            return transformed, metadata, valid_mask

        with Image.open(image_path) as image:
            raw = pil_to_raw_tensor(image, mode="rgb")
        transformed, metadata = self._apply_record_spatial_transform(raw, record)
        valid_mask = self._valid_content_patch_mask(
            image_transform=metadata,
            record=record,
        )
        return transformed, metadata, valid_mask

    def _validate_required_prior_paths(self) -> None:
        checked_roots: dict[str, bool] = {}
        for record in self.records:
            identity = _record_identity(record)
            root_value = str(record.get("resolved_root") or "").strip()
            if root_value and root_value not in checked_roots:
                checked_roots[root_value] = Path(root_value).exists()
            if root_value and not checked_roots[root_value]:
                raise FileNotFoundError(f"Runtime prior root does not exist for {identity}: {root_value}")
            for field_name in REQUIRED_PRIOR_FIELDS:
                raw_value = record.get(field_name)
                if not str(raw_value or "").strip():
                    raise ValueError(f"Manifest {identity} is missing required field: {field_name}")

    def _load_attention_map(self, record: dict[str, Any]) -> torch.Tensor:
        tensor, _metadata = self._load_attention_map_with_metadata(record)
        return tensor

    def _load_attention_map_with_metadata(
        self,
        record: dict[str, Any],
        image_transform: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        target_h, target_w = self._target_size_for_record(record)
        attention_path = _resolve_optional_path(record.get("attention_map_path"), self.manifest_dir)
        if attention_path is None and self.attention_map_dir is not None:
            attention_path = (self.attention_map_dir / f"{record['image_id']}.png").resolve()

        if attention_path is None or not attention_path.exists():
            if str(record.get("gaze_supervision_source", "")).strip().lower() == "no_gaze":
                raw = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            else:
                raw = torch.ones((1, target_h, target_w), dtype=torch.float32)
            return self._apply_record_spatial_transform(raw, record)

        if attention_path.suffix.lower() == ".npy":
            attention_map = _load_single_channel_npy(attention_path).clamp_min_(0.0)
            if str(record.get("gaze_sidecar_coordinate_space", "")).strip() == "canonical_target_canvas":
                if tuple(attention_map.shape[-2:]) != (target_h, target_w):
                    raise ValueError(
                        f"Canonical attention sidecar has shape {tuple(attention_map.shape[-2:])}, "
                        f"expected {(target_h, target_w)} for {_record_identity(record)}."
                    )
                if image_transform is None:
                    raise ValueError("Canonical gaze sidecar requires image transform geometry.")
                attention_map = attention_map
                metadata = dict(image_transform)
            else:
                attention_map, metadata = self._apply_record_spatial_transform(attention_map, record)
            max_value = float(attention_map.max().item()) if attention_map.numel() > 0 else 0.0
            if max_value > 0.0:
                return attention_map / max_value, metadata

            warnings.warn(
                f"Attention map contains no positive values, keeping zeros: {attention_path}",
                stacklevel=2,
            )
            return attention_map, metadata

        with Image.open(attention_path) as image:
            raw = pil_to_raw_tensor(image, mode="gray")
        return self._apply_record_spatial_transform(raw, record)

    def _load_high_conf_mask(self, record: dict[str, Any]) -> torch.Tensor:
        tensor, _metadata = self._load_high_conf_mask_with_metadata(record)
        return tensor

    def _load_high_conf_mask_with_metadata(
        self,
        record: dict[str, Any],
        image_transform: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        target_h, target_w = self._target_size_for_record(record)
        mask_path = _resolve_optional_path(record.get("high_conf_mask_path"), self.manifest_dir)
        if mask_path is None or not mask_path.exists():
            raw = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            return self._apply_record_spatial_transform(raw, record, is_mask=True)

        if mask_path.suffix.lower() == ".npy":
            mask = _load_single_channel_npy(mask_path)
            if str(record.get("gaze_sidecar_coordinate_space", "")).strip() == "canonical_target_canvas":
                if tuple(mask.shape[-2:]) != (target_h, target_w):
                    raise ValueError(
                        f"Canonical high-conf sidecar has shape {tuple(mask.shape[-2:])}, "
                        f"expected {(target_h, target_w)} for {_record_identity(record)}."
                    )
                if image_transform is None:
                    raise ValueError("Canonical gaze sidecar requires image transform geometry.")
                metadata = dict(image_transform)
            else:
                mask, metadata = self._apply_record_spatial_transform(mask, record, is_mask=True)
            return (mask >= 0.5).to(dtype=torch.float32), metadata

        with Image.open(mask_path) as image:
            raw = pil_to_raw_tensor(image, mode="gray")
        mask, metadata = self._apply_record_spatial_transform(raw, record, is_mask=True)
        return (mask >= 0.5).to(dtype=torch.float32), metadata

    def _resolve_text_prompt(self, record: dict[str, Any]) -> tuple[str, str, str]:
        lookup_prompt = self.prompt_lookup.get(str(record["image_id"]).strip(), "").strip()
        if lookup_prompt:
            return lookup_prompt, PROMPT_SOURCE_TEXT_PROMPT_PATH, ""
        raise ValueError(
            "V6 Stage 1 requires an Effective Report text_prompt for "
            f"{_record_identity(record)}; Structured Prompt construction is forbidden at runtime."
        )

    def _resolve_teacher_latent_path(self, record: dict[str, Any]) -> Path | None:
        latent_path = _resolve_optional_path(record.get("teacher_latent_path"), self.manifest_dir)
        if latent_path is None and self.teacher_latent_dir is not None:
            latent_path = (self.teacher_latent_dir / f"{record['image_id']}.npy").resolve()
        return latent_path

    def _resolve_teacher_manifest_entry(self, image_id: str) -> dict[str, Any]:
        return self.teacher_manifest_lookup.get(str(image_id).strip(), {})

    def _resolve_teacher_source_type(
        self,
        image_id: str,
        teacher_latent_path: Path | None,
    ) -> str:
        manifest_entry = self._resolve_teacher_manifest_entry(image_id)
        source_type = str(manifest_entry.get("source_type", "")).strip()
        if source_type:
            return source_type
        if teacher_latent_path is not None and teacher_latent_path.exists():
            return TEACHER_SOURCE_DETERMINISTIC_FIXTURE
        return TEACHER_SOURCE_MISSING

    def summarize_teacher_latent_availability(self) -> dict[str, int]:
        deterministic_count = 0
        real_clip_count = 0
        missing_count = 0
        for record in self.records:
            latent_path = self._resolve_teacher_latent_path(record)
            source_type = self._resolve_teacher_source_type(
                image_id=str(record["image_id"]).strip(),
                teacher_latent_path=latent_path,
            )
            if (
                latent_path is None
                or not latent_path.exists()
                or latent_path.suffix.lower() != ".npy"
            ):
                missing_count += 1
            elif source_type == TEACHER_SOURCE_REAL_CLIP_IMAGE:
                real_clip_count += 1
            else:
                deterministic_count += 1
        return {
            "total_samples": len(self.records),
            f"{TEACHER_SOURCE_DETERMINISTIC_FIXTURE}_count": deterministic_count,
            f"{TEACHER_SOURCE_REAL_CLIP_IMAGE}_count": real_clip_count,
            f"{TEACHER_SOURCE_MISSING}_count": missing_count,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = _resolve_optional_path(record.get("image_path"), self.manifest_dir)
        if self.dataset_entry_v2_enabled:
            if resolve_canonical_image_mode(record.get("canonical_image_mode")) == CANONICAL_IMAGE_MODE_SOURCE_RUNTIME:
                image_path = _resolve_optional_path(record.get("source_image_path"), self.manifest_dir)
            else:
                image_path = self.image_release_root / str(record.get("canonical_image_path") or "")
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist for sample {record['image_id']}: {image_path}"
            )

        teacher_latent_path = self._resolve_teacher_latent_path(record)
        teacher_source_type = self._resolve_teacher_source_type(
            image_id=str(record["image_id"]).strip(),
            teacher_latent_path=teacher_latent_path,
        )
        teacher_manifest_entry = self._resolve_teacher_manifest_entry(str(record["image_id"]).strip())

        text_prompt, text_prompt_source, text_prompt_warning = self._resolve_text_prompt(record)
        if text_prompt_warning:
            warnings.warn(text_prompt_warning, stacklevel=2)

        image_tensor, image_transform, valid_content_patch_mask = (
            self._load_image_tensor_with_metadata(image_path, record)
        )
        if self.dataset_entry_v2_enabled:
            # E1 canonical pixels already carry the frozen resize/pad result.
            # Reapplying the legacy transform (or resolving source priors) would
            # silently create a second geometry authority.
            attention_map = torch.zeros_like(image_tensor[:1])
            high_conf_mask = torch.zeros_like(image_tensor[:1])
            attention_transform = image_transform
            high_conf_transform = image_transform
            patch_count = int(record["patch_count"])
            if int(record.get("gaze_training_enabled", 0)) == 0:
                patch_gaze_weight = torch.zeros(patch_count, dtype=torch.float32)
                gaze_status = "NO_GAZE"
            else:
                patch_gaze_weight = torch.from_numpy(
                    self._v2_gaze.read(
                        {
                            "gaze_weight_shard_key": record.get("gaze_weight_shard_key"),
                            "shard_sha256": record.get("gaze_shard_sha256"),
                            "offset": int(record.get("gaze_offset", -1)),
                            "length": int(record.get("gaze_length", -1)),
                            "dtype": "float32",
                            "shape": [patch_count],
                            "slice_sha256": record.get("gaze_slice_sha256"),
                            "gaze_weight_sha256": record.get("gaze_slice_sha256"),
                            "storage_version": "gaze_weight_npy_shard_v1",
                        },
                        expected_length=patch_count,
                    ).copy()
                )
                gaze_status = str(record.get("gaze_status") or "")
                if gaze_status == "NO_GAZE":
                    raise ValueError("E2 no-gaze row must set gaze_training_enabled=0.")
        else:
            attention_map, attention_transform = self._load_attention_map_with_metadata(
                record,
                image_transform,
            )
            high_conf_mask, high_conf_transform = self._load_high_conf_mask_with_metadata(
                record,
                image_transform,
            )

        sample: dict[str, Any] = {
            "image": image_tensor,
            "image_id": str(record["image_id"]).strip(),
            "manifest_row_index": int(record.get("__manifest_row_index__", 0)),
            "effective_report_sha256": str(record.get("effective_report_sha256", "")).strip().lower(),
            "image_path": str(image_path),
            "modality": str(record["modality"]).strip().lower(),
            "study_description": str(record.get("study_description", "")).strip(),
            "gaze_supervision_source": str(record.get("gaze_supervision_source", "")).strip(),
            "attention_map_path": str(record.get("attention_map_path", "")).strip(),
            "high_conf_mask_path": str(record.get("high_conf_mask_path", "")).strip(),
            "patch_gaze_weight_path": str(record.get("patch_gaze_weight_path", "")).strip(),
            "prior_status": str(record.get("prior_status", "")).strip(),
            "audit_status": str(record.get("audit_status", "")).strip(),
            "coverage_ratio": str(record.get("coverage_ratio", "")).strip(),
            "high_conf_area_ratio": str(record.get("high_conf_area_ratio", "")).strip(),
            "inside_ratio": str(record.get("inside_ratio", "")).strip(),
            "text_prompt": text_prompt,
            "text_prompt_source": text_prompt_source,
            "text_prompt_warning": text_prompt_warning,
            "attention_map": attention_map,
            "high_conf_mask": high_conf_mask,
            "valid_content_patch_mask": valid_content_patch_mask,
            "transform_policy": str(image_transform["transform_policy"]),
            "transform_spec_checksum": str(image_transform["transform_spec_checksum"]),
            "image_transform_geometry_json": json.dumps(
                image_transform["transform_geometry"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "image_transform_geometry_checksum": str(
                image_transform["transform_geometry_checksum"]
            ),
            "attention_transform_spec_checksum": str(
                attention_transform["transform_spec_checksum"]
            ),
            "attention_transform_geometry_json": json.dumps(
                attention_transform["transform_geometry"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "attention_transform_geometry_checksum": str(
                attention_transform["transform_geometry_checksum"]
            ),
            "high_conf_mask_transform_spec_checksum": str(
                high_conf_transform["transform_spec_checksum"]
            ),
            "high_conf_mask_transform_geometry_json": json.dumps(
                high_conf_transform["transform_geometry"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "high_conf_mask_transform_geometry_checksum": str(
                high_conf_transform["transform_geometry_checksum"]
            ),
            "teacher_latent_path": str(teacher_latent_path) if teacher_latent_path is not None else "",
            "teacher_latent_source_type": teacher_source_type,
            "teacher_model_name": str(teacher_manifest_entry.get("teacher_model_name", "")).strip(),
            "teacher_latent_exists": bool(
                teacher_latent_path is not None and teacher_latent_path.exists()
            ),
        }
        if self.dataset_entry_v2_enabled:
            sample["patch_gaze_weight"] = patch_gaze_weight
            sample["gaze_status"] = gaze_status
        for concept in CONCEPT_TARGET_FIELDS:
            sample[concept] = str(record.get(concept, "")).strip()
            for suffix in CONCEPT_TARGET_METADATA_SUFFIXES:
                key = f"{concept}_{suffix}"
                sample[key] = str(record.get(key, "")).strip()
        return sample

    def source_runtime_cache_stats(self) -> dict[str, int]:
        """Per-worker local cache counters; not formal training provenance."""
        if not self.dataset_entry_v2_enabled:
            return {}
        return self._source_runtime_cache.stats.to_dict()
