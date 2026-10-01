from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PATH_LIKE_SOURCE_KEYS = {
    "primary_manifest",
    "metadata_csv",
    "breast_annotations",
    "finding_annotations",
    "text_prompt_jsonl",
    "text_prompt_csv",
    "gaze_manifest",
    "teacher_latent_manifest",
    "teacher_latent_dir",
    "report_manifest",
    "dataset_root",
}

PATH_LIST_SOURCE_KEYS = {
    "teacher_latent_manifest_candidates",
    "teacher_latent_dir_candidates",
}


@dataclass(frozen=True)
class DatasetRegistryEntry:
    name: str
    adapter: str
    dataset_root: Path
    modality: str
    sub_modality: str
    data_role: str
    enabled: bool
    official_split_priority: bool
    small_precise_test_set: bool
    downstream_only: bool
    external_eval: bool
    gaze_holdout_only: bool
    default_test_bucket: str
    notes: str
    source: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)


def _resolve_path(raw_value: Any, project_root: Path) -> Path:
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _normalize_source(raw_source: Any, project_root: Path) -> dict[str, Any]:
    if raw_source is None:
        return {}
    if not isinstance(raw_source, dict):
        raise ValueError("dataset source must be a mapping when provided.")

    normalized: dict[str, Any] = {}
    for key, value in raw_source.items():
        if value is None:
            normalized[key] = None
        elif key in PATH_LIST_SOURCE_KEYS:
            if not isinstance(value, list):
                raise ValueError(f"dataset source field '{key}' must be a list when provided.")
            normalized[key] = [_resolve_path(item, project_root) for item in value]
        elif key in PATH_LIKE_SOURCE_KEYS or key.endswith("_path") or key.endswith("_dir"):
            normalized[key] = _resolve_path(value, project_root)
        else:
            normalized[key] = value
    return normalized


def load_dataset_registry(config_path: str | Path) -> list[DatasetRegistryEntry]:
    registry_path = Path(config_path).expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(f"Dataset registry config does not exist: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset registry config must contain a mapping: {registry_path}")

    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("dataset_registry.yaml must contain a top-level 'datasets' list.")

    project_root = registry_path.parents[1]
    entries: list[DatasetRegistryEntry] = []
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("Each dataset registry entry must be a mapping.")
        name = str(item.get("name", "")).strip()
        adapter = str(item.get("adapter", "base")).strip()
        if not name:
            raise ValueError("Dataset registry entry is missing required field: name")
        dataset_root_raw = item.get("dataset_root", project_root / "data" / name)
        entries.append(
            DatasetRegistryEntry(
                name=name,
                adapter=adapter or "base",
                dataset_root=_resolve_path(dataset_root_raw, project_root),
                modality=str(item.get("modality", "")).strip().lower(),
                sub_modality=str(item.get("sub_modality", "")).strip().lower(),
                data_role=str(item.get("data_role", "pretrain")).strip().lower(),
                enabled=bool(item.get("enabled", True)),
                official_split_priority=bool(item.get("official_split_priority", True)),
                small_precise_test_set=bool(item.get("small_precise_test_set", False)),
                downstream_only=bool(item.get("downstream_only", False)),
                external_eval=bool(item.get("external_eval", False)),
                gaze_holdout_only=bool(item.get("gaze_holdout_only", False)),
                default_test_bucket=str(
                    item.get("default_test_bucket", "internal_test")
                ).strip(),
                notes=str(item.get("notes", "")).strip(),
                source=_normalize_source(item.get("source"), project_root),
                defaults=dict(item.get("defaults", {}) or {}),
            )
        )
    return entries
