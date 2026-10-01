"""Runtime projection reader for materialized P0-B sparse concept targets."""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from breast_pretrain.data.stage1_sparse_concept_contract import FrozenConceptSchema, RuntimeConceptTarget, load_frozen_concept_schema


@dataclass(frozen=True)
class ConceptRuntimeAssets:
    targets_path: Path
    manifest_path: Path
    schema_sha256: str
    rows_by_image_id: dict[str, int]
    effective_report_hashes: dict[str, str]


def load_concept_runtime_assets(
    targets_path: str | Path,
    manifest_path: str | Path,
    schema: FrozenConceptSchema,
) -> ConceptRuntimeAssets:
    """Open deterministic assets and fail closed on identity/schema drift."""
    target_path, csv_path = Path(targets_path), Path(manifest_path)
    if not target_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Formal sparse concept runtime assets are required; legacy fallback is forbidden.")
    archive = np.load(target_path, allow_pickle=False)
    embedded_hash = str(archive.get("concept_schema_sha256", ""))
    if embedded_hash != schema.yaml_sha256:
        raise ValueError("Formal concept runtime schema hash mismatch.")
    rows_by_image_id: dict[str, int] = {}
    hashes: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "row_index", "effective_report_sha256", "train_eligible", "clinical_graph_enabled"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("stage1_concept_manifest.csv misses formal identity/eligibility fields.")
        for row in reader:
            image_id = str(row["image_id"]).strip()
            if not image_id or image_id in rows_by_image_id:
                raise ValueError("Concept runtime manifest contains an empty or duplicate image_id.")
            if str(row["train_eligible"]).lower() not in {"1", "true"} or str(row["clinical_graph_enabled"]).lower() not in {"1", "true"}:
                raise ValueError("Graph-excluded/ineligible rows must not enter formal concept runtime assets.")
            rows_by_image_id[image_id] = int(row["row_index"])
            hashes[image_id] = str(row["effective_report_sha256"])
    return ConceptRuntimeAssets(target_path, csv_path, schema.yaml_sha256, rows_by_image_id, hashes)


def runtime_target_from_npz(
    archive: np.lib.npyio.NpzFile,
    spec_id: str,
    row_indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return target, concept mask and optional per-value mask without broadcasting."""
    ids = np.asarray(list(row_indices), dtype=np.int64)
    target_key = f"{spec_id}__target"
    mask_key = f"{spec_id}__valid_target_mask"
    value_mask_key = f"{spec_id}__value_valid_mask"
    if target_key not in archive or mask_key not in archive:
        raise ValueError(f"Formal runtime assets lack required tensors for {spec_id}.")
    target = archive[target_key][ids]
    mask = archive[mask_key][ids].astype(bool)
    value_mask = archive[value_mask_key][ids].astype(bool) if value_mask_key in archive else None
    if target.ndim == 2 and value_mask is None:
        raise ValueError(f"Multilabel formal target {spec_id} lacks per-value valid mask.")
    return target, mask, value_mask


def runtime_lineage_digest(image_ids: Iterable[str], hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{image_id}\t{hashes[image_id]}" for image_id in sorted(image_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_tensor_target(target: RuntimeConceptTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Small materializer helper used only after a frozen roster is supplied."""
    return np.asarray(target.target), np.asarray(target.valid_target_mask, dtype=np.bool_), target.value_valid_mask


@lru_cache(maxsize=4)
def _cached_assets(schema_path: str, schema_hash: str, targets_path: str, manifest_path: str):
    schema = load_frozen_concept_schema(schema_path, expected_sha256=schema_hash)
    return schema, load_concept_runtime_assets(targets_path, manifest_path, schema)


def load_formal_p0b_batch_targets(
    p0b: dict[str, object], image_ids: Iterable[str], effective_report_hashes: Iterable[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load only materialized sparse-target tensors; manifest metadata never enters."""
    required = ("concept_schema_path", "concept_schema_sha256", "concept_runtime_npz_path", "concept_runtime_manifest_path")
    if any(not p0b.get(key) for key in required):
        raise ValueError("Formal P0-B runtime requires sparse target assets; legacy metadata fallback is forbidden.")
    schema, assets = _cached_assets(*(str(p0b[key]) for key in required))
    image_ids = list(image_ids)
    hashes = list(effective_report_hashes)
    if len(image_ids) != len(hashes):
        raise ValueError("Formal P0-B image/hash batch length mismatch.")
    indices: list[int] = []
    for image_id, report_hash in zip(image_ids, hashes):
        if image_id not in assets.rows_by_image_id or assets.effective_report_hashes[image_id] != report_hash:
            raise ValueError(f"Formal P0-B effective_report_sha256 lineage mismatch for image_id={image_id}")
        indices.append(assets.rows_by_image_id[image_id])
    targets: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    value_masks: dict[str, np.ndarray] = {}
    with np.load(assets.targets_path, allow_pickle=False) as archive:
        for concept_id in schema.direct_concept_ids:
            target, mask, value_mask = runtime_target_from_npz(archive, concept_id, indices)
            targets[concept_id], masks[concept_id] = target, mask
            if value_mask is not None:
                value_masks[concept_id] = value_mask
    return targets, masks, value_masks
