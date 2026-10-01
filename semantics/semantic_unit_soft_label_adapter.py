"""Read-only bridge from frozen Stage D semantic units to image batches."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breast_pretrain.semantics.semantic_soft_labels import SemanticSoftLabelBatch


RESOLVER_VERSION = "semantic_unit_soft_label_adapter_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class SemanticUnitTarget:
    semantic_unit_id: str
    semantic_target_index: int
    neighbor_indices: torch.Tensor
    values: torch.Tensor


class SemanticUnitSoftLabelAdapter:
    """Resolve frozen image->unit bindings without any runtime search."""

    def __init__(
        self,
        *,
        mapping_path: str | Path,
        topk_path: str | Path,
        source_root: str | Path | None = None,
        expected_schema_version: str | None = None,
        expected_binding_version: str | None = None,
        expected_mapping_sha256: str | None = None,
        expected_topk_sha256: str | None = None,
    ) -> None:
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        self.topk_path = Path(topk_path).expanduser().resolve()
        self.source_root = Path(source_root).expanduser().resolve() if source_root else self.mapping_path.parent
        for path, label in ((self.mapping_path, "semantic-unit mapping"), (self.topk_path, "semantic-unit top-k")):
            if not path.is_file():
                raise FileNotFoundError(f"Frozen {label} asset does not exist: {path}")
        self.mapping_sha256 = _sha256(self.mapping_path)
        self.topk_sha256 = _sha256(self.topk_path)
        if expected_mapping_sha256 and self.mapping_sha256 != str(expected_mapping_sha256).lower():
            raise ValueError("Frozen semantic-unit mapping SHA256 mismatch.")
        if expected_topk_sha256 and self.topk_sha256 != str(expected_topk_sha256).lower():
            raise ValueError("Frozen semantic-unit top-k SHA256 mismatch.")

        self._image_to_target: dict[str, SemanticUnitTarget] = {}
        with self.mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Frozen semantic-unit mapping is empty: {self.mapping_path}")
        required = {"image_id", "semantic_unit_id", "semantic_unit_index"}
        missing = sorted(required.difference(rows[0]))
        if missing:
            raise ValueError("Frozen semantic-unit mapping misses fields: " + ", ".join(missing))

        payload = np.load(self.topk_path, allow_pickle=False)
        required_arrays = {"indptr", "indices", "scores"}
        missing_arrays = sorted(required_arrays.difference(payload.files))
        if missing_arrays:
            raise ValueError("Frozen semantic-unit top-k misses arrays: " + ", ".join(missing_arrays))
        self.indptr = np.asarray(payload["indptr"], dtype=np.int64)
        self.indices = np.asarray(payload["indices"], dtype=np.int64)
        self.scores = np.asarray(payload["scores"], dtype=np.float32)
        if (
            self.indptr.ndim != 1
            or len(self.indptr) < 2
            or len(self.indices) != len(self.scores)
            or int(self.indptr[0]) != 0
            or np.any(np.diff(self.indptr) < 0)
            or int(self.indptr[-1]) != len(self.indices)
        ):
            raise ValueError("Frozen semantic-unit top-k sparse arrays have invalid shapes.")
        unit_ids_array = payload["semantic_unit_ids"] if "semantic_unit_ids" in payload.files else (payload["image_ids"] if "image_ids" in payload.files else None)
        if unit_ids_array is None:
            unit_count = len(self.indptr) - 1
            indexed_ids: dict[int, str] = {}
            jsonl_path = self.topk_path.with_suffix(".jsonl")
            if jsonl_path.is_file():
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle):
                        record = json.loads(line)
                        unit_index = int(record.get("query_semantic_unit_index", -1))
                        unit_id = _text(record.get("semantic_unit_id"))
                        if not unit_id or not 0 <= unit_index < unit_count:
                            raise ValueError(f"Frozen top-k JSONL has invalid unit binding at line {line_number}.")
                        prior = indexed_ids.setdefault(unit_index, unit_id)
                        if prior != unit_id:
                            raise ValueError(f"Frozen top-k JSONL has conflicting ID at unit index {unit_index}.")
            else:
                for row_number, row in enumerate(rows):
                    try:
                        unit_index = int(row.get("semantic_unit_index", -1))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Semantic-unit mapping has invalid unit index at row {row_number}.") from exc
                    unit_id = _text(row.get("semantic_unit_id"))
                    if not unit_id or not 0 <= unit_index < unit_count:
                        raise ValueError(f"Semantic-unit mapping has invalid unit binding at row {row_number}.")
                    prior = indexed_ids.setdefault(unit_index, unit_id)
                    if prior != unit_id:
                        raise ValueError(f"Semantic-unit mapping has conflicting ID at unit index {unit_index}.")
            if len(indexed_ids) != unit_count:
                raise ValueError("Frozen semantic-unit mapping does not cover every top-k unit index.")
            unit_ids_array = [indexed_ids[index] for index in range(unit_count)]
        self.unit_ids = tuple(_text(item) for item in unit_ids_array.tolist()) if hasattr(unit_ids_array, "tolist") else tuple(_text(item) for item in unit_ids_array)
        unit_count = len(self.indptr) - 1
        if len(self.unit_ids) != unit_count or len(set(self.unit_ids)) != unit_count:
            raise ValueError("Frozen semantic-unit top-k IDs do not align with indptr rows.")
        if self.indices.size and (int(self.indices.min()) < 0 or int(self.indices.max()) >= unit_count):
            raise ValueError("Frozen semantic-unit top-k target index is out of range.")
        if not np.isfinite(self.scores).all() or np.any(self.scores < 0.0) or np.any(self.scores > 1.0):
            raise ValueError("Frozen semantic-unit top-k scores must be finite and in [0, 1].")
        self.max_neighbors = max(int(self.indptr[i + 1] - self.indptr[i]) for i in range(unit_count))
        for row_number, row in enumerate(rows):
            image_id = _text(row.get("image_id"))
            unit_id = _text(row.get("semantic_unit_id"))
            if not image_id or image_id in self._image_to_target:
                raise ValueError(f"Semantic-unit mapping has duplicate/empty image_id at row {row_number}.")
            try:
                unit_index = int(row.get("semantic_unit_index", -1))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Semantic-unit mapping has invalid unit index at row {row_number}.") from exc
            if not 0 <= unit_index < unit_count or self.unit_ids[unit_index] != unit_id:
                raise ValueError(f"Semantic-unit mapping row {row_number} does not match frozen top-k IDs.")
            start, end = int(self.indptr[unit_index]), int(self.indptr[unit_index + 1])
            indices = np.full(self.max_neighbors, -1, dtype=np.int64)
            scores = np.zeros(self.max_neighbors, dtype=np.float32)
            length = end - start
            indices[:length] = self.indices[start:end]
            scores[:length] = self.scores[start:end]
            self._image_to_target[image_id] = SemanticUnitTarget(
                semantic_unit_id=unit_id,
                semantic_target_index=unit_index,
                neighbor_indices=torch.from_numpy(indices),
                values=torch.from_numpy(scores),
            )

        self.schema_version = _text(expected_schema_version) or self._read_metadata_schema_version()
        self.binding_version = _text(expected_binding_version) or "stage1_semantic_unit_binding_v1"

    def _read_metadata_schema_version(self) -> str:
        for name in ("resolved_config.json", "source_authority_lineage.json", "validation.json"):
            path = self.source_root / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            value = payload.get("schema_version") if isinstance(payload, dict) else None
            if value:
                return _text(value)
        return "stage1_semantic_unit_bounded_retrieval_v1"

    @property
    def source_metadata(self) -> dict[str, str]:
        return {
            "source_root": str(self.source_root),
            "mapping_path": str(self.mapping_path),
            "mapping_sha256": self.mapping_sha256,
            "topk_path": str(self.topk_path),
            "topk_sha256": self.topk_sha256,
            "schema_version": self.schema_version,
            "binding_version": self.binding_version,
            "resolver_version": RESOLVER_VERSION,
        }

    def lookup(self, image_id: str) -> SemanticUnitTarget:
        key = _text(image_id)
        target = self._image_to_target.get(key)
        if target is None:
            raise KeyError(f"No frozen semantic-unit binding for image_id={key!r}.")
        return target

    def resolve_batch(
        self,
        image_ids: list[str],
        row_indices: list[int] | None = None,
        semantic_target_indices: torch.Tensor | None = None,
    ) -> SemanticSoftLabelBatch:
        targets = [self.lookup(image_id) for image_id in image_ids]
        if semantic_target_indices is not None:
            expected = torch.tensor([item.semantic_target_index for item in targets], dtype=torch.long)
            if not torch.equal(semantic_target_indices.detach().cpu().to(dtype=torch.long), expected):
                raise ValueError("Batch semantic_target_index differs from frozen image-to-unit binding.")
        unit_indices = [item.semantic_target_index for item in targets]
        matrix = torch.zeros((len(targets), len(targets)), dtype=torch.float32)
        positions_by_unit: dict[int, list[int]] = {}
        for position, unit_index in enumerate(unit_indices):
            positions_by_unit.setdefault(unit_index, []).append(position)
        for source_position, unit_index in enumerate(unit_indices):
            start, end = int(self.indptr[unit_index]), int(self.indptr[unit_index + 1])
            for target_index, score in zip(self.indices[start:end], self.scores[start:end]):
                for target_position in positions_by_unit.get(int(target_index), ()):
                    matrix[source_position, target_position] = max(
                        float(matrix[source_position, target_position]), float(score)
                    )
        matrix = torch.maximum(matrix, matrix.transpose(0, 1))
        matrix.fill_diagonal_(1.0)
        return SemanticSoftLabelBatch(
            matrix=matrix.clamp_(0.0, 1.0),
            batch_positions=list(range(len(image_ids))),
            image_ids=[_text(item) for item in image_ids],
            warnings=[],
        )


__all__ = ["RESOLVER_VERSION", "SemanticUnitSoftLabelAdapter", "SemanticUnitTarget"]
