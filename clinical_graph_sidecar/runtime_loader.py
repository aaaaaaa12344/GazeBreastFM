from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from breast_pretrain.clinical_graph_sidecar.status_contract import (
    ALLOWED_STATUSES,
    OBSERVED_STATUSES,
    observed_mask_for,
)
from breast_pretrain.clinical_graph_sidecar.index import ClinicalGraphSidecarIndex


@dataclass(frozen=True)
class ClinicalGraphV2Sample:
    """One evidence-only V2 clinical graph vector aligned to the canonical node order."""

    node_values: torch.Tensor
    observed_mask: torch.Tensor
    node_index: torch.Tensor


class ClinicalGraphV2SidecarLoader:
    """Read V2 JSONL sidecars without introducing a row-order runtime dependency.

    The manifest-to-sidecar join is by stable sample/image identity.  Values at
    unobserved positions are zeroed here and again in ``prepare_stage1_joint_batch``
    so missing evidence cannot become a negative target or graph feature.
    """

    def __init__(
        self,
        *,
        sidecar_path: str | Path,
        nodes_path: str | Path,
        index_path: str | Path | None = None,
    ) -> None:
        self.sidecar_path = Path(sidecar_path).expanduser().resolve()
        self.nodes_path = Path(nodes_path).expanduser().resolve()
        if not self.sidecar_path.is_file():
            raise FileNotFoundError(f"Clinical Graph V2 sidecar not found: {self.sidecar_path}")
        if not self.nodes_path.is_file():
            raise FileNotFoundError(f"Clinical Graph V2 nodes file not found: {self.nodes_path}")

        self.node_ids = self._load_node_ids(self.nodes_path)
        self.node_index = torch.arange(len(self.node_ids), dtype=torch.long)
        self._by_sample_id: dict[str, ClinicalGraphV2Sample] = {}
        self._by_image_id: dict[str, ClinicalGraphV2Sample] = {}
        self._index = ClinicalGraphSidecarIndex(index_path) if index_path is not None else None
        self._handle = None
        self._handle_pid: int | None = None
        self._handle_lock = threading.Lock()
        if self._index is None:
            # Kept only for backwards-compatible non-formal callers. Formal
            # runtime must pass index_path and therefore never eagerly parses
            # the frozen multi-gigabyte sidecar.
            self._load_sidecar()

    @staticmethod
    def _load_node_ids(nodes_path: Path) -> tuple[str, ...]:
        if nodes_path.suffix.lower() == ".jsonl":
            node_ids: list[str] = []
            try:
                with nodes_path.open("r", encoding="utf-8-sig") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Clinical Graph V2 nodes JSONL line {line_number} is not valid JSON: {nodes_path}"
                            ) from exc
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"Clinical Graph V2 nodes JSONL line {line_number} is not an object: {nodes_path}"
                            )
                        node_id = str(row.get("node_id", "")).strip()
                        if node_id:
                            node_ids.append(node_id)
                if not node_ids:
                    raise ValueError(
                        "Clinical Graph V2 nodes JSONL has no canonical node_id field; "
                        "provide the frozen node-index JSON/CSV instead: " + str(nodes_path)
                    )
            except UnicodeDecodeError as exc:
                raise ValueError(f"Clinical Graph V2 nodes JSONL is not UTF-8: {nodes_path}") from exc
        elif nodes_path.suffix.lower() == ".json":
            try:
                payload = json.loads(nodes_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Clinical Graph V2 node index JSON is unreadable: {nodes_path}") from exc
            if isinstance(payload, dict):
                ordered = sorted(payload.items(), key=lambda item: int(item[1]))
                node_ids = [str(key).strip() for key, _ in ordered]
            elif isinstance(payload, list):
                node_ids = [str(item.get("node_id", item.get("canonical_concept", ""))).strip() for item in payload if isinstance(item, dict)]
            else:
                raise ValueError(f"Clinical Graph V2 node index JSON must be an object or list: {nodes_path}")
        else:
            node_ids = []
        if node_ids:
            if any(not node_id for node_id in node_ids):
                raise ValueError(f"Clinical Graph V2 nodes file has an empty node_id: {nodes_path}")
            if len(set(node_ids)) != len(node_ids):
                raise ValueError(f"Clinical Graph V2 nodes file has duplicate node_id values: {nodes_path}")
            return tuple(node_ids)
        with nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        node_ids = tuple(str(row.get("node_id", "")).strip() for row in rows)
        if not node_ids or any(not node_id for node_id in node_ids):
            raise ValueError(f"Clinical Graph V2 nodes file has an empty node_id: {nodes_path}")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"Clinical Graph V2 nodes file has duplicate node_id values: {nodes_path}")
        return node_ids

    @staticmethod
    def _identity(row: dict[str, Any], key: str, line_number: int) -> str:
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(
                f"Clinical Graph V2 sidecar line {line_number} is missing {key}."
            )
        return value

    def _parse_sample(self, row: dict[str, Any], line_number: int) -> ClinicalGraphV2Sample:
        values = row.get("concept_values")
        masks = row.get("observed_mask")
        statuses = row.get("status")
        source_types = row.get("source_type", {})
        if not isinstance(values, dict) or not isinstance(masks, dict) or not isinstance(statuses, dict):
            raise ValueError(
                "Clinical Graph V2 sidecar line "
                f"{line_number} must contain concept_values, observed_mask, and status mappings."
            )

        missing = [node_id for node_id in self.node_ids if node_id not in values or node_id not in masks or node_id not in statuses]
        extra = sorted((set(values) | set(masks) | set(statuses)) - set(self.node_ids))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing node ids: {missing[:5]}")
            if extra:
                details.append(f"unknown node ids: {extra[:5]}")
            raise ValueError(
                f"Clinical Graph V2 sidecar line {line_number} does not match canonical node index ({'; '.join(details)})."
            )

        node_values = torch.zeros(len(self.node_ids), dtype=torch.float32)
        observed = torch.zeros(len(self.node_ids), dtype=torch.bool)
        for index, node_id in enumerate(self.node_ids):
            status = str(statuses[node_id]).strip()
            if status not in ALLOWED_STATUSES:
                raise ValueError(
                    f"Clinical Graph V2 sidecar line {line_number} has invalid status for {node_id}: {status!r}."
                )
            source_type = str(source_types.get(node_id, "")).strip() if isinstance(source_types, dict) else ""
            try:
                raw_mask = int(masks[node_id])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Clinical Graph V2 sidecar line {line_number} has non-binary observed_mask for {node_id}."
                ) from exc
            # V6 keeps high-confidence report-derived model inference as a
            # first-class graph input.  It must be explicitly labelled; a
            # generic inferred status cannot silently become an observation.
            permit_inferred = status == "inferred_high_confidence" and source_type == "model_inference"
            expected_mask = observed_mask_for(status, permit_inferred=permit_inferred)
            if raw_mask not in {0, 1} or raw_mask != expected_mask:
                raise ValueError(
                    "Clinical Graph V2 sidecar observed_mask/status contract violation at "
                    f"line {line_number}, node {node_id}: mask={raw_mask}, status={status!r}."
                )
            if status not in OBSERVED_STATUSES and not permit_inferred:
                continue
            observed[index] = True
            raw_value = values.get(node_id)
            if isinstance(raw_value, (int, float)) and torch.isfinite(torch.tensor(float(raw_value))):
                node_values[index] = float(raw_value) if status != "explicit_absent" else 0.0
            else:
                node_values[index] = 1.0 if status == "present" else 0.0

        return ClinicalGraphV2Sample(
            node_values=node_values,
            observed_mask=observed,
            node_index=self.node_index,
        )

    @staticmethod
    def _store_unique(mapping: dict[str, ClinicalGraphV2Sample], key: str, sample: ClinicalGraphV2Sample, *, field: str) -> None:
        if key in mapping:
            raise ValueError(f"Clinical Graph V2 sidecar has duplicate {field}: {key!r}.")
        mapping[key] = sample

    def _load_sidecar(self) -> None:
        with self.sidecar_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Clinical Graph V2 sidecar line {line_number} is not valid JSON."
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Clinical Graph V2 sidecar line {line_number} is not an object.")
                sample = self._parse_sample(row, line_number)
                self._store_unique(
                    self._by_sample_id,
                    self._identity(row, "sample_id", line_number),
                    sample,
                    field="sample_id",
                )
                self._store_unique(
                    self._by_image_id,
                    self._identity(row, "image_id", line_number),
                    sample,
                    field="image_id",
                )
        if not self._by_image_id:
            raise ValueError(f"Clinical Graph V2 sidecar is empty: {self.sidecar_path}")

    def _get_handle(self):
        import os

        pid = os.getpid()
        with self._handle_lock:
            if self._handle is None or self._handle_pid != pid:
                if self._handle is not None:
                    self._handle.close()
                self._handle = self.sidecar_path.open("rb")
                self._handle_pid = pid
            return self._handle

    def _lookup_lazy(self, key: str) -> ClinicalGraphV2Sample:
        record = self._index.lookup(key) if self._index is not None else None
        if record is None:
            raise KeyError(key)
        handle = self._get_handle()
        with self._handle_lock:
            handle.seek(record.offset)
            raw = handle.read(record.length)
        if len(raw) != record.length:
            raise IOError(
                f"Clinical Graph indexed read truncated for key={key!r}: "
                f"expected={record.length} observed={len(raw)}"
            )
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Clinical Graph indexed record is not valid JSON: key={key!r}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Clinical Graph indexed record is not an object: key={key!r}")
        sample = self._parse_sample(row, record.offset)
        sample_id = self._identity(row, "sample_id", record.offset)
        image_id = self._identity(row, "image_id", record.offset)
        if key not in {sample_id, image_id}:
            raise ValueError(f"Clinical Graph index key mismatch: requested={key!r}")
        return sample

    def lookup(self, record: dict[str, Any]) -> ClinicalGraphV2Sample:
        sample_id = str(record.get("sample_id", "")).strip()
        image_id = str(record.get("image_id", "")).strip()
        if self._index is not None:
            sample_key = sample_id or image_id
            if not sample_key:
                raise KeyError("Clinical Graph lookup requires sample_id or image_id.")
            if sample_id and image_id:
                sample_record = self._index.lookup(sample_id)
                image_record = self._index.lookup(image_id)
                if (sample_record.offset, sample_record.length) != (image_record.offset, image_record.length):
                    raise ValueError(
                        "Clinical Graph V2 sidecar identity mismatch for manifest record "
                        f"sample_id={sample_id!r}, image_id={image_id!r}."
                    )
            sample = self._lookup_lazy(sample_key)
            by_sample = sample if sample_id else None
            by_image = sample if image_id else None
        else:
            by_sample = self._by_sample_id.get(sample_id) if sample_id else None
            by_image = self._by_image_id.get(image_id) if image_id else None
        if by_sample is not None and by_image is not None and by_sample is not by_image and self._index is None:
            raise ValueError(
                "Clinical Graph V2 sidecar identity mismatch for manifest record "
                f"sample_id={sample_id!r}, image_id={image_id!r}."
            )
        sample = by_sample or by_image
        if sample is None:
            raise KeyError(
                "Clinical Graph V2 sidecar has no record for manifest sample "
                f"sample_id={sample_id!r}, image_id={image_id!r}."
            )
        return sample


__all__ = ["ClinicalGraphV2Sample", "ClinicalGraphV2SidecarLoader"]
