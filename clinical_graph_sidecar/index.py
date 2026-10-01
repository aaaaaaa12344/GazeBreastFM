from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INDEX_SCHEMA_VERSION = "clinical_graph_sidecar_index_v1"


@dataclass(frozen=True)
class ClinicalGraphIndexRecord:
    key: str
    offset: int
    length: int


class ClinicalGraphSidecarIndex:
    """Small identity-to-byte-range index for a frozen JSONL sidecar.

    Only offsets and identity keys are resident in memory.  The graph payload
    remains in the immutable source JSONL and is parsed on demand.
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Clinical Graph sidecar index not found: {self.index_path}")
        self._records: dict[str, ClinicalGraphIndexRecord] = {}
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    key = str(row["key"]).strip()
                    offset = int(row["offset"])
                    length = int(row["length"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid Graph index record at line {line_number}: {self.index_path}") from exc
                if not key or offset < 0 or length <= 0:
                    raise ValueError(f"Invalid Graph index boundary at line {line_number}: {self.index_path}")
                if key in self._records:
                    raise ValueError(f"Duplicate Graph index key {key!r}: {self.index_path}")
                self._records[key] = ClinicalGraphIndexRecord(key=key, offset=offset, length=length)
        if not self._records:
            raise ValueError(f"Graph sidecar index is empty: {self.index_path}")

    def lookup(self, key: str) -> ClinicalGraphIndexRecord:
        normalized = str(key).strip()
        try:
            return self._records[normalized]
        except KeyError as exc:
            raise KeyError(f"Graph sidecar index has no key: {normalized!r}") from exc

    def __len__(self) -> int:
        return len(self._records)


def _json_key(row: dict[str, Any], field: str, line_number: int) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"Graph sidecar line {line_number} is missing {field}.")
    return value


def materialize_streaming_index(
    *,
    source_path: str | Path,
    index_path: str | Path,
    source_sha256: str | None = None,
) -> dict[str, object]:
    """Build an index by streaming one JSONL record at a time."""
    import hashlib

    source = Path(source_path).expanduser().resolve()
    target = Path(index_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Graph sidecar source not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    record_count = 0
    duplicate_sample_id_count = 0
    duplicate_image_id_count = 0
    seen_sample: set[str] = set()
    seen_image: set[str] = set()
    boundary_ok = True
    temp = target.with_name(target.name + f".tmp.{os.getpid()}")
    try:
        with source.open("rb") as src, temp.open("w", encoding="utf-8", newline="\n") as out:
            while True:
                offset = src.tell()
                raw = src.readline()
                if not raw:
                    break
                digest.update(raw)
                length = len(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Graph sidecar line {record_count + 1} is not valid JSON.") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Graph sidecar line {record_count + 1} is not an object.")
                sample_id = _json_key(row, "sample_id", record_count + 1)
                image_id = _json_key(row, "image_id", record_count + 1)
                if sample_id in seen_sample:
                    duplicate_sample_id_count += 1
                if image_id in seen_image:
                    duplicate_image_id_count += 1
                seen_sample.add(sample_id)
                seen_image.add(image_id)
                # A single record is indexed under both stable identities.  A
                # frozen sidecar may use the same value for both fields, so do
                # not emit duplicate index keys.
                for key in dict.fromkeys((sample_id, image_id)):
                    out.write(json.dumps({"key": key, "offset": offset, "length": length}, ensure_ascii=False, separators=(",", ":")) + "\n")
                record_count += 1
                if src.tell() != offset + length:
                    boundary_ok = False
        if source_sha256 is not None and digest.hexdigest() != str(source_sha256).lower():
            raise ValueError(
                f"Graph source SHA256 mismatch: expected {source_sha256}, observed {digest.hexdigest()}"
            )
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "source_path": str(source),
        "source_sha256": digest.hexdigest(),
        "source_size": source.stat().st_size,
        "source_record_count": record_count,
        "index_path": str(target),
        "index_record_count": len(seen_sample | seen_image),
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "duplicate_image_id_count": duplicate_image_id_count,
        "offset_boundary_check": boundary_ok,
        "status": "PASS" if boundary_ok and not duplicate_sample_id_count and not duplicate_image_id_count else "BLOCKED",
    }


__all__ = ["ClinicalGraphIndexRecord", "ClinicalGraphSidecarIndex", "INDEX_SCHEMA_VERSION", "materialize_streaming_index"]
