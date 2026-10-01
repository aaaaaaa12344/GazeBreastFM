"""Materialization and read-only validation for V6.1 full-pool candidate registries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from breast_pretrain.data.stage1_fullpool_binding_schema import (
    CANDIDATE_REGISTRY_SCHEMA_VERSION,
    CASE_REGISTRY_SCHEMA_VERSION,
    VALIDATION_RECEIPT_SCHEMA_VERSION,
    build_case_record,
    canonical_json_bytes,
    immutable_record_sha256,
    materialize_candidate_record,
    sha256_bytes,
    sha256_file,
    validate_candidate_record,
    validate_case_record,
)
from breast_pretrain.data.stage1_fullpool_identity_validation import validate_candidate_identity_consistency


CANDIDATE_REGISTRY_FILENAME = "stage1_fullpool_candidate_registry.jsonl"
CASE_REGISTRY_FILENAME = "stage1_fullpool_case_registry.jsonl"
SHA256S_FILENAME = "stage1_fullpool_candidate_sha256s.txt"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object.")
            rows.append(value)
    return rows


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")


def _case_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset_id"]), str(row["case_id"]), str(row["report_unit_id"]))


def build_candidate_registry(
    source_rows: Iterable[Mapping[str, Any]], *, binding_config_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze explicit authority records into image and case registries.

    ``source_rows`` are already authority-resolved records.  This function does
    no path discovery, content generation, identity guessing, or asset repair.
    """

    candidates = [
        materialize_candidate_record(row, binding_config_sha256=binding_config_sha256)
        for row in source_rows
    ]
    candidates.sort(key=lambda row: (str(row["dataset_id"]), str(row["canonical_image_id"])))
    image_ids = [str(row["image_id"]) for row in candidates]
    canonical_image_ids = [str(row["canonical_image_id"]) for row in candidates]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_id must be globally unique in the candidate registry.")
    if len(canonical_image_ids) != len(set(canonical_image_ids)):
        raise ValueError("canonical_image_id must be globally unique in the candidate registry.")
    identity_errors = validate_candidate_identity_consistency(candidates)
    if identity_errors:
        raise ValueError("invalid candidate identity/split consistency: " + " | ".join(identity_errors))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[_case_key(row)].append(row)
    cases = [build_case_record(groups[key]) for key in sorted(groups)]
    return candidates, cases


def materialize_candidate_registry(
    source_rows: Iterable[Mapping[str, Any]], *, binding_config_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    """Write a new immutable candidate registry; existing registry files are never overwritten."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / CANDIDATE_REGISTRY_FILENAME
    case_path = root / CASE_REGISTRY_FILENAME
    checksums_path = root / SHA256S_FILENAME
    collisions = [path for path in (candidate_path, case_path, checksums_path) if path.exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite immutable registry files: " + ", ".join(map(str, collisions)))
    candidates, cases = build_candidate_registry(source_rows, binding_config_sha256=binding_config_sha256)
    _write_jsonl_new(candidate_path, candidates)
    _write_jsonl_new(case_path, cases)
    candidate_sha = sha256_file(str(candidate_path))
    case_sha = sha256_file(str(case_path))
    checksums_path.write_text(
        f"{candidate_sha}  {CANDIDATE_REGISTRY_FILENAME}\n{case_sha}  {CASE_REGISTRY_FILENAME}\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "registry_schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "case_registry_schema_version": CASE_REGISTRY_SCHEMA_VERSION,
        "candidate_registry_path": str(candidate_path),
        "case_registry_path": str(case_path),
        "candidate_registry_sha256": candidate_sha,
        "case_registry_sha256": case_sha,
        "candidate_row_count": len(candidates),
        "case_row_count": len(cases),
    }


def validate_candidate_registry_rows(
    candidate_rows: list[Mapping[str, Any]], case_rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Read-only validation.  The returned receipt never changes input rows."""

    errors: list[str] = []
    image_ids: set[str] = set()
    canonical_image_ids: set[str] = set()
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    readiness = Counter()
    exclusions = Counter()
    blocking = Counter()
    for index, row in enumerate(candidate_rows, start=1):
        for error in validate_candidate_record(row):
            errors.append(f"candidate row {index}: {error}")
        image_id = str(row.get("image_id", ""))
        if image_id in image_ids:
            errors.append(f"candidate row {index}: duplicate image_id={image_id!r}.")
        image_ids.add(image_id)
        image_id = str(row.get("canonical_image_id", ""))
        if image_id in canonical_image_ids:
            errors.append(f"candidate row {index}: duplicate canonical_image_id={image_id!r}.")
        canonical_image_ids.add(image_id)
        try:
            groups[_case_key(row)].append(row)
        except KeyError:
            errors.append(f"candidate row {index}: cannot derive case key.")
        readiness["ready" if row.get("candidate_ready") is True else "not_ready"] += 1
        exclusions.update(str(code) for code in row.get("exclusion_reason_codes", []))
        blocking.update(str(code) for code in row.get("blocking_reason_codes", []))
    errors.extend(validate_candidate_identity_consistency(candidate_rows))
    expected_cases: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for key, rows in groups.items():
        try:
            expected_cases[key] = build_case_record(rows)
        except ValueError as exc:
            errors.append(f"case {key}: {exc}")
    seen_case_keys: set[tuple[str, str, str]] = set()
    for index, row in enumerate(case_rows, start=1):
        for error in validate_case_record(row):
            errors.append(f"case row {index}: {error}")
        key = (str(row.get("dataset_id", "")), str(row.get("case_id", "")), str(row.get("report_unit_id", "")))
        if key in seen_case_keys:
            errors.append(f"case row {index}: duplicate case key={key}.")
        seen_case_keys.add(key)
        expected = expected_cases.get(key)
        if expected is None:
            errors.append(f"case row {index}: no matching candidate rows for case key={key}.")
        elif canonical_json_bytes(expected) != canonical_json_bytes(row):
            errors.append(f"case row {index}: contents do not match candidate-derived case registry row.")
    missing_case_rows = sorted(set(expected_cases) - seen_case_keys)
    if missing_case_rows:
        errors.append("case registry misses candidate-derived case keys: " + ", ".join(map(str, missing_case_rows[:20])))
    return {
        "validation_schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors[:200],
        "candidate_row_count": len(candidate_rows),
        "case_row_count": len(case_rows),
        "readiness_summary": dict(sorted(readiness.items())),
        "blocking_reason_counts": dict(sorted(blocking.items())),
        "exclusion_reason_counts": dict(sorted(exclusions.items())),
        "validation_pass_ready_row_count": readiness["ready"] if not errors else 0,
    }


def validate_candidate_registry_files(
    candidate_registry_path: str | Path, case_registry_path: str | Path
) -> dict[str, Any]:
    """Validate frozen registry files without writing or repairing them."""

    candidate_path = Path(candidate_registry_path).expanduser().resolve()
    case_path = Path(case_registry_path).expanduser().resolve()
    candidate_sha_before = sha256_file(str(candidate_path))
    case_sha_before = sha256_file(str(case_path))
    receipt = validate_candidate_registry_rows(_read_jsonl(candidate_path), _read_jsonl(case_path))
    candidate_sha_after = sha256_file(str(candidate_path))
    case_sha_after = sha256_file(str(case_path))
    receipt.update(
        {
            "candidate_registry_path": str(candidate_path),
            "case_registry_path": str(case_path),
            "candidate_registry_sha256": candidate_sha_before,
            "case_registry_sha256": case_sha_before,
            "registry_immutable_during_validation": (
                candidate_sha_before == candidate_sha_after and case_sha_before == case_sha_after
            ),
        }
    )
    receipt["validation_receipt_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "validation_receipt_sha256"})
    )
    return receipt


def load_explicit_authority_rows(config: Mapping[str, Any], *, config_path: str | Path) -> list[dict[str, Any]]:
    """Read only explicitly declared JSONL record sources; never scan a dataset root."""

    config_file = Path(config_path).expanduser().resolve()
    input_path_value = config.get("input_image_records_jsonl")
    if not isinstance(input_path_value, str) or not input_path_value.strip():
        raise ValueError("input_image_records_jsonl must be an explicit JSONL path; directory scanning is forbidden.")
    input_path = Path(input_path_value).expanduser()
    if not input_path.is_absolute():
        input_path = (config_file.parent / input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"explicit input_image_records_jsonl does not exist: {input_path}")
    authorities = config.get("dataset_authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("dataset_authorities must be a non-empty explicit list.")
    authority_by_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    required_authority_fields = ("dataset_id", "dataset_version", "authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256")
    for index, authority in enumerate(authorities):
        if not isinstance(authority, dict) or any(not str(authority.get(field, "")).strip() for field in required_authority_fields):
            raise ValueError(f"dataset_authorities[{index}] misses required authority binding fields.")
        key = (str(authority["dataset_id"]), str(authority["dataset_version"]))
        if key in authority_by_dataset:
            raise ValueError(f"duplicate declared dataset authority for {key}.")
        authority_by_dataset[key] = {field: authority[field] for field in required_authority_fields[2:]}
    rows = _read_jsonl(input_path)
    for index, row in enumerate(rows, start=1):
        key = (str(row.get("dataset_id", "")), str(row.get("dataset_version", "")))
        if key not in authority_by_dataset:
            raise ValueError(f"input row {index} has no declared versioned dataset authority for {key}.")
        if "dataset_authority" in row:
            raise ValueError("input image record must not override dataset_authority from the binding config.")
        row["dataset_authority"] = authority_by_dataset[key]
    return rows
