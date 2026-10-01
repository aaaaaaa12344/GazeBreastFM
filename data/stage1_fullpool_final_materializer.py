"""Deterministic final Stage 1 bundle materializer (Work7 hotfix 1).

Consumes only already-frozen upstream authorities:

* a validated immutable candidate registry (image + case rows and its
  read-only validation receipt),
* the final Image Branch Stage1 roster/split receipt,
* Work456 frozen semantic receipts/contracts.

It projects only hard-gate-ready rows, never recomputes E1 / Stage0 / E2 /
Effective Report / Embedding / Clinical Graph / concept / prototype /
semantic soft-label V2, and writes a new versioned runtime bundle directory.

The candidate registry is an immutable status ledger and is NOT a training
manifest.  Only this materializer may create ``train_eligible=true`` rows in a
new final runtime manifest; the DataLoader must never become an eligibility
authority by filtering the registry at runtime.

Production materialization is fail-closed: without a final Image Branch roster
receipt whose terminal state is frozen/formal, no final bundle is written.
Tiny synthetic fixtures are allowed for tests only.

Hotfix 1 contract alignment: the materializer writes the runtime manifest under
the exact V6 standard filename ``manifest_stage1_v6.csv`` that the formal
launcher consumes (``formal_v6_entry_gate.assert_v6_formal_entry``), with the
V6 required columns plus finalization columns.  There is exactly one runtime
manifest per bundle; no split-brain between a final manifest and a launcher
manifest is possible.  The final roster receipt is self-hash-verified and its
rows carry candidate identity/split authority; the bundle SHA manifest is
recomputed per file during validation.

Hotfix 2 control-plane closure:

* the final bundle root assembles the complete ``V6_STANDARD_FILENAMES``
  runtime bundle by reference: every V6 asset except the projected manifest is
  hardlinked (same inode, zero copy) from the frozen upstream V6 bundle root
  and recorded (filename / source path / real SHA256) in a deterministic V6
  asset index.  Nothing is recomputed or copied; hardlinks keep
  ``Path.resolve()`` inside the bundle root so the existing formal config
  validator's under-root checks stay valid;
* the candidate validation receipt is exactly bound: its self-hash is
  verified and its frozen ``candidate_registry_sha256`` / ``case_registry_sha256``
  must equal the real SHA256 of the current registry files (structural
  re-validation alone is not a binding);
* final exclusion ids (``MANUAL_REJECTED_FINAL`` case ids and discarded /
  review / reject / no-gaze / gaze-disabled image ids) are collected
  unconditionally -- never pre-filtered against the projected manifest.  The
  final validator performs the explicit set intersection and fails on any
  overlap;
* the bundle SHA closure covers the V6 asset index (which carries the V6 asset
  references/hashes consumed by the runtime and Entry Contract) together with
  manifest / mapping / inventory, avoiding any circular hash.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from breast_pretrain.data.stage1_fullpool_binding_schema import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from breast_pretrain.data.stage1_fullpool_candidate_registry import (
    validate_candidate_registry_rows,
)
from breast_pretrain.data.stage1_v6_contract import (
    V6_STANDARD_FILENAMES,
    _REQUIRED_COLUMNS as _V6_REQUIRED_COLUMNS,
)

FINAL_MANIFEST_FILENAME = "manifest_stage1_v6.csv"
ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME = "final_row_image_case_report_mapping.jsonl"
BUNDLE_INVENTORY_FILENAME = "stage1_final_bundle_inventory.json"
BUNDLE_SHA_MANIFEST_FILENAME = "stage1_final_bundle_sha256s.txt"
MATERIALIZATION_RECEIPT_FILENAME = "stage1_final_bundle_receipt.json"
V6_ASSET_INDEX_FILENAME = "stage1_final_v6_asset_index.json"

V6_ASSET_INDEX_SCHEMA_VERSION = "stage1_final_v6_asset_index_v1"

# V6 assets that may legitimately be empty in a frozen bundle (mirrors the
# allow-empty set of ``validate_stage1_v6_bundle``).
_V6_ALLOW_EMPTY = frozenset({"graph_edges", "runtime_graph_edges", "final_rejected", "discarded_images"})

FINAL_BUNDLE_SCHEMA_VERSION = "stage1_final_bundle_v1"
MATERIALIZER_VERSION = "work7_final_materializer_v1"
ROSTER_TERMINAL_FORMAL_STATES = frozenset({"FINAL_FROZEN", "FORMAL_FROZEN", "FINAL_ROSTER_FROZEN"})

# Entry Contract V6.1 finalization columns on top of the V6 runtime columns.
_FINAL_EXTRA_COLUMNS = (
    "canonical_image_id",
    "study_id",
    "laterality",
    "view",
    "concept_schema_version",
    "gaze_prior_key",
    "gaze_to_patch_projection_key",
    "gaze_prior_sha256",
    "gaze_to_patch_projection_sha256",
    "patch_geometry_sha256",
    "valid_content_mask_sha256",
    "text_source_type",
)
FINAL_MANIFEST_COLUMNS = tuple(sorted(_V6_REQUIRED_COLUMNS)) + _FINAL_EXTRA_COLUMNS


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON authority must be an object: {path}")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool_flag(value: Any) -> bool:
    return value is True or _text(value).lower() in {"1", "true", "yes"}


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


@dataclass(frozen=True)
class SemanticReceiptRefs:
    concept_schema_path: str | None = None
    concept_schema_sha256: str | None = None
    concept_target_receipt_path: str | None = None
    concept_target_receipt_sha256: str | None = None
    prototype_receipt_path: str | None = None
    prototype_receipt_sha256: str | None = None
    soft_label_v2_receipt_path: str | None = None
    soft_label_v2_receipt_sha256: str | None = None
    semantic_contract_path: str | None = None
    semantic_contract_sha256: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "concept_schema_path": self.concept_schema_path,
            "concept_schema_sha256": self.concept_schema_sha256,
            "concept_target_receipt_path": self.concept_target_receipt_path,
            "concept_target_receipt_sha256": self.concept_target_receipt_sha256,
            "prototype_receipt_path": self.prototype_receipt_path,
            "prototype_receipt_sha256": self.prototype_receipt_sha256,
            "soft_label_v2_receipt_path": self.soft_label_v2_receipt_path,
            "soft_label_v2_receipt_sha256": self.soft_label_v2_receipt_sha256,
            "semantic_contract_path": self.semantic_contract_path,
            "semantic_contract_sha256": self.semantic_contract_sha256,
        }


def parse_semantic_receipt_refs(config: Mapping[str, Any], config_path: Path) -> SemanticReceiptRefs:
    """Read explicit Work456 receipt references; missing semantic receipts stay None.

    Semantic asset receipts may legitimately be absent only while the upstream
    authority has not materialized them; the materializer still records the
    declared state and the final validator decides hard-fail policy.
    """

    def _resolve(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path(config_path).parent / path).resolve()
        return str(path)

    semantic = config.get("semantic_receipts")
    semantic = semantic if isinstance(semantic, Mapping) else {}
    return SemanticReceiptRefs(
        concept_schema_path=_resolve(semantic.get("concept_schema_path")),
        concept_schema_sha256=_text(semantic.get("concept_schema_sha256")) or None,
        concept_target_receipt_path=_resolve(semantic.get("concept_target_receipt_path")),
        concept_target_receipt_sha256=_text(semantic.get("concept_target_receipt_sha256")) or None,
        prototype_receipt_path=_resolve(semantic.get("prototype_receipt_path")),
        prototype_receipt_sha256=_text(semantic.get("prototype_receipt_sha256")) or None,
        soft_label_v2_receipt_path=_resolve(semantic.get("soft_label_v2_receipt_path")),
        soft_label_v2_receipt_sha256=_text(semantic.get("soft_label_v2_receipt_sha256")) or None,
        semantic_contract_path=_resolve(semantic.get("semantic_contract_path")),
        semantic_contract_sha256=_text(semantic.get("semantic_contract_sha256")) or None,
    )


def _verify_candidate_validation_receipt_bind(
    validation_receipt: dict[str, Any],
    receipt_path: Path,
    candidate_path: Path,
    case_path: Path,
) -> None:
    """Exact-bind the candidate validation receipt to the current registry files.

    The receipt must carry its own self-hash and freeze the SHA256 of the
    candidate and case registries it validated; both must equal the real
    SHA256 of the files handed to the materializer right now.  Structural
    re-validation may run in addition, but never instead of this binding.
    """
    declared_self = _text(validation_receipt.get("validation_receipt_sha256"))
    recomputed_self = sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in validation_receipt.items() if key != "validation_receipt_sha256"}
        )
    )
    if not declared_self or declared_self != recomputed_self:
        raise ValueError(
            f"candidate validation receipt self-hash mismatch; receipt is not immutable: {receipt_path}"
        )
    for field, actual_path in (
        ("candidate_registry_sha256", candidate_path),
        ("case_registry_sha256", case_path),
    ):
        frozen = _text(validation_receipt.get(field))
        actual = sha256_file(str(actual_path))
        if not frozen or frozen != actual:
            raise ValueError(
                f"candidate validation receipt does not exactly bind the current registry files: "
                f"{field} frozen={frozen!r}, actual file={actual} ({actual_path})"
            )


def _verify_roster_receipt_self_hash(receipt: dict[str, Any], path: Path) -> None:
    """Verify the roster receipt's own SHA256 against its canonical content."""
    declared = _text(receipt.get("receipt_sha256") or receipt.get("roster_receipt_sha256"))
    if not declared:
        raise ValueError("final Image Branch roster receipt must carry its own sha256.")
    recomputed = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key not in {"receipt_sha256", "roster_receipt_sha256"}})
    )
    if declared != recomputed:
        raise ValueError(
            "final Image Branch roster receipt self-hash mismatch; receipt is not immutable: "
            f"{path} (declared={declared}, recomputed={recomputed})"
        )


def _read_roster_receipt(path: Path) -> dict[str, Any]:
    _require_file(path, "final Image Branch roster receipt")
    receipt = _read_json(path)
    _verify_roster_receipt_self_hash(receipt, path)
    terminal = _text(receipt.get("terminal_status") or receipt.get("status"))
    if terminal not in ROSTER_TERMINAL_FORMAL_STATES:
        raise ValueError(
            "final Image Branch roster is not formal/frozen; production materialization "
            f"must fail closed (terminal_status={terminal!r})."
        )
    if "split" not in receipt and "patient_split" not in receipt:
        raise ValueError("final Image Branch roster receipt must declare the frozen split policy.")
    return receipt


def _roster_membership_keys(receipt: dict[str, Any]) -> tuple[str, ...] | None:
    """Return the identity fields used to match roster membership rows."""
    declared = receipt.get("membership_fields")
    if isinstance(declared, list) and declared and all(isinstance(item, str) and item for item in declared):
        return tuple(str(item) for item in declared)
    for candidate in ("image_id", "canonical_image_id", "case_id"):
        if _text(receipt.get(candidate)):
            return (candidate,)
    return None


_ROSTER_IDENTITY_FIELDS = ("patient_id", "patient_split", "case_id")


def _load_roster_rows(receipt: dict[str, Any], receipt_path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    """Explicit roster membership rows; the receipt may embed them or name a manifest.

    Every roster row must carry candidate identity authority (patient_id /
    patient_split / case_id) in addition to the membership key: image
    membership alone is not enough for the final roster to be authoritative.
    """
    raw_rows = receipt.get("rows") or receipt.get("membership_rows")
    roster_path_value = receipt.get("roster_path") or receipt.get("membership_manifest_path")
    if not isinstance(raw_rows, list):
        if not isinstance(roster_path_value, str) or not roster_path_value:
            raise ValueError("final Image Branch roster receipt must embed rows or name an explicit membership manifest.")
        roster_path = Path(roster_path_value).expanduser()
        if not roster_path.is_absolute():
            roster_path = Path(receipt_path).parent / roster_path
        _require_file(roster_path, "final Image Branch roster membership manifest")
        raw_rows = _read_jsonl(roster_path)
    keys = _roster_membership_keys(receipt)
    if keys is None:
        raise ValueError("final Image Branch roster receipt must declare membership_fields or a membership identity key.")
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"roster membership row {index} must be an object.")
        key = tuple(_text(row.get(field)) for field in keys)
        if not all(key):
            raise ValueError(f"roster membership row {index} misses membership fields {keys}.")
        if key in by_key:
            raise ValueError(f"roster membership row {index} duplicates membership key {key!r}.")
        missing_identity = [field for field in _ROSTER_IDENTITY_FIELDS if not _text(row.get(field))]
        if missing_identity:
            raise ValueError(
                f"roster membership row {index} misses candidate identity authority fields: "
                + ", ".join(missing_identity)
            )
        by_key[key] = dict(row)
    return by_key


def _candidate_row_key(row: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_text(row.get(field)) for field in keys)


def _assert_roster_identity_authority(candidate_row: Mapping[str, Any], roster_row: Mapping[str, Any], image_id: str) -> None:
    """Candidate identity/split must exactly match the final roster row authority."""
    for field in _ROSTER_IDENTITY_FIELDS:
        candidate_value = _text(candidate_row.get(field))
        roster_value = _text(roster_row.get(field))
        if candidate_value != roster_value:
            raise ValueError(
                f"candidate image_id={image_id!r} identity field {field} ({candidate_value!r}) "
                f"does not match final roster row authority ({roster_value!r})."
            )


def _asset_text(asset: Mapping[str, Any] | None, field: str, default: str = "") -> str:
    if asset is None:
        return default
    return _text(asset.get(field)) or default


def project_final_manifest_row(candidate_row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one ready candidate row into the final runtime manifest row.

    Only fields already frozen in the candidate registry are referenced; no
    identity, eligibility, or semantic field is recomputed here.  The output
    covers the Stage1 Entry Contract V6.1 identity/image/patch/gaze/text/
    embedding/graph/concept field set.
    """
    report = candidate_row.get("effective_report")
    report = report if isinstance(report, Mapping) else {}
    stage0 = candidate_row.get("stage0")
    stage0 = stage0 if isinstance(stage0, Mapping) else {}
    gaze = candidate_row.get("gaze")
    gaze = gaze if isinstance(gaze, Mapping) else {}
    patch = candidate_row.get("patch")
    patch = patch if isinstance(patch, Mapping) else {}
    embedding = candidate_row.get("embedding")
    embedding = embedding if isinstance(embedding, Mapping) else {}
    graph = candidate_row.get("clinical_graph")
    graph = graph if isinstance(graph, Mapping) else {}
    concept = candidate_row.get("concept")
    concept = concept if isinstance(concept, Mapping) else {}
    report_hash = _text(report.get("effective_report_sha256"))
    row: dict[str, Any] = {
        "image_id": _text(candidate_row.get("image_id")),
        "image_path": _text(candidate_row.get("image_path")),
        "modality": _text(candidate_row.get("modality")),
        "patient_id": _text(candidate_row.get("patient_id")),
        "case_id": _text(candidate_row.get("case_id")),
        "report_unit_id": _text(candidate_row.get("report_unit_id")),
        "split": _text(candidate_row.get("patient_split")),
        "dataset_id": _text(candidate_row.get("dataset_id")),
        "source_image_sha256": _text(candidate_row.get("source_image_sha256")),
        "final_case_route": _text(report.get("final_report_route")),
        "train_eligible": "1",
        "visual_training_enabled": "1",
        "gaze_training_enabled": "1",
        "text_semantic_enabled": "1",
        "clinical_graph_enabled": "1",
        "concept_target_enabled": "1",
        "semantic_soft_label_enabled": "1",
        "report_id": _text(candidate_row.get("report_id")),
        "effective_report_sha256": report_hash,
        "text_prompt_key": _asset_text(report, "artifact_path_or_key"),
        "embedding_key": _asset_text(embedding, "embedding_key", _asset_text(embedding, "artifact_path_or_key")),
        "graph_key": _asset_text(graph, "graph_sidecar_path", _asset_text(graph, "artifact_path_or_key")),
        "patch_geometry_key": _asset_text(patch, "patch_geometry_key"),
        "patch_token_order_version": _asset_text(patch, "patch_token_order_version"),
        "valid_content_mask_key": _asset_text(patch, "valid_content_mask_key"),
        "patch_asset_mode": _asset_text(patch, "patch_asset_mode", "materialized"),
        "gaze_prior_available": "1",
        "gaze_prior_quality": _asset_text(gaze, "gaze_prior_quality", "usable"),
        # Finalization columns (Entry Contract V6.1 checks).
        "canonical_image_id": _text(candidate_row.get("canonical_image_id")),
        "study_id": _text(candidate_row.get("study_id")),
        "laterality": _text(candidate_row.get("laterality")),
        "view": _text(candidate_row.get("view")),
        "concept_schema_version": _asset_text(concept, "concept_schema_version"),
        "gaze_prior_key": _asset_text(gaze, "gaze_prior_key", _asset_text(gaze, "artifact_path_or_key")),
        "gaze_to_patch_projection_key": _asset_text(gaze, "gaze_to_patch_projection_key"),
        "gaze_prior_sha256": _asset_text(gaze, "gaze_prior_sha256"),
        "gaze_to_patch_projection_sha256": _asset_text(gaze, "gaze_to_patch_projection_sha256"),
        "patch_geometry_sha256": _asset_text(patch, "patch_geometry_sha256"),
        "valid_content_mask_sha256": _asset_text(patch, "valid_content_mask_sha256"),
        "stage0_usable": "1" if stage0.get("usable") is True else "0",
        "text_source_type": _text(report.get("text_source_type")),
    }
    return row


def _validate_manifest_row(row: Mapping[str, Any], index: int, errors: list[str]) -> None:
    if len(errors) >= 200:
        return
    for field in FINAL_MANIFEST_COLUMNS:
        if not _text(row.get(field)):
            errors.append(f"final manifest row {index} misses required column {field}.")
    if _bool_flag(row.get("train_eligible")) and row.get("split") not in {None, "", "train"}:
        errors.append(f"final manifest row {index} has train_eligible=true outside split=train.")
    if _text(row.get("modality")).lower() in {"mammo", "mammography", "ffdm", "dbt", "cesm"}:
        if not _text(row.get("study_id")) or not _text(row.get("laterality")):
            errors.append(f"final manifest row {index}: mammography requires study_id and laterality.")


def materialize_final_bundle(
    *,
    candidate_registry_path: str | Path,
    case_registry_path: str | Path,
    candidate_validation_receipt_path: str | Path,
    final_roster_receipt_path: str | Path,
    v6_bundle_root: str | Path,
    output_dir: str | Path,
    semantic_receipts: SemanticReceiptRefs | None = None,
    allow_synthetic_roster: bool = False,
) -> dict[str, Any]:
    """Write a new versioned final Stage 1 bundle from frozen authorities only.

    ``v6_bundle_root`` is the frozen upstream V6 runtime bundle.  Every V6
    standard asset except the projected manifest is hardlinked (same inode,
    zero copy) from it into the final root and recorded in the V6 asset index;
    nothing is recomputed or copied.  ``allow_synthetic_roster`` is a test-only
    escape hatch for tiny synthetic fixtures; production callers must leave it
    False.  The roster receipt still has to declare a formal/frozen terminal
    state either way.
    """
    candidate_path = Path(candidate_registry_path).expanduser().resolve()
    case_path = Path(case_registry_path).expanduser().resolve()
    validation_receipt_path = Path(candidate_validation_receipt_path).expanduser().resolve()
    roster_receipt_path = Path(final_roster_receipt_path).expanduser().resolve()
    v6_root = Path(v6_bundle_root).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()

    _require_file(candidate_path, "candidate registry")
    _require_file(case_path, "case registry")
    _require_file(validation_receipt_path, "candidate validation receipt")
    _require_file(roster_receipt_path, "final Image Branch roster receipt")
    _require_file(v6_root / FINAL_MANIFEST_FILENAME, "frozen V6 bundle manifest")

    if root == v6_root:
        raise ValueError("the final bundle root must be a new directory, never the frozen V6 bundle root.")

    validation_receipt = _read_json(validation_receipt_path)
    if _text(validation_receipt.get("status")) != "PASS":
        raise ValueError(
            "candidate validation receipt is not PASS; validated candidate registry required "
            "before final bundle materialization."
        )
    # Hotfix 2: the receipt must exactly bind the registry files being consumed.
    _verify_candidate_validation_receipt_bind(
        validation_receipt, validation_receipt_path, candidate_path, case_path
    )

    roster_receipt = _read_roster_receipt(roster_receipt_path)
    if not allow_synthetic_roster and _bool_flag(roster_receipt.get("synthetic_test_fixture", False)):
        raise ValueError("synthetic roster receipts are only allowed in tests.")

    candidate_rows = _read_jsonl(candidate_path)
    case_rows = _read_jsonl(case_path)
    # Read-only re-validation guards against a registry mutated after its receipt.
    revalidation = validate_candidate_registry_rows(candidate_rows, case_rows)
    if revalidation["status"] != "PASS":
        raise ValueError(
            "candidate registry re-validation failed before materialization: "
            + " | ".join(revalidation["errors"][:10])
        )

    ready_rows = [row for row in candidate_rows if row.get("candidate_ready") is True]
    if not ready_rows:
        raise ValueError("no candidate_ready=true rows in the validated candidate registry.")

    membership_keys = _roster_membership_keys(roster_receipt)
    roster_rows = _load_roster_rows(roster_receipt, roster_receipt_path)
    if membership_keys is None:
        raise ValueError("final Image Branch roster receipt must declare membership identity fields.")

    projected: list[dict[str, Any]] = []
    excluded_by_roster: list[str] = []
    for row in ready_rows:
        key = _candidate_row_key(row, membership_keys)
        if key not in roster_rows:
            excluded_by_roster.append(_text(row.get("image_id")))
            continue
        _assert_roster_identity_authority(row, roster_rows[key], _text(row.get("image_id")))
        projected.append(project_final_manifest_row(row))

    if not projected:
        raise ValueError(
            "zero ready candidate rows intersect the final Image Branch roster; "
            "refusing to write an empty formal bundle."
        )

    errors: list[str] = []
    for index, row in enumerate(projected, start=1):
        _validate_manifest_row(row, index, errors)
    if errors:
        raise ValueError("final manifest projection failed:\n  " + "\n  ".join(errors))

    projected.sort(
        key=lambda row: (
            _text(row["dataset_id"]),
            _text(row["split"]),
            _text(row["canonical_image_id"]),
        )
    )
    image_ids = [row["image_id"] for row in projected]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("final manifest contains duplicate image_id values.")
    canonical_image_ids = [row["canonical_image_id"] for row in projected]
    if len(canonical_image_ids) != len(set(canonical_image_ids)):
        raise ValueError("final manifest contains duplicate canonical_image_id values.")

    manifest_image_ids = set(image_ids)
    final_exclusion: dict[str, list[str]] = {
        "final_rejected_case_ids": [],
        "discarded_image_ids": [],
        "stage0_review_ids": [],
        "stage0_reject_ids": [],
        "no_gaze_ids": [],
        "gaze_disabled_ids": [],
    }
    for row in candidate_rows:
        codes = row.get("exclusion_reason_codes")
        if not isinstance(codes, list):
            continue
        image_id = _text(row.get("image_id"))
        case_id = _text(row.get("case_id"))
        for code in codes:
            if str(code) == "MANUAL_REJECTED_FINAL":
                # Exclusion proof for final rejection is case-scoped.  Collect
                # UNCONDITIONALLY: if a rejected case polluted the manifest the
                # validator must see the overlap and fail.
                if case_id:
                    final_exclusion["final_rejected_case_ids"].append(case_id)
            else:
                key = {
                    "DISCARDED_IMAGE": "discarded_image_ids",
                    "STAGE0_REVIEW": "stage0_review_ids",
                    "STAGE0_REJECT": "stage0_reject_ids",
                    "STAGE0_NO_GAZE": "no_gaze_ids",
                    "STAGE0_GAZE_DISABLED": "gaze_disabled_ids",
                }.get(str(code))
                # Image-scoped ids are collected unconditionally as well; the
                # validator intersects them with the manifest image set.
                if key is not None and image_id:
                    final_exclusion[key].append(image_id)
    for key in final_exclusion:
        final_exclusion[key] = sorted(set(final_exclusion[key]))

    # ------------------------------------------------------------------
    # V6 runtime bundle assembly by reference (Hotfix 2).
    # ------------------------------------------------------------------
    v6_manifest_rows = _read_v6_manifest_rows(v6_root)
    v6_image_ids = {_text(row.get("image_id")) for row in v6_manifest_rows}
    if v6_image_ids != manifest_image_ids:
        raise ValueError(
            "the final projected manifest image set must exactly match the frozen V6 bundle "
            f"manifest (final={len(manifest_image_ids)}, v6={len(v6_image_ids)}); "
            "the final bundle is only a deterministic projection of the frozen V6 bundle."
        )
    v6_asset_index: dict[str, Any] = {
        "index_schema_version": V6_ASSET_INDEX_SCHEMA_VERSION,
        "v6_bundle_root": str(v6_root),
        "assets": {},
        "referenced_assets": [],
    }
    for key, filename in V6_STANDARD_FILENAMES.items():
        if key == "manifest":
            continue  # the final root carries the projected manifest, not a link
        source = v6_root / filename
        if not source.is_file() or (source.stat().st_size == 0 and key not in _V6_ALLOW_EMPTY):
            raise FileNotFoundError(f"frozen V6 bundle lacks required asset {key}: {source}")
        v6_asset_index["assets"][key] = {
            "filename": filename,
            "source_path": str(source),
            "sha256": sha256_file(str(source)),
        }
    if len(v6_asset_index["assets"]) != len(V6_STANDARD_FILENAMES) - 1:
        raise ValueError("V6 asset index must cover every standard V6 asset except the projected manifest.")
    # V6 assets referenced by relative path inside the bundle (BI-RADS prior
    # files) must also be resolvable from the final root; record + symlink them.
    birads_manifest_path = v6_root / V6_STANDARD_FILENAMES["birads_priors"]
    _require_file(birads_manifest_path, "frozen V6 BI-RADS prior manifest")
    with birads_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = _text(row.get("prior_path"))
            if not relative:
                continue
            source = (v6_root / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"frozen V6 bundle BI-RADS prior file is missing: {source}")
            v6_asset_index["referenced_assets"].append(
                {"filename": relative, "source_path": str(source), "sha256": sha256_file(str(source))}
            )
    v6_asset_index["referenced_assets"].sort(key=lambda item: item["filename"])

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / FINAL_MANIFEST_FILENAME
    mapping_path = root / ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME
    inventory_path = root / BUNDLE_INVENTORY_FILENAME
    sha_manifest_path = root / BUNDLE_SHA_MANIFEST_FILENAME
    receipt_path = root / MATERIALIZATION_RECEIPT_FILENAME
    index_path = root / V6_ASSET_INDEX_FILENAME
    linked_paths = [root / filename for filename in V6_STANDARD_FILENAMES.values() if filename != FINAL_MANIFEST_FILENAME]
    linked_paths.extend(root / item["filename"] for item in v6_asset_index["referenced_assets"])
    collisions = [
        p for p in (manifest_path, mapping_path, inventory_path, sha_manifest_path, receipt_path, index_path, *linked_paths)
        if p.exists()
    ]
    if collisions:
        raise FileExistsError("refusing to overwrite an existing final bundle: " + ", ".join(map(str, collisions)))
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in projected:
            writer.writerow(row)

    for key, asset in v6_asset_index["assets"].items():
        target = root / asset["filename"]
        _hardlink(asset["source_path"], target, f"V6 asset {key}")
    for referenced in v6_asset_index["referenced_assets"]:
        target = root / referenced["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _hardlink(referenced["source_path"], target, f"referenced V6 asset {referenced['filename']}")
    index_path.write_text(
        json.dumps(v6_asset_index, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    row_mappings: list[dict[str, Any]] = []
    for row_index, row in enumerate(projected):
        row_mappings.append(
            {
                "manifest_row_index": row_index,
                "image_id": row["image_id"],
                "canonical_image_id": row["canonical_image_id"],
                "case_id": row["case_id"],
                "report_unit_id": row["report_unit_id"],
                "report_id": row["report_id"],
                "effective_report_sha256": row["effective_report_sha256"],
            }
        )
    mapping_payload = canonical_json_bytes(row_mappings)
    mapping_path.write_bytes(mapping_payload + b"\n")

    inventory: dict[str, Any] = {
        "bundle_schema_version": FINAL_BUNDLE_SCHEMA_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "input_receipts": {
            "candidate_registry_path": str(candidate_path),
            "candidate_registry_sha256": sha256_file(str(candidate_path)),
            "case_registry_path": str(case_path),
            "case_registry_sha256": sha256_file(str(case_path)),
            "candidate_validation_receipt_path": str(validation_receipt_path),
            "candidate_validation_receipt_sha256": sha256_file(str(validation_receipt_path)),
            "final_roster_receipt_path": str(roster_receipt_path),
            "final_roster_receipt_sha256": sha256_file(str(roster_receipt_path)),
            "final_roster_terminal_status": _text(roster_receipt.get("terminal_status") or roster_receipt.get("status")),
        },
        "semantic_receipts": (semantic_receipts.to_record() if semantic_receipts is not None else {}),
        "v6_bundle_root": str(v6_root),
        "v6_asset_index": {
            "filename": V6_ASSET_INDEX_FILENAME,
            "asset_count": len(v6_asset_index["assets"]),
        },
        "counts": {
            "candidate_rows": len(candidate_rows),
            "candidate_ready_rows": len(ready_rows),
            "roster_rows": len(roster_rows),
            "projected_manifest_rows": len(projected),
            "ready_rows_excluded_by_roster": len(excluded_by_roster),
            "case_rows": len(case_rows),
            "dataset_count": len({_text(row["dataset_id"]) for row in projected}),
            "case_count": len({_text(row["case_id"]) for row in projected}),
            "image_count": len(projected),
        },
        "deterministic_ordering": "dataset_id, split, canonical_image_id",
        "row_mapping": {"filename": ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME, "row_count": len(row_mappings)},
        "excluded_ready_image_ids": excluded_by_roster,
        "final_exclusion": final_exclusion,
    }
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    # Bundle SHA manifest covers exactly the immutable bundle-local data files
    # (manifest / mapping / inventory / V6 asset index).  It never includes
    # itself or the materialization receipt (the receipt closes over the SHA
    # manifest via its own self-hash), so there is no circular dependency.
    sha_manifest_lines = [
        f"{sha256_file(str(manifest_path))}  {manifest_path.name}",
        f"{sha256_file(str(mapping_path))}  {mapping_path.name}",
        f"{sha256_file(str(inventory_path))}  {inventory_path.name}",
        f"{sha256_file(str(index_path))}  {index_path.name}",
    ]
    sha_manifest_path.write_text("\n".join(sha_manifest_lines) + "\n", encoding="utf-8")

    receipt = copy.deepcopy(inventory)
    receipt.update(
        {
            "output_bundle_root": str(root),
            "manifest_sha256": sha256_file(str(manifest_path)),
            "mapping_sha256": sha256_file(str(mapping_path)),
            "inventory_sha256": sha256_file(str(inventory_path)),
            "sha_manifest_sha256": sha256_file(str(sha_manifest_path)),
            "v6_asset_index_sha256": sha256_file(str(index_path)),
            "materialization_receipt_sha256": "",
        }
    )
    # The bundle SHA closure covers the V6 asset index (the carrier of the V6
    # asset references/hashes consumed by the runtime and Entry Contract)
    # together with manifest / mapping / inventory.
    receipt["bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "manifest": receipt["manifest_sha256"],
                "mapping": receipt["mapping_sha256"],
                "inventory": receipt["inventory_sha256"],
                "v6_asset_index": receipt["v6_asset_index_sha256"],
            }
        )
    )
    receipt["materialization_receipt_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "materialization_receipt_sha256"})
    )
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def _read_v6_manifest_rows(v6_root: Path) -> list[dict[str, str]]:
    """Read the frozen upstream V6 manifest (identity authority for the final set)."""
    with (v6_root / FINAL_MANIFEST_FILENAME).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _hardlink(source: str, target: Path, label: str) -> None:
    """Reference a frozen asset in-place: hardlink (same inode), never a copy.

    Hardlinks keep ``Path.resolve()`` inside the final bundle root (symlinks
    would escape it and break the existing formal config under-root checks)
    while sharing storage with the frozen V6 bundle.
    """
    try:
        os.link(source, target)
    except OSError as exc:
        raise OSError(
            f"final bundle could not hardlink {label} from {source} to {target}: {exc} "
            "(the final bundle root must reside on the same filesystem as the frozen V6 bundle root)"
        ) from exc


def load_final_bundle_receipt(bundle_root: str | Path) -> dict[str, Any]:
    """Read the materialization receipt of an existing final bundle."""
    root = Path(bundle_root).expanduser().resolve()
    path = root / MATERIALIZATION_RECEIPT_FILENAME
    _require_file(path, "final bundle materialization receipt")
    return _read_json(path)


__all__ = [
    "BUNDLE_INVENTORY_FILENAME",
    "BUNDLE_SHA_MANIFEST_FILENAME",
    "FINAL_BUNDLE_SCHEMA_VERSION",
    "FINAL_MANIFEST_COLUMNS",
    "FINAL_MANIFEST_FILENAME",
    "MATERIALIZATION_RECEIPT_FILENAME",
    "MATERIALIZER_VERSION",
    "ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME",
    "V6_ASSET_INDEX_FILENAME",
    "V6_ASSET_INDEX_SCHEMA_VERSION",
    "SemanticReceiptRefs",
    "load_final_bundle_receipt",
    "materialize_final_bundle",
    "parse_semantic_receipt_refs",
    "project_final_manifest_row",
]
