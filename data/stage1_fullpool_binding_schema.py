"""Immutable, fail-closed schema for the V6.1 Stage 1 full-pool candidate registry.

This module deliberately models candidate accounting, not a runtime manifest.
In particular, ``train_eligible`` is forbidden here: only a later final-bundle
materializer may create that runtime-only field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any


CANDIDATE_REGISTRY_SCHEMA_VERSION = "stage1_fullpool_candidate_registry_v1"
CASE_REGISTRY_SCHEMA_VERSION = "stage1_fullpool_case_registry_v1"
VALIDATION_RECEIPT_SCHEMA_VERSION = "stage1_fullpool_candidate_validation_v1"

class AssetFamilyStatus(str, Enum):
    READY = "READY"
    PENDING = "PENDING"
    BLOCKED = "BLOCKED"
    EXCLUDED = "EXCLUDED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    EXCLUDED = "EXCLUDED"


class BlockingReasonCode(str, Enum):
    PENDING_FINAL_CLOSURE = "PENDING_FINAL_CLOSURE"
    PENDING_EFFECTIVE_REPORT = "PENDING_EFFECTIVE_REPORT"
    PENDING_EMBEDDING = "PENDING_EMBEDDING"
    PENDING_GRAPH = "PENDING_GRAPH"
    PENDING_CONCEPTS = "PENDING_CONCEPTS"
    PENDING_SOFT_LABEL = "PENDING_SOFT_LABEL"
    BLOCKED_INVALID_AUTHORITY = "BLOCKED_INVALID_AUTHORITY"
    BLOCKED_HASH_LINEAGE = "BLOCKED_HASH_LINEAGE"
    BLOCKED_INVALID_CONCEPT_CONTRACT = "BLOCKED_INVALID_CONCEPT_CONTRACT"


class ExclusionReasonCode(str, Enum):
    STAGE0_REVIEW = "STAGE0_REVIEW"
    STAGE0_REJECT = "STAGE0_REJECT"
    STAGE0_NO_GAZE = "STAGE0_NO_GAZE"
    STAGE0_GAZE_DISABLED = "STAGE0_GAZE_DISABLED"
    MANUAL_REJECTED_FINAL = "MANUAL_REJECTED_FINAL"
    DISCARDED_IMAGE = "DISCARDED_IMAGE"
    IDENTITY_HARD_FAILURE = "IDENTITY_HARD_FAILURE"
    SPLIT_HARD_FAILURE = "SPLIT_HARD_FAILURE"
    DUPLICATE_HARD_EXCLUSION = "DUPLICATE_HARD_EXCLUSION"


ASSET_STATUSES = frozenset(item.value for item in AssetFamilyStatus)
AUDIT_STATUSES = frozenset(item.value for item in AuditStatus)
TEXT_SOURCE_TYPES = frozenset({"real", "generated"})
TARGET_STATUSES = frozenset({"positive", "explicit_negative", "missing", "not_applicable", "uncertain"})
MODALITY_SCOPES = frozenset({"shared", "mammography", "ultrasound", "MRI"})
SUPERVISION_MODE_KEYS = frozenset({"direct", "prototype", "consistency"})

BLOCKING_REASON_CODES = frozenset(item.value for item in BlockingReasonCode)
EXCLUSION_REASON_CODES = frozenset(item.value for item in ExclusionReasonCode)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_IDENTITY_FIELDS = (
    "dataset_id",
    "dataset_version",
    "modality",
    "image_id",
    "canonical_image_id",
    "patient_id",
    "patient_split",
    "case_id",
    "report_unit_id",
    "report_id",
    "source_image_sha256",
    "canonical_image_sha256",
    "image_membership_sha256",
    "binding_config_sha256",
)
_REQUIRED_ASSETS = (
    "stage0",
    "gaze",
    "patch",
    "effective_report",
    "embedding",
    "clinical_graph",
    "concept",
    "semantic_soft_label",
)
_REQUIRED_AUDITS = (
    "identity",
    "patient_split",
    "duplicate",
    "final_reject",
    "discarded_image",
    "hash_lineage",
)
REQUIRED_FIELD_DEFINITIONS = {
    "identity": _REQUIRED_IDENTITY_FIELDS,
    "dataset_authority": ("authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256"),
    "asset_binding_when_ready": ("status", "authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256"),
    "audit_binding_when_pass": ("status", "authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256"),
    "P0-B_concept_schema": ("concept_schema_version", "concept_id", "modality_scope", "target_type", "value_space", "supervision_modes", "source_policy", "confidence_policy"),
    "P0-B_case_target": ("concept_id", "target_value", "target_status", "modality_applicable", "target_available", "valid_target_mask", "source_type", "confidence", "effective_report_sha256", "case_report_hash_lineage"),
    "derived_registry_fields": ("candidate_ready", "hard_gate_status", "hard_gate_failures", "candidate_record_sha256"),
}
_ACCEPTED_REPORT_ROUTES = frozenset(
    {
        "AUTOMATIC_ACCEPTED",
        "MANUAL_ACCEPTED",
        "ACCEPTED_REAL_REPORT",
        "ACCEPTED_GENERATED_ORIGINAL",
        "ACCEPTED_GENERATED_AFTER_REPAIR",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize values deterministically for immutable hashes and JSONL records."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def normalize_reason_codes(value: Any, allowed_codes: frozenset[str], field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of reason-code strings.")
    codes = []
    for item in value:
        code = _text(item)
        if not code:
            raise ValueError(f"{field_name} must not contain an empty reason code.")
        if code not in allowed_codes:
            raise ValueError(f"{field_name} contains unsupported code {code!r}.")
        codes.append(code)
    if len(set(codes)) != len(codes):
        raise ValueError(f"{field_name} must not contain duplicate reason codes.")
    return sorted(codes)


def _append(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)


def _validate_nonempty_fields(record: Mapping[str, Any], fields: Sequence[str], errors: list[str]) -> None:
    for field in fields:
        if not _text(record.get(field)):
            _append(errors, f"missing required field {field}.")


def _validate_present_sha256_fields(value: Any, field_path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{field_path}.{key}" if field_path else str(key)
            if str(key).endswith("_sha256") and child not in (None, "") and not is_sha256(child):
                _append(errors, f"{child_path} must be a lowercase 64-character SHA256.")
            _validate_present_sha256_fields(child, child_path, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_present_sha256_fields(child, f"{field_path}[{index}]", errors)


def _validate_asset_binding(name: str, asset: Mapping[str, Any], errors: list[str]) -> None:
    status = _text(asset.get("status"))
    if status not in ASSET_STATUSES:
        _append(errors, f"asset {name} has invalid status {status!r}.")
        return
    if status == "READY":
        _validate_nonempty_fields(
            asset,
            ("authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256"),
            errors,
        )
        for field in ("artifact_sha256", "receipt_sha256"):
            if not is_sha256(asset.get(field)):
                _append(errors, f"asset {name}.{field} must be a lowercase SHA256 when READY.")


def _validate_audit_binding(name: str, audit: Mapping[str, Any], errors: list[str]) -> None:
    status = _text(audit.get("status"))
    if status not in AUDIT_STATUSES:
        _append(errors, f"audit {name} has invalid status {status!r}.")
        return
    if status == "PASS":
        _validate_nonempty_fields(
            audit,
            ("authority_root", "artifact_path_or_key", "artifact_sha256", "receipt_path", "receipt_sha256"),
            errors,
        )
        for field in ("artifact_sha256", "receipt_sha256"):
            if not is_sha256(audit.get(field)):
                _append(errors, f"audit {name}.{field} must be a lowercase SHA256 when PASS.")


def _asset_status(record: Mapping[str, Any], name: str) -> str:
    asset = _mapping(record.get(name))
    return _text(asset.get("status")) if asset is not None else ""


def _has_frozen_binding(value: Any) -> bool:
    binding = _mapping(value)
    if binding is None:
        return False
    return all(
        _text(binding.get(field))
        for field in ("authority_root", "artifact_path_or_key", "receipt_path")
    ) and all(is_sha256(binding.get(field)) for field in ("artifact_sha256", "receipt_sha256"))


def _stage0_exclusion_code(stage0: Mapping[str, Any]) -> str | None:
    route = _text(stage0.get("route")).lower().replace("_", "-")
    return {
        "review": "STAGE0_REVIEW",
        "reject": "STAGE0_REJECT",
        "no-gaze": "STAGE0_NO_GAZE",
        "gaze-disabled": "STAGE0_GAZE_DISABLED",
    }.get(route)


def _validate_effective_report(asset: Mapping[str, Any], errors: list[str]) -> None:
    if _text(asset.get("status")) != "READY":
        return
    _validate_nonempty_fields(
        asset,
        (
            "final_report_route",
            "effective_report_sha256",
            "effective_report_text_sha256",
            "text_source_type",
            "final_report_route",
        ),
        errors,
    )
    if _text(asset.get("final_report_route")) not in _ACCEPTED_REPORT_ROUTES:
        _append(errors, "effective_report.final_report_route is not final accepted.")
    if _text(asset.get("text_source_type")) not in TEXT_SOURCE_TYPES:
        _append(errors, "effective_report.text_source_type must be real or generated; structured is forbidden.")
    if asset.get("effective_report_sha256") != asset.get("effective_report_text_sha256"):
        _append(errors, "effective_report text SHA256 must equal effective_report_sha256.")
    if not is_sha256(asset.get("effective_report_sha256")):
        _append(errors, "effective_report.effective_report_sha256 must be a lowercase SHA256.")


def _validate_embedding(asset: Mapping[str, Any], report_hash: str, errors: list[str]) -> None:
    if _text(asset.get("status")) != "READY":
        return
    _validate_nonempty_fields(asset, ("embedding_key", "embedding_effective_report_sha256"), errors)
    if asset.get("embedding_effective_report_sha256") != report_hash:
        _append(errors, "embedding effective_report_sha256 lineage does not match Effective Report.")
    if not isinstance(asset.get("embedding_dim"), int) or int(asset["embedding_dim"]) <= 0:
        _append(errors, "embedding.embedding_dim must be a positive integer.")


def _validate_graph(asset: Mapping[str, Any], report_hash: str, errors: list[str]) -> None:
    if _text(asset.get("status")) != "READY":
        return
    _validate_nonempty_fields(asset, ("graph_input_kind", "graph_effective_report_sha256", "graph_schema_version"), errors)
    if asset.get("graph_input_kind") != "effective_report":
        _append(errors, "clinical_graph.graph_input_kind must be effective_report.")
    if asset.get("graph_effective_report_sha256") != report_hash:
        _append(errors, "Clinical Graph effective_report_sha256 lineage does not match Effective Report.")


def _validate_concept_targets(asset: Mapping[str, Any], record: Mapping[str, Any], report_hash: str, errors: list[str]) -> None:
    from breast_pretrain.data.stage1_fullpool_concept_validation import validate_p0b_concept_binding

    errors.extend(validate_p0b_concept_binding(asset, record, report_hash))


def _validate_semantic(asset: Mapping[str, Any], report_hash: str, graph: Mapping[str, Any], errors: list[str]) -> None:
    if _text(asset.get("status")) != "READY":
        return
    _validate_nonempty_fields(asset, ("semantic_effective_report_sha256", "semantic_graph_sha256", "semantic_schema_version"), errors)
    if asset.get("semantic_effective_report_sha256") != report_hash:
        _append(errors, "semantic soft-label effective_report_sha256 lineage does not match Effective Report.")
    if asset.get("semantic_graph_sha256") != graph.get("artifact_sha256"):
        _append(errors, "semantic soft-label graph SHA256 lineage does not match Clinical Graph artifact.")


def _validate_reason_state(record: Mapping[str, Any], errors: list[str]) -> tuple[list[str], list[str]]:
    try:
        blocking = normalize_reason_codes(record.get("blocking_reason_codes"), BLOCKING_REASON_CODES, "blocking_reason_codes")
    except ValueError as exc:
        _append(errors, str(exc))
        blocking = []
    try:
        exclusion = normalize_reason_codes(record.get("exclusion_reason_codes"), EXCLUSION_REASON_CODES, "exclusion_reason_codes")
    except ValueError as exc:
        _append(errors, str(exc))
        exclusion = []
    if blocking and exclusion:
        _append(errors, "blocking_reason_codes and exclusion_reason_codes are mutually exclusive.")
    pending_assets = {
        "effective_report": {"PENDING_FINAL_CLOSURE", "PENDING_EFFECTIVE_REPORT"},
        "embedding": {"PENDING_EFFECTIVE_REPORT", "PENDING_EMBEDDING"},
        "clinical_graph": {"PENDING_GRAPH"},
        "concept": {"PENDING_CONCEPTS"},
        "semantic_soft_label": {"PENDING_SOFT_LABEL"},
    }
    for asset_name, allowed_codes in pending_assets.items():
        if _asset_status(record, asset_name) == "PENDING" and not (set(blocking) & allowed_codes):
            _append(errors, f"pending {asset_name} must carry its recoverable blocking_reason_code.")
    if any(_asset_status(record, name) == "BLOCKED" for name in _REQUIRED_ASSETS) and not blocking:
        _append(errors, "BLOCKED required asset family must carry blocking_reason_codes.")
    if any(_asset_status(record, name) == "EXCLUDED" for name in _REQUIRED_ASSETS) and not exclusion:
        _append(errors, "EXCLUDED required asset family must carry exclusion_reason_codes.")
    stage0 = _mapping(record.get("stage0")) or {}
    stage0_code = _stage0_exclusion_code(stage0)
    if stage0_code and stage0_code not in exclusion:
        _append(errors, f"Stage0 route requires exclusion_reason_code={stage0_code}.")
    if exclusion and any(_asset_status(record, name) == "PENDING" for name in _REQUIRED_ASSETS):
        _append(errors, "permanently excluded record must not carry recoverable PENDING asset status.")
    return blocking, exclusion


def compute_candidate_ready(record: Mapping[str, Any]) -> bool:
    """Compute frozen readiness without mutating the candidate record."""

    if any(_asset_status(record, name) != "READY" for name in _REQUIRED_ASSETS):
        return False
    if not _has_frozen_binding(record.get("dataset_authority")):
        return False
    if any(not _has_frozen_binding(record.get(name)) for name in _REQUIRED_ASSETS):
        return False
    audits = _mapping(record.get("audits"))
    if audits is None or any(
        not isinstance(audits.get(name), Mapping)
        or _text(audits[name].get("status")) != "PASS"
        or not _has_frozen_binding(audits[name])
        for name in _REQUIRED_AUDITS
    ):
        return False
    if record.get("blocking_reason_codes") or record.get("exclusion_reason_codes"):
        return False
    stage0 = _mapping(record.get("stage0")) or {}
    gaze = _mapping(record.get("gaze")) or {}
    patch = _mapping(record.get("patch")) or {}
    report = _mapping(record.get("effective_report")) or {}
    if stage0.get("usable") is not True or _stage0_exclusion_code(stage0) is not None:
        return False
    if gaze.get("gaze_training_enabled") is not True:
        return False
    if gaze.get("gaze_prior_available") is not True or _text(gaze.get("gaze_prior_quality")) != "usable":
        return False
    required_gaze = (
        "gaze_prior_sha256",
        "gaze_to_patch_projection_key",
        "gaze_to_patch_projection_sha256",
        "patch_token_order_version",
    )
    required_patch = (
        "patch_geometry_key",
        "patch_geometry_sha256",
        "valid_content_mask_key",
        "valid_content_mask_sha256",
        "patch_token_order_version",
    )
    if any(not _text(gaze.get(field)) for field in required_gaze) or any(not _text(patch.get(field)) for field in required_patch):
        return False
    if not (_text(gaze.get("gaze_prior_key")) or _text(gaze.get("artifact_path_or_key"))):
        return False
    if any(not is_sha256(gaze.get(field)) for field in ("gaze_prior_sha256", "gaze_to_patch_projection_sha256")):
        return False
    if any(not is_sha256(patch.get(field)) for field in ("patch_geometry_sha256", "valid_content_mask_sha256")):
        return False
    if gaze.get("patch_token_order_version") != patch.get("patch_token_order_version"):
        return False
    if _text(report.get("final_report_route")) not in _ACCEPTED_REPORT_ROUTES:
        return False
    if _text(report.get("text_source_type")) not in TEXT_SOURCE_TYPES:
        return False
    from breast_pretrain.data.stage1_fullpool_identity_validation import candidate_has_ready_modality_identity

    if not candidate_has_ready_modality_identity(record):
        return False
    return True


def _expected_hard_gate_status(record: Mapping[str, Any]) -> str:
    if compute_candidate_ready(record):
        return "PASS"
    if record.get("exclusion_reason_codes"):
        return "EXCLUDED"
    if record.get("blocking_reason_codes"):
        return "PENDING"
    return "BLOCKED"


def immutable_record_sha256(record: Mapping[str, Any], hash_field: str) -> str:
    payload = copy.deepcopy(dict(record))
    payload.pop(hash_field, None)
    return sha256_bytes(canonical_json_bytes(payload))


def validate_candidate_record(record: Any, *, verify_derived_fields: bool = True) -> list[str]:
    """Return all bounded validation errors for one frozen candidate registry row."""

    errors: list[str] = []
    item = _mapping(record)
    if item is None:
        return ["candidate registry row must be an object."]
    if "train_eligible" in item:
        _append(errors, "candidate registry must not contain train_eligible.")
    if item.get("registry_schema_version") != CANDIDATE_REGISTRY_SCHEMA_VERSION:
        _append(errors, "candidate registry row has unsupported registry_schema_version.")
    _validate_nonempty_fields(item, _REQUIRED_IDENTITY_FIELDS, errors)
    for field in ("source_image_sha256", "canonical_image_sha256", "image_membership_sha256", "binding_config_sha256"):
        if not is_sha256(item.get(field)):
            _append(errors, f"{field} must be a lowercase SHA256.")
    authority = _mapping(item.get("dataset_authority"))
    if authority is None:
        _append(errors, "dataset_authority must be an object.")
    else:
        _validate_asset_binding("dataset_authority", {"status": "READY", **authority}, errors)
    for name in _REQUIRED_ASSETS:
        asset = _mapping(item.get(name))
        if asset is None:
            _append(errors, f"asset {name} must be an object.")
        else:
            _validate_asset_binding(name, asset, errors)
    audits = _mapping(item.get("audits"))
    if audits is None:
        _append(errors, "audits must be an object.")
    else:
        for name in _REQUIRED_AUDITS:
            audit = _mapping(audits.get(name))
            if audit is None:
                _append(errors, f"audit {name} must be an authority binding object.")
            else:
                _validate_audit_binding(name, audit, errors)
    blocking, exclusion = _validate_reason_state(item, errors)
    report = _mapping(item.get("effective_report")) or {}
    embedding = _mapping(item.get("embedding")) or {}
    graph = _mapping(item.get("clinical_graph")) or {}
    concept = _mapping(item.get("concept")) or {}
    semantic = _mapping(item.get("semantic_soft_label")) or {}
    _validate_effective_report(report, errors)
    report_hash = _text(report.get("effective_report_sha256"))
    _validate_embedding(embedding, report_hash, errors)
    _validate_graph(graph, report_hash, errors)
    _validate_concept_targets(concept, item, report_hash, errors)
    _validate_semantic(semantic, report_hash, graph, errors)
    stage0 = _mapping(item.get("stage0")) or {}
    gaze = _mapping(item.get("gaze")) or {}
    patch = _mapping(item.get("patch")) or {}
    if _asset_status(item, "stage0") == "READY" and stage0.get("usable") is not True:
        _append(errors, "READY stage0 must have usable=true.")
    stage0_code = _stage0_exclusion_code(stage0)
    if _asset_status(item, "stage0") == "READY" and stage0_code is not None:
        _append(errors, f"READY stage0 must not use formal-excluded route={stage0.get('route')!r}.")
    if _asset_status(item, "gaze") == "READY":
        if gaze.get("gaze_training_enabled") is not True or gaze.get("gaze_prior_available") is not True:
            _append(errors, "READY gaze must have gaze_training_enabled=true and gaze_prior_available=true.")
        if _text(gaze.get("gaze_prior_quality")) != "usable":
            _append(errors, "READY gaze.gaze_prior_quality must be formal usable.")
        _validate_nonempty_fields(
            gaze,
            ("gaze_to_patch_projection_key", "patch_token_order_version"),
            errors,
        )
        if not (_text(gaze.get("gaze_prior_key")) or _text(gaze.get("artifact_path_or_key"))):
            _append(errors, "READY gaze requires gaze_prior_key or an artifact reference.")
        for field in ("gaze_prior_sha256", "gaze_to_patch_projection_sha256"):
            if not is_sha256(gaze.get(field)):
                _append(errors, f"READY gaze.{field} must be a lowercase SHA256.")
    if _asset_status(item, "patch") == "READY":
        _validate_nonempty_fields(patch, ("patch_geometry_key", "valid_content_mask_key", "patch_token_order_version"), errors)
        for field in ("patch_geometry_sha256", "valid_content_mask_sha256"):
            if not is_sha256(patch.get(field)):
                _append(errors, f"READY patch.{field} must be a lowercase SHA256.")
    if _asset_status(item, "gaze") == "READY" and _asset_status(item, "patch") == "READY":
        if gaze.get("patch_token_order_version") != patch.get("patch_token_order_version"):
            _append(errors, "gaze and patch patch_token_order_version must match.")
        for gaze_field, patch_field in (
            ("projection_geometry_checksum", "patch_geometry_checksum"),
            ("projection_geometry_version", "patch_geometry_version"),
        ):
            if _text(gaze.get(gaze_field)) and _text(patch.get(patch_field)) and gaze.get(gaze_field) != patch.get(patch_field):
                _append(errors, f"gaze.{gaze_field} must match patch.{patch_field} when both are frozen.")
    _validate_present_sha256_fields(item, "", errors)
    if verify_derived_fields:
        expected_ready = compute_candidate_ready(item)
        if item.get("candidate_ready") is not expected_ready:
            _append(errors, "candidate_ready does not match deterministic asset readiness.")
        from breast_pretrain.data.stage1_fullpool_identity_validation import validate_ready_modality_identity

        errors.extend(validate_ready_modality_identity(item))
        expected_gate_status = _expected_hard_gate_status(item)
        if item.get("hard_gate_status") != expected_gate_status:
            _append(errors, "hard_gate_status does not match immutable reason/readiness state.")
        expected_failures = sorted([*blocking, *exclusion])
        if item.get("hard_gate_failures") != expected_failures:
            _append(errors, "hard_gate_failures does not match immutable reason codes.")
        if item.get("candidate_record_sha256") != immutable_record_sha256(item, "candidate_record_sha256"):
            _append(errors, "candidate_record_sha256 does not match canonical record serialization.")
    return errors


def materialize_candidate_record(source_record: Mapping[str, Any], *, binding_config_sha256: str) -> dict[str, Any]:
    """Freeze one authority-derived source row; never infer or repair assets."""

    record = copy.deepcopy(dict(source_record))
    forbidden = {"train_eligible", "candidate_ready", "hard_gate_status", "hard_gate_failures", "candidate_record_sha256"}
    present = sorted(forbidden & set(record))
    if present:
        raise ValueError("candidate source record must not supply derived/runtime fields: " + ", ".join(present))
    if not is_sha256(binding_config_sha256):
        raise ValueError("binding_config_sha256 must be a lowercase SHA256.")
    record["registry_schema_version"] = CANDIDATE_REGISTRY_SCHEMA_VERSION
    record["binding_config_sha256"] = binding_config_sha256
    record["blocking_reason_codes"] = normalize_reason_codes(
        record.get("blocking_reason_codes"), BLOCKING_REASON_CODES, "blocking_reason_codes"
    )
    record["exclusion_reason_codes"] = normalize_reason_codes(
        record.get("exclusion_reason_codes"), EXCLUSION_REASON_CODES, "exclusion_reason_codes"
    )
    record["candidate_ready"] = compute_candidate_ready(record)
    record["hard_gate_status"] = _expected_hard_gate_status(record)
    record["hard_gate_failures"] = sorted([*record["blocking_reason_codes"], *record["exclusion_reason_codes"]])
    record["candidate_record_sha256"] = immutable_record_sha256(record, "candidate_record_sha256")
    errors = validate_candidate_record(record)
    if errors:
        raise ValueError("invalid candidate source record: " + " | ".join(errors))
    return record


def build_case_record(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the immutable case/report registry row for one candidate-row group."""

    if not candidate_rows:
        raise ValueError("cannot build a case registry row from no candidate rows.")
    first = candidate_rows[0]
    keys = ("dataset_id", "case_id", "report_unit_id", "report_id", "patient_id", "patient_split")
    if any(any(row.get(key) != first.get(key) for key in keys) for row in candidate_rows[1:]):
        raise ValueError("candidate rows in a case group disagree on dataset/case/report identity.")
    shared_asset_names = ("effective_report", "embedding", "clinical_graph", "concept", "semantic_soft_label")
    for name in shared_asset_names:
        baseline = canonical_json_bytes(first.get(name))
        if any(canonical_json_bytes(row.get(name)) != baseline for row in candidate_rows[1:]):
            raise ValueError(f"candidate rows in one case group disagree on shared {name} binding.")
    image_ids = sorted(str(row["canonical_image_id"]) for row in candidate_rows)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("case group contains duplicate canonical_image_id values.")
    report = _mapping(first.get("effective_report")) or {}
    record: dict[str, Any] = {
        "registry_schema_version": CASE_REGISTRY_SCHEMA_VERSION,
        "dataset_id": first["dataset_id"],
        "case_id": first["case_id"],
        "report_unit_id": first["report_unit_id"],
        "report_id": first["report_id"],
        "patient_id": first["patient_id"],
        "patient_split": first["patient_split"],
        "study_ids": sorted({_text(row.get("study_id")) for row in candidate_rows if _text(row.get("study_id"))}),
        "effective_report_sha256": report.get("effective_report_sha256", ""),
        "case_image_ids": image_ids,
        "case_image_membership_sha256": sha256_bytes(canonical_json_bytes(image_ids)),
        "case_report_cardinality": 1,
        "case_candidate_ready": all(row.get("candidate_ready") is True for row in candidate_rows),
        "candidate_record_sha256s": sorted(str(row["candidate_record_sha256"]) for row in candidate_rows),
    }
    record["case_record_sha256"] = immutable_record_sha256(record, "case_record_sha256")
    return record


def validate_case_record(record: Any) -> list[str]:
    errors: list[str] = []
    item = _mapping(record)
    if item is None:
        return ["case registry row must be an object."]
    if item.get("registry_schema_version") != CASE_REGISTRY_SCHEMA_VERSION:
        _append(errors, "case registry row has unsupported registry_schema_version.")
    _validate_nonempty_fields(item, ("dataset_id", "case_id", "report_unit_id", "report_id", "patient_id", "patient_split"), errors)
    study_ids = item.get("study_ids")
    if not isinstance(study_ids, list) or study_ids != sorted(set(study_ids)) or any(not _text(value) for value in study_ids):
        _append(errors, "study_ids must be a sorted unique list of explicit study identities.")
    image_ids = item.get("case_image_ids")
    if not isinstance(image_ids, list) or not image_ids or image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        _append(errors, "case_image_ids must be a non-empty sorted unique list.")
    elif item.get("case_image_membership_sha256") != sha256_bytes(canonical_json_bytes(image_ids)):
        _append(errors, "case_image_membership_sha256 does not match case_image_ids.")
    if item.get("case_report_cardinality") != 1:
        _append(errors, "case_report_cardinality must be exactly 1.")
    if not isinstance(item.get("case_candidate_ready"), bool):
        _append(errors, "case_candidate_ready must be boolean.")
    if item.get("case_record_sha256") != immutable_record_sha256(item, "case_record_sha256"):
        _append(errors, "case_record_sha256 does not match canonical record serialization.")
    return errors
