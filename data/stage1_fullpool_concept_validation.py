"""P0-B sparse case-target validation for the immutable Stage 1 candidate registry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_MODALITY_SCOPES = frozenset({"shared", "mammography", "ultrasound", "MRI"})
_TARGET_STATUSES = frozenset({"positive", "explicit_negative", "missing", "not_applicable", "uncertain"})
_SUPERVISION_MODES = frozenset({"direct", "prototype", "consistency"})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _append(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)


def _has_target_value(item: Mapping[str, Any]) -> bool:
    value = item.get("target_value")
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _valid_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_p0b_concept_binding(
    asset: Mapping[str, Any], record: Mapping[str, Any], effective_report_sha256: str
) -> list[str]:
    """Validate a READY concept asset without deriving targets or supervision modes."""

    errors: list[str] = []
    if _text(asset.get("status")) != "READY":
        return errors
    schema_version = _text(asset.get("concept_schema_version"))
    if not schema_version:
        _append(errors, "concept.concept_schema_version is required.")
    if asset.get("concept_effective_report_sha256") != effective_report_sha256:
        _append(errors, "concept effective_report_sha256 lineage does not match Effective Report.")
    if asset.get("concept_case_id") != record.get("case_id"):
        _append(errors, "concept_case_id does not match record case_id.")
    schema = _mapping(asset.get("concept_schema"))
    if schema is None:
        return [*errors, "concept.concept_schema must be an object."]
    if schema.get("concept_schema_version") != schema_version:
        _append(errors, "asset.concept_schema_version must equal concept_schema.concept_schema_version.")
    concepts = schema.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        return [*errors, "concept.concept_schema.concepts must be a non-empty list."]
    schema_ids: set[str] = set()
    scopes: set[str] = set()
    for index, value in enumerate(concepts):
        concept = _mapping(value)
        if concept is None:
            _append(errors, f"concept schema item {index} must be an object.")
            continue
        if "target_state" in concept:
            _append(errors, f"concept schema item {index} uses forbidden target_state.")
        for field in (
            "concept_schema_version",
            "concept_id",
            "modality_scope",
            "target_type",
            "value_space",
            "source_policy",
            "confidence_policy",
        ):
            if not _text(concept.get(field)):
                _append(errors, f"concept schema item {index} misses {field}.")
        if concept.get("concept_schema_version") != schema_version:
            _append(errors, f"concept schema item {index} version does not match asset.concept_schema_version.")
        concept_id = _text(concept.get("concept_id"))
        if concept_id in schema_ids:
            _append(errors, f"concept schema has duplicate concept_id={concept_id!r}.")
        schema_ids.add(concept_id)
        scope = _text(concept.get("modality_scope"))
        scopes.add(scope)
        if scope not in _MODALITY_SCOPES:
            _append(errors, f"concept schema item {index} has invalid modality_scope={scope!r}.")
        modes = _mapping(concept.get("supervision_modes"))
        if modes is None or set(modes) != _SUPERVISION_MODES or not all(isinstance(item, bool) for item in modes.values()):
            _append(errors, f"concept schema item {index} must carry boolean direct/prototype/consistency supervision_modes.")
        elif not any(modes.values()):
            _append(errors, f"concept schema item {index} must enable at least one supervision mode.")
    missing_scopes = sorted(_MODALITY_SCOPES - scopes)
    if missing_scopes:
        _append(errors, "concept schema must include shared and all modality-specific scopes: " + ", ".join(missing_scopes))
    targets = asset.get("case_targets")
    if not isinstance(targets, list) or not targets:
        return [*errors, "concept.case_targets must be a non-empty sparse case-level list when READY."]
    for index, value in enumerate(targets):
        target = _mapping(value)
        if target is None:
            _append(errors, f"concept target {index} must be an object.")
            continue
        if "target_state" in target:
            _append(errors, f"concept target {index} uses forbidden target_state.")
        for field in ("concept_id", "target_status", "source_type", "effective_report_sha256"):
            if not _text(target.get(field)):
                _append(errors, f"concept target {index} misses {field}.")
        if "target_value" not in target:
            _append(errors, f"concept target {index} must explicitly contain target_value.")
        if "confidence" not in target or not _valid_confidence(target.get("confidence")):
            _append(errors, f"concept target {index}.confidence must be a finite numeric value.")
        concept_id = _text(target.get("concept_id"))
        if concept_id not in schema_ids:
            _append(errors, f"concept target {index} concept_id={concept_id!r} is not in the frozen concept schema.")
        status = _text(target.get("target_status"))
        if status not in _TARGET_STATUSES:
            _append(errors, f"concept target {index} has invalid target_status={status!r}.")
        booleans: dict[str, bool] = {}
        for field in ("modality_applicable", "target_available", "valid_target_mask"):
            value_bool = target.get(field)
            if not isinstance(value_bool, bool):
                _append(errors, f"concept target {index}.{field} must be boolean.")
            else:
                booleans[field] = value_bool
        if len(booleans) == 3:
            expected_mask = booleans["modality_applicable"] and booleans["target_available"]
            if booleans["valid_target_mask"] is not expected_mask:
                _append(errors, f"concept target {index} valid_target_mask must equal modality_applicable × target_available.")
            if status in {"positive", "explicit_negative"}:
                if not all(booleans[field] for field in ("modality_applicable", "target_available", "valid_target_mask")):
                    _append(errors, f"concept target {index} {status} must be applicable, available, and valid.")
                if not _has_target_value(target):
                    _append(errors, f"concept target {index} {status} must carry a non-empty target_value.")
            elif status == "missing":
                if booleans != {"modality_applicable": True, "target_available": False, "valid_target_mask": False}:
                    _append(errors, f"concept target {index} missing must be applicable, unavailable, and invalid.")
            elif status == "not_applicable":
                if booleans != {"modality_applicable": False, "target_available": False, "valid_target_mask": False}:
                    _append(errors, f"concept target {index} not_applicable must be inapplicable, unavailable, and invalid.")
            elif status == "uncertain" and booleans["valid_target_mask"] is not False:
                _append(errors, f"concept target {index} uncertain must not participate in a valid target mask.")
        if target.get("effective_report_sha256") != effective_report_sha256:
            _append(errors, f"concept target {index} effective_report_sha256 lineage mismatch.")
        lineage = _mapping(target.get("case_report_hash_lineage"))
        if lineage is None or any(
            lineage.get(field) != record.get(field)
            for field in ("case_id", "report_unit_id", "report_id")
        ) or lineage.get("effective_report_sha256") != effective_report_sha256:
            _append(errors, f"concept target {index} has invalid case/report/hash lineage.")
    return errors
