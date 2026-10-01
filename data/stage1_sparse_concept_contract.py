"""Frozen P0-B sparse concept authority parsing and deterministic projection.

This module intentionally has no manifest-metadata reconstruction path.  It is
the only formal path from sparse clinical assertions to direct-loss tensors.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


SCHEMA_VERSION = "TRI_MODAL_CONCEPT_SCHEMA_CANDIDATE_V0.3.2"
VALID_STATUSES = frozenset({"positive", "explicit_negative", "missing", "not_applicable", "uncertain"})
FORBIDDEN_SOURCES = frozenset({"structured_prompt", "filename", "path", "dataset_id", "engineering_qc_state", "bbox", "mask", "radiomics", "provenance_family", "dicom_acquisition_metadata"})
_REQUIRED_ROW_FIELDS = frozenset({
    "concept_schema_version", "concept_id", "case_id", "report_unit_id", "report_id",
    "effective_report_sha256", "modality", "target_type", "case_value_cardinality",
    "target_value", "target_status", "modality_applicable", "target_available",
    "valid_target_mask", "source_type", "source_ref", "confidence", "graph_node_ids",
    "scope_type", "scope_key", "supervision_modes", "materializer_version",
    "source_authority_hashes",
})


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ConceptSpec:
    concept_id: str
    modality_scope: str
    target_type: str
    value_space: tuple[str, ...]
    case_value_cardinality: str
    supervision_modes: dict[str, bool]
    source_policy: str
    confidence_policy: str
    modality_applicability_rule: str

    @property
    def direct(self) -> bool:
        return bool(self.supervision_modes.get("direct", False))


@dataclass(frozen=True)
class FrozenConceptSchema:
    version: str
    concepts: dict[str, ConceptSpec]
    source_policy_registry: dict[str, dict[str, Any]]
    confidence_policy_registry: dict[str, dict[str, Any]]
    yaml_sha256: str

    @property
    def direct_concept_ids(self) -> tuple[str, ...]:
        return tuple(spec.concept_id for spec in self.concepts.values() if spec.direct)

    def allowed_sources(self, spec: ConceptSpec, mode: str) -> frozenset[str]:
        policy = self.source_policy_registry.get(spec.source_policy)
        if not isinstance(policy, dict):
            raise ValueError(f"Unknown source policy for {spec.concept_id}: {spec.source_policy}")
        return frozenset(str(item) for item in policy.get(f"{mode}_allowed_sources", []))

    def validate_row_for_mode(self, spec: ConceptSpec, row: "SparseConceptRow", mode: str) -> bool:
        source = str(row.payload["source_type"])
        underlying = str(row.payload.get("underlying_source_type", ""))
        effective_source = underlying if source == "graph_assertion" else source
        if effective_source not in self.allowed_sources(spec, mode):
            return False
        confidence_policy = self.confidence_policy_registry.get(spec.confidence_policy)
        if not isinstance(confidence_policy, dict):
            raise ValueError(f"Unknown confidence policy for {spec.concept_id}: {spec.confidence_policy}")
        threshold = confidence_policy.get("model_inference_threshold")
        if effective_source == "model_inference" and (threshold is None or row.confidence < float(threshold)):
            return False
        rule = spec.modality_applicability_rule.upper()
        if "DWI" in rule or "CONTRAST_ENHANCED_MRI" in rule or "TRUSTED_T2" in rule:
            if not bool(row.payload.get("sequence_available", False)):
                return False
        if "DOPPLER" in rule and not bool(row.payload.get("doppler_available", False)):
            return False
        return True


def load_frozen_concept_schema(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenConceptSchema:
    """Load V0.3.2 and reject any authority/hash/count drift."""
    schema_path = Path(path)
    actual_hash = sha256_file(schema_path)
    if expected_sha256 is not None and actual_hash != expected_sha256.lower():
        raise ValueError(f"Concept schema SHA256 mismatch: expected {expected_sha256}, got {actual_hash}")
    try:
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid concept schema YAML {schema_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("concept_schema_version") != SCHEMA_VERSION:
        raise ValueError("P0-B requires TRI_MODAL_CONCEPT_SCHEMA_CANDIDATE_V0.3.2.")
    raw_concepts = payload.get("concepts")
    registry = payload.get("source_policy_registry")
    confidence_registry = payload.get("confidence_policy_registry")
    if not isinstance(raw_concepts, list) or not isinstance(registry, dict) or not isinstance(confidence_registry, dict):
        raise ValueError("Frozen schema is missing source/confidence policy registries.")
    concepts: dict[str, ConceptSpec] = {}
    for raw in raw_concepts:
        if not isinstance(raw, dict):
            raise ValueError("Frozen schema contains a non-mapping concept.")
        try:
            spec = ConceptSpec(
                concept_id=str(raw["concept_id"]), modality_scope=str(raw["modality_scope"]),
                target_type=str(raw["target_type"]), value_space=tuple(str(v) for v in raw["value_space"]),
                case_value_cardinality=str(raw["case_value_cardinality"]),
                supervision_modes={str(k): bool(v) for k, v in raw["supervision_modes"].items()},
                source_policy=str(raw["source_policy"]), confidence_policy=str(raw["confidence_policy"]),
                modality_applicability_rule=str(raw["modality_applicability_rule"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed concept schema entry: {raw!r}") from exc
        if not spec.concept_id or spec.concept_id in concepts or not spec.value_space:
            raise ValueError(f"Duplicate/invalid frozen concept id: {spec.concept_id!r}")
        concepts[spec.concept_id] = spec
    scope_counts = {
        "shared": sum(spec.modality_scope.lower() == "shared" for spec in concepts.values()),
        "mammography": sum(spec.modality_scope.lower() == "mammography" for spec in concepts.values()),
        "ultrasound": sum(spec.modality_scope.lower() == "ultrasound" for spec in concepts.values()),
        "mri": sum(spec.modality_scope.lower() == "mri" for spec in concepts.values()),
    }
    mode_counts = {mode: sum(bool(spec.supervision_modes.get(mode)) for spec in concepts.values()) for mode in ("direct", "prototype", "consistency")}
    if len(concepts) != 37 or scope_counts != {"shared": 8, "mammography": 10, "ultrasound": 7, "mri": 12} or mode_counts != {"direct": 35, "prototype": 31, "consistency": 35}:
        raise ValueError(f"Frozen schema count drift: concepts={len(concepts)}, scopes={scope_counts}, modes={mode_counts}")
    if {"shared.modality_identity", "mammography.projection_view"} & set(
        spec.concept_id for spec in concepts.values() if spec.direct
    ):
        raise ValueError("Audit-only concepts must not have formal direct heads.")
    return FrozenConceptSchema(SCHEMA_VERSION, concepts, registry, confidence_registry, actual_hash)


@dataclass(frozen=True)
class SparseConceptRow:
    payload: dict[str, Any]

    @property
    def concept_id(self) -> str:
        return str(self.payload["concept_id"])

    @property
    def target_value(self) -> str:
        return str(self.payload.get("target_value", ""))

    @property
    def status(self) -> str:
        return str(self.payload["target_status"])

    @property
    def confidence(self) -> float:
        return float(self.payload["confidence"])

    def canonical_scope(self) -> tuple[str, str]:
        return str(self.payload["scope_type"]), str(self.payload["scope_key"])

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], schema: FrozenConceptSchema) -> "SparseConceptRow":
        missing = sorted(_REQUIRED_ROW_FIELDS - set(raw))
        if missing:
            raise ValueError("Sparse concept row misses fields: " + ", ".join(missing))
        concept_id = str(raw["concept_id"])
        spec = schema.concepts.get(concept_id)
        if spec is None:
            raise ValueError(f"Sparse concept row has unknown concept_id={concept_id!r}")
        if str(raw["concept_schema_version"]) != schema.version:
            raise ValueError(f"Sparse concept row schema version mismatch for {concept_id}")
        if str(raw["target_type"]) != spec.target_type or str(raw["case_value_cardinality"]) != spec.case_value_cardinality:
            raise ValueError(f"Sparse concept row type/cardinality mismatch for {concept_id}")
        if raw["supervision_modes"] != spec.supervision_modes:
            raise ValueError(f"Sparse concept row supervision_modes mismatch for {concept_id}")
        status = str(raw["target_status"])
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid target_status={status!r} for {concept_id}")
        value = str(raw.get("target_value", ""))
        if status in {"positive", "explicit_negative"} and value not in spec.value_space:
            raise ValueError(f"Sparse concept row target_value is outside frozen value_space for {concept_id}")
        report_hash = str(raw["effective_report_sha256"])
        if not re.fullmatch(r"[0-9a-fA-F]{64}", report_hash):
            raise ValueError(f"Invalid effective_report_sha256 for {concept_id}")
        authority_hashes = raw["source_authority_hashes"]
        if not isinstance(authority_hashes, dict) or not authority_hashes or any(not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)) for value in authority_hashes.values()):
            raise ValueError(f"Invalid source_authority_hashes ledger for {concept_id}")
        try:
            confidence = float(raw["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence for {concept_id}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence outside [0,1] for {concept_id}")
        source = str(raw["source_type"])
        if source in FORBIDDEN_SOURCES:
            raise ValueError(f"Forbidden target authority source={source!r} for {concept_id}")
        return cls(dict(raw))


@dataclass(frozen=True)
class RuntimeConceptTarget:
    concept_id: str
    target: int | np.ndarray
    valid_target_mask: bool
    value_valid_mask: np.ndarray | None
    raw_assertion_values: tuple[str, ...]
    target_available: bool
    conflict: bool
    conflict_ledger: tuple[dict[str, str], ...] = ()


def _is_modality_applicable(spec: ConceptSpec, modality: str, rows: list[SparseConceptRow]) -> bool:
    if spec.modality_scope.lower() == "shared":
        return any(bool(row.payload["modality_applicable"]) for row in rows)
    return modality.strip().lower() == spec.modality_scope.lower() and any(bool(row.payload["modality_applicable"]) for row in rows)


def _legal_rows(schema: FrozenConceptSchema, spec: ConceptSpec, rows: Iterable[SparseConceptRow], mode: str) -> list[SparseConceptRow]:
    return [row for row in rows if schema.validate_row_for_mode(spec, row, mode)]


def project_direct_target(
    schema: FrozenConceptSchema,
    concept_id: str,
    modality: str,
    rows: Iterable[SparseConceptRow],
    frozen_sample_scope: dict[str, str] | None = None,
) -> RuntimeConceptTarget:
    """Project only lawful sparse assertions; ambiguity/conflict always masks."""
    spec = schema.concepts[concept_id]
    candidates = [row for row in rows if row.concept_id == concept_id]
    legal = _legal_rows(schema, spec, candidates, "direct")
    scope_exact_miss = False
    if frozen_sample_scope:
        scope_type = str(frozen_sample_scope.get("scope_type", ""))
        scope_key = str(frozen_sample_scope.get("scope_key", ""))
        if scope_type and scope_key:
            exact = [row for row in legal if row.canonical_scope() == (scope_type, scope_key)]
            if exact:
                legal = exact
            else:
                scope_exact_miss = True
    applicable = _is_modality_applicable(spec, modality, candidates)
    values = tuple(sorted({row.target_value for row in legal if row.status in {"positive", "explicit_negative"} and row.target_value in spec.value_space}))
    scope_values: dict[tuple[str, str], set[str]] = {}
    for row in legal:
        if row.status in {"positive", "explicit_negative"} and row.target_value in spec.value_space:
            scope_values.setdefault(row.canonical_scope(), set()).add(row.target_value)
    conflict = any(len(scope) > 1 for scope in scope_values.values())
    ledger = tuple(
        {"concept_id": concept_id, "scope_type": scope_type, "scope_key": scope_key, "values": ",".join(sorted(values))}
        for (scope_type, scope_key), values in sorted(scope_values.items()) if len(values) > 1
    )
    if scope_exact_miss:
        placeholder = np.zeros(len(spec.value_space), dtype=np.float32) if spec.target_type == "categorical_multilabel" else 0
        value_mask = np.zeros(len(spec.value_space), dtype=np.bool_) if spec.target_type == "categorical_multilabel" else None
        return RuntimeConceptTarget(concept_id, placeholder, False, value_mask, (), False, False, ())
    if spec.target_type == "categorical_multilabel":
        target = np.zeros(len(spec.value_space), dtype=np.float32)
        mask = np.zeros(len(spec.value_space), dtype=np.bool_)
        for row in legal:
            if row.target_value not in spec.value_space:
                continue
            index = spec.value_space.index(row.target_value)
            if row.status == "positive":
                target[index], mask[index] = 1.0, True
            elif row.status == "explicit_negative":
                target[index], mask[index] = 0.0, True
        if conflict:
            # Conflict can only invalidate the disputed values; never broadcast a mask.
            for scoped in scope_values.values():
                if len(scoped) > 1:
                    for value in scoped:
                        mask[spec.value_space.index(value)] = False
        return RuntimeConceptTarget(concept_id, target, bool(applicable and mask.any()), mask, values, bool(mask.any()), conflict, ledger)
    resolved = set()
    for value_set in scope_values.values():
        if len(value_set) == 1:
            resolved.update(value_set)
    if not scope_values:
        resolved = set(values)
    if conflict or len(resolved) != 1 or not applicable:
        placeholder: int = 0
        return RuntimeConceptTarget(concept_id, placeholder, False, None, values, False, conflict, ledger)
    value = next(iter(resolved))
    if spec.target_type == "binary":
        target = 1 if value == "present" else 0
    else:
        target = spec.value_space.index(value)
    return RuntimeConceptTarget(concept_id, target, True, None, values, True, False, ledger)


def read_sparse_concept_jsonl(path: str | Path, schema: FrozenConceptSchema) -> list[SparseConceptRow]:
    rows: list[SparseConceptRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid sparse concept JSONL {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Sparse concept JSONL {path}:{line_number} is not an object")
            rows.append(SparseConceptRow.from_mapping(raw, schema))
    return rows
