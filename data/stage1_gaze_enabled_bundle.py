from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breast_pretrain.data.bucketed_stage1_dataloader import normalize_stage1_modality
from breast_pretrain.data.clinical_concept_canonicalization import canonicalize_concept_label
from breast_pretrain.data.stage1_identity_linkage import (
    identity_key,
    resolve_stage1_identity_linkage,
)
from breast_pretrain.data.stage1_gaze_bundle_sidecars import (
    clean_builder_managed_outputs,
    filter_birads_prior_manifest_and_files,
    filter_case_concept_vectors,
    filter_text_prompts_and_embeddings,
)


MANIFEST_NAME = "manifest_stage1_semantic.csv"
TEXT_PROMPTS_NAME = "text_prompts.jsonl"
PROMPT_EMBEDDINGS_NAME = "stage1_prompt_embeddings.json"
SEMANTIC_MANIFEST_NAME = "stage1_semantic_soft_label_manifest.csv"
SEMANTIC_TOPK_NPZ_NAME = "stage1_semantic_soft_labels_topk.npz"
SEMANTIC_TOPK_JSONL_NAME = "stage1_semantic_soft_labels_topk.jsonl"
BIRADS_PRIOR_MANIFEST_NAME = "stage1_birads_prior_manifest.csv"
CASE_CONCEPT_VECTOR_NAME = "tri_modal_case_concept_vector.jsonl"

EXPECTED_GAZE_SUPERVISION_SOURCE = "diffeye_generated_gaze"
EXPECTED_GAZE_PRIOR_STATUS = "usable_prior"
EXPECTED_GAZE_PRIOR_AUDIT_STATUS = "pass_audit"
EXPECTED_GAZE_LOSS_ENABLED = "1"
ALLOWED_AUDITED_AGGREGATION_SOURCES = {"strict_valid", "weak_valid"}
ALLOWED_AUDITED_PRIOR_QC_LEVELS = {"strict", "weak_qc"}
FORBIDDEN_AUDITED_AGGREGATION_SOURCES = {"all_for_diagnostic_only"}

DEFAULT_EXPECTED_MODALITY_COUNTS = {
    "mammography": 170,
    "mri": 173,
    "ultrasound": 176,
}

AUDITED_PATH_FIELD_ALIASES = {
    "attention_map_path": ("attention_map_path", "aggregated_attention_map_path", "heatmap_path"),
    "high_conf_mask_path": ("high_conf_mask_path", "mask_path", "high_confidence_mask_path"),
    "trajectory_qc_path": ("trajectory_qc_path", "trajectory_level_qc_path", "qc_path"),
}

AUDITED_METADATA_FIELD_ALIASES = {
    "prior_qc_level": ("prior_qc_level", "qc_level", "attention_prior_qc_level"),
    "aggregation_source": ("aggregation_source", "aggregated_source", "aggregation_manifest_path"),
}

AUDITED_STATUS_FIELD_ALIASES = {
    "prior_status": ("prior_status", "gaze_prior_status"),
    "gaze_loss_enabled": ("gaze_loss_enabled", "loss_enabled"),
    "aggregation_source": AUDITED_METADATA_FIELD_ALIASES["aggregation_source"],
    "prior_qc_level": AUDITED_METADATA_FIELD_ALIASES["prior_qc_level"],
    "audit_status": (
        "gaze_prior_audit_status",
        "audit_status",
        "prior_audit_status",
        "attention_prior_audit_status",
    ),
}

OUTPUT_MANIFEST_EXTRA_COLUMNS = (
    "canonical_stage1_image_id",
    "audited_prior_image_id",
    "stage0_inference_image_id",
    "identity_linkage_method",
    "identity_linkage_resolved_image_path",
    "gaze_supervision_source",
    "gaze_prior_status",
    "gaze_prior_audit_status",
    "gaze_loss_enabled",
    "attention_map_path",
    "high_conf_mask_path",
    "trajectory_qc_path",
    "prior_qc_level",
    "aggregation_source",
)

CONCEPT_TARGET_FIELDS = (
    "view",
    "laterality",
    "density",
    "finding",
    "birads",
    "cancer_label",
    "benign_malignant_label",
    "mri_sequence",
    "mri_treatment_response",
)
CONCEPT_TARGET_METADATA_SUFFIXES = (
    "observed_mask",
    "source",
    "confidence",
    "status",
)
OBSERVED_STATUS_VALUES = {"confirmed", "observed", "verified", "structured_label", "present"}
UNOBSERVED_STATUS_VALUES = {
    "",
    "missing",
    "raw_unconfirmed",
    "unknown",
    "not_applicable",
    "not_mentioned",
    "unconfirmed",
}
UNKNOWN_CONCEPT_VALUES = {"", "unknown", "unknown_not_provided", "missing", "not_applicable", "nan", "none"}


@dataclass(frozen=True)
class AuditedManifestSpec:
    modality: str
    path: Path


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy_mask(value: Any) -> bool | None:
    normalized = _clean(value).lower()
    if normalized in {"1", "true", "yes", "y", "observed", "confirmed"}:
        return True
    if normalized in {"0", "false", "no", "n", "missing", "unknown", "not_applicable", ""}:
        return False
    return None


def _canonical_observed_mask_input(row: dict[str, str], concept: str, status: str) -> str:
    raw_mask = _clean(row.get(f"{concept}_observed_mask"))
    if raw_mask:
        return raw_mask
    if _clean(status).lower() in OBSERVED_STATUS_VALUES:
        return "1"
    return "0"


def _concept_is_observed(row: dict[str, str], concept: str) -> bool:
    status = _clean(row.get(f"{concept}_status")) or _clean(row.get(f"{concept}_raw_status"))
    canonical = canonicalize_concept_label(
        value=row.get(concept),
        status=status,
        observed_mask=_canonical_observed_mask_input(row, concept, status),
        source=row.get(f"{concept}_source"),
        confidence=row.get(f"{concept}_confidence"),
    )
    return canonical.observed_mask


def _concept_missing_status(row: dict[str, str], concept: str) -> str:
    status = _clean(row.get(f"{concept}_status")).lower()
    if status in {"raw_unconfirmed", "not_applicable"}:
        return status
    value = _clean(row.get(concept)).lower()
    if value == "not_applicable":
        return "not_applicable"
    raw_status = _clean(row.get(f"{concept}_raw_status")).lower()
    if raw_status == "raw_unconfirmed":
        return "raw_unconfirmed"
    return "missing"


def _concept_unobserved_source(status: str) -> str:
    if status in {"raw_unconfirmed", "not_applicable"}:
        return status
    return "missing"


def _benign_malignant_observed(row: dict[str, str]) -> tuple[bool, str]:
    missing_status = _concept_missing_status(row, "benign_malignant_label")
    canonical = canonicalize_concept_label(
        value=row.get("benign_malignant_label"),
        status=_clean(row.get("benign_malignant_label_status")) or missing_status,
        observed_mask=_canonical_observed_mask_input(
            row,
            "benign_malignant_label",
            _clean(row.get("benign_malignant_label_status")) or missing_status,
        ),
        source=row.get("benign_malignant_label_source"),
        confidence=row.get("benign_malignant_label_confidence"),
    )
    if canonical.observed_mask:
        explicit_source = _clean(row.get("benign_malignant_label_source"))
        return True, explicit_source or "structured_label"
    if canonical.status in {"raw_unconfirmed", "not_applicable"}:
        return False, _concept_unobserved_source(canonical.status)
    value = _clean(row.get("benign_malignant_label")).lower()
    if value in UNKNOWN_CONCEPT_VALUES:
        return False, _concept_unobserved_source(missing_status)

    from breast_pretrain.text.clinical_concepts import normalize_benign_malignant_label
    if normalize_benign_malignant_label(value):
        explicit_source = _clean(row.get("benign_malignant_label_source"))
        return True, explicit_source or "structured_label"

    status = _clean(row.get("benign_malignant_label_status")).lower()
    if status in OBSERVED_STATUS_VALUES:
        return True, "structured_label"

    if _concept_is_observed(row, "cancer_label"):
        return True, "inherited_from_cancer_label"

    return False, _concept_unobserved_source(missing_status)


def _mri_sequence_observed(row: dict[str, str]) -> tuple[bool, str]:
    modality = _clean(row.get("modality")).lower()
    if modality == "mammo":
        modality = "mammography"
    if modality == "us":
        modality = "ultrasound"
    if modality != "mri":
        return False, "not_applicable"

    from breast_pretrain.text.clinical_concepts import normalize_mri_sequence

    raw_value = _clean(row.get("mri_sequence"))
    canonical = normalize_mri_sequence(raw_value)
    if canonical:
        row["mri_sequence"] = canonical
        return True, _clean(row.get("mri_sequence_source")) or "structured_label"

    for field in ("series_description", "sequence", "mr_series"):
        alt = _clean(row.get(field))
        candidate = normalize_mri_sequence(alt)
        if candidate:
            row["mri_sequence"] = candidate
            return True, _clean(row.get("mri_sequence_source")) or f"inferred_from_{field}"

    return False, "missing"


def fill_concept_target_metadata(row: dict[str, str]) -> None:
    for concept in CONCEPT_TARGET_FIELDS:
        if concept == "benign_malignant_label":
            observed, default_source = _benign_malignant_observed(row)
        elif concept == "mri_sequence":
            observed, default_source = _mri_sequence_observed(row)
        else:
            observed = _concept_is_observed(row, concept)
            default_source = "structured_label" if observed else _concept_unobserved_source(
                _concept_missing_status(row, concept)
            )

        mask_key = f"{concept}_observed_mask"
        row[mask_key] = "1" if observed else "0"

        status_key = f"{concept}_status"
        if not _clean(row.get(status_key)):
            if observed:
                row[status_key] = "confirmed"
            elif default_source in {"not_applicable", "raw_unconfirmed"}:
                row[status_key] = default_source
            else:
                row[status_key] = _concept_missing_status(row, concept)

        source_key = f"{concept}_source"
        if not _clean(row.get(source_key)):
            row[source_key] = default_source

        confidence_key = f"{concept}_confidence"
        if not _clean(row.get(confidence_key)):
            row[confidence_key] = "1.0" if observed else "0.0"


def _key(image_id: Any, modality: Any) -> tuple[str, str]:
    return identity_key(image_id, modality)


def _manifest_modality(row: dict[str, str], fallback: str) -> str:
    return normalize_stage1_modality(row.get("modality") or fallback)


def _first_present(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = _clean(row.get(alias))
        if value:
            return value
    return ""


def _audit_field(row: dict[str, str], field_name: str) -> str:
    return _first_present(row, AUDITED_STATUS_FIELD_ALIASES[field_name])


def _validate_audited_row(row: dict[str, str], path: Path, row_number: int) -> dict[str, str]:
    values = {
        "prior_status": _audit_field(row, "prior_status"),
        "gaze_loss_enabled": _audit_field(row, "gaze_loss_enabled"),
        "aggregation_source": _audit_field(row, "aggregation_source"),
        "prior_qc_level": _audit_field(row, "prior_qc_level"),
        "audit_status": _audit_field(row, "audit_status"),
    }
    errors: list[str] = []
    if values["prior_status"] != EXPECTED_GAZE_PRIOR_STATUS:
        errors.append(f"prior_status={values['prior_status']!r}")
    if values["gaze_loss_enabled"] != EXPECTED_GAZE_LOSS_ENABLED:
        errors.append(f"gaze_loss_enabled={values['gaze_loss_enabled']!r}")
    if values["aggregation_source"] in FORBIDDEN_AUDITED_AGGREGATION_SOURCES:
        errors.append(f"aggregation_source={values['aggregation_source']!r} is diagnostic-only")
    if values["aggregation_source"] not in ALLOWED_AUDITED_AGGREGATION_SOURCES:
        errors.append(f"aggregation_source={values['aggregation_source']!r}")
    if values["prior_qc_level"] not in ALLOWED_AUDITED_PRIOR_QC_LEVELS:
        errors.append(f"prior_qc_level={values['prior_qc_level']!r}")
    if values["audit_status"] != EXPECTED_GAZE_PRIOR_AUDIT_STATUS:
        errors.append(f"audit_status={values['audit_status']!r}")
    if errors:
        image_id = _clean(row.get("image_id")) or "<missing image_id>"
        raise ValueError(
            f"Invalid audited usable prior row in {path} row {row_number} image_id={image_id}: "
            + ", ".join(errors)
        )
    return values


def _resolve_existing(raw_value: str, *, base_dir: Path, field_name: str, image_id: str) -> Path:
    value = _clean(raw_value)
    if not value:
        raise ValueError(f"Audited prior {image_id} has empty {field_name}.")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, _project_root() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Audited prior {image_id} missing {field_name}: {value}. Checked: {checked}")


def _portable_path(path: Path, output_root: Path) -> str:
    resolved = path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    try:
        return os.path.relpath(resolved, output).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _manifest_fieldnames(source_fieldnames: list[str]) -> list[str]:
    fieldnames = list(source_fieldnames)
    for column in OUTPUT_MANIFEST_EXTRA_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    for concept in CONCEPT_TARGET_FIELDS:
        for suffix in CONCEPT_TARGET_METADATA_SUFFIXES:
            column = f"{concept}_{suffix}"
            if column not in fieldnames:
                fieldnames.append(column)
    if "row_index" in fieldnames:
        fieldnames.remove("row_index")
        fieldnames.insert(0, "row_index")
    else:
        fieldnames.insert(0, "row_index")
    return fieldnames


def _read_audited_manifest(
    spec: AuditedManifestSpec,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    rows, _fieldnames = read_csv(spec.path)
    base_dir = spec.path.parent.resolve()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    validation_counts: dict[str, Any] = {
        "path": str(spec.path.expanduser().resolve()),
        "modality": normalize_stage1_modality(spec.modality),
        "row_count": len(rows),
        "validated_usable_prior_count": 0,
        "aggregation_source_counts": Counter(),
        "prior_qc_level_counts": Counter(),
        "audit_status_counts": Counter(),
    }
    for row_number, row in enumerate(rows, start=1):
        image_id = _clean(row.get("image_id"))
        if not image_id:
            raise ValueError(f"{spec.path} row {row_number} is missing image_id.")
        canonical_stage1_image_id = _clean(row.get("canonical_stage1_image_id"))
        audited_values = _validate_audited_row(row, spec.path, row_number)
        validation_counts["validated_usable_prior_count"] += 1
        validation_counts["aggregation_source_counts"].update([audited_values["aggregation_source"]])
        validation_counts["prior_qc_level_counts"].update([audited_values["prior_qc_level"]])
        validation_counts["audit_status_counts"].update([audited_values["audit_status"]])
        modality = _manifest_modality(row, spec.modality)
        key = (canonical_stage1_image_id or image_id, modality)
        if key in result:
            raise ValueError(f"Duplicate audited usable prior key in {spec.path}: {key}")
        resolved_paths = {
            target: _resolve_existing(
                _first_present(row, aliases),
                base_dir=base_dir,
                field_name=target,
                image_id=image_id,
            )
            for target, aliases in AUDITED_PATH_FIELD_ALIASES.items()
        }
        result[key] = {
            "row": row,
            "manifest_path": spec.path.resolve(),
            "manifest_row_number": row_number,
            "modality": modality,
            "audited_image_id": image_id,
            "canonical_stage1_image_id": canonical_stage1_image_id,
            "resolved_paths": resolved_paths,
            "audited_values": audited_values,
        }
    validation_counts["aggregation_source_counts"] = dict(sorted(validation_counts["aggregation_source_counts"].items()))
    validation_counts["prior_qc_level_counts"] = dict(sorted(validation_counts["prior_qc_level_counts"].items()))
    validation_counts["audit_status_counts"] = dict(sorted(validation_counts["audit_status_counts"].items()))
    return result, validation_counts


def _load_audited_manifests(specs: list[AuditedManifestSpec]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    audited: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_counts: list[dict[str, Any]] = []
    for spec in specs:
        manifest_records, counts = _read_audited_manifest(spec)
        manifest_counts.append(counts)
        for key, record in manifest_records.items():
            if key in audited:
                raise ValueError(f"Duplicate audited usable prior key across manifests: {key}")
            audited[key] = record
    return audited, {
        "audited_input_manifest_count": len(specs),
        "audited_input_row_count": sum(int(item["row_count"]) for item in manifest_counts),
        "audited_input_validated_usable_prior_count": sum(
            int(item["validated_usable_prior_count"]) for item in manifest_counts
        ),
        "audited_input_manifests": manifest_counts,
    }


def _selected_and_excluded_rows(
    source_rows: list[dict[str, str]],
    audited: dict[tuple[str, str], dict[str, Any]],
    source_manifest_dir: Path,
    output_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], set[tuple[str, str]], dict[str, Any]]:
    selected: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    linkage: list[dict[str, Any]] = []
    identity_result = resolve_stage1_identity_linkage(
        source_rows=source_rows,
        audited=audited,
        source_manifest_dir=source_manifest_dir,
    )
    matches_by_source_index = {match.source_index: match for match in identity_result.matches}

    for source_index, source_row in enumerate(source_rows):
        match = matches_by_source_index.get(source_index)
        if match is None:
            excluded_row = dict(source_row)
            excluded_row["exclusion_reason"] = "diagnostic_only_no_usable_audited_prior"
            excluded.append(excluded_row)
            continue

        key = match.source_key
        audited_record = match.audited_record
        audited_row = audited_record["row"]
        resolved_paths: dict[str, Path] = audited_record["resolved_paths"]
        row = dict(source_row)
        row["row_index"] = str(len(selected))
        row["image_id"] = match.canonical_stage1_image_id
        row["modality"] = key[1]
        row["canonical_stage1_image_id"] = match.canonical_stage1_image_id
        row["audited_prior_image_id"] = match.audited_prior_image_id
        row["stage0_inference_image_id"] = match.stage0_inference_image_id
        row["identity_linkage_method"] = match.method
        row["identity_linkage_resolved_image_path"] = match.resolved_source_image_path
        row["gaze_supervision_source"] = EXPECTED_GAZE_SUPERVISION_SOURCE
        row["gaze_prior_status"] = EXPECTED_GAZE_PRIOR_STATUS
        row["gaze_prior_audit_status"] = EXPECTED_GAZE_PRIOR_AUDIT_STATUS
        row["gaze_loss_enabled"] = EXPECTED_GAZE_LOSS_ENABLED
        for field_name, path in resolved_paths.items():
            row[field_name] = _portable_path(path, output_root)
        for field_name, aliases in AUDITED_METADATA_FIELD_ALIASES.items():
            value = _first_present(audited_row, aliases)
            if value:
                row[field_name] = value
        fill_concept_target_metadata(row)
        selected.append(row)
        linkage.append(
            {
                "source_row_index": source_index,
                "output_row_index": len(selected) - 1,
                "image_id": key[0],
                "modality": key[1],
                "canonical_stage1_image_id": match.canonical_stage1_image_id,
                "audited_prior_image_id": match.audited_prior_image_id,
                "linkage_method": match.method,
                "source_image_path": match.source_image_path,
                "audited_image_path": match.audited_image_path,
                "resolved_source_image_path": match.resolved_source_image_path,
                "resolved_audited_image_path": match.resolved_audited_image_path,
                "linkage_unique": int(match.linkage_unique),
                "audited_manifest_path": str(audited_record["manifest_path"]),
                "audited_manifest_row_number": audited_record["manifest_row_number"],
                "attention_map_path": row["attention_map_path"],
                "high_conf_mask_path": row["high_conf_mask_path"],
                "trajectory_qc_path": row["trajectory_qc_path"],
            }
        )
    return selected, excluded, linkage, identity_result.used_audited_keys, identity_result.summary


def _build_sparse_sidecar(output_root: Path, top_k: int) -> dict[str, Any]:
    scripts_root = _project_root() / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from build_stage1_sparse_semantic_soft_labels import build_sparse_semantic_soft_labels

    return build_sparse_semantic_soft_labels(
        manifest_path=output_root / MANIFEST_NAME,
        output_dir=output_root,
        top_k=top_k,
        allow_raw_unconfirmed=False,
        text_prompts_path=output_root / TEXT_PROMPTS_NAME,
    )


def _validate_counts(
    rows: list[dict[str, str]],
    expected_modality_counts: dict[str, int],
    audited: dict[tuple[str, str], dict[str, Any]],
    used_audited: set[tuple[str, str]],
) -> dict[str, Any]:
    modality_counts = Counter(normalize_stage1_modality(row.get("modality")) for row in rows)
    image_ids = [_clean(row.get("image_id")) for row in rows]
    duplicate_image_ids = len(image_ids) - len(set(image_ids))
    unused_audited = sorted(audited_key for audited_key in audited if audited_key not in used_audited)
    expected_total = sum(expected_modality_counts.values())
    errors: list[str] = []
    if len(rows) != expected_total:
        errors.append(f"selected row count={len(rows)} does not match expected {expected_total}")
    for modality, expected_count in expected_modality_counts.items():
        actual = int(modality_counts.get(modality, 0))
        if actual != int(expected_count):
            errors.append(f"{modality} count={actual} does not match expected {expected_count}")
    if duplicate_image_ids:
        errors.append(f"duplicate image_id count={duplicate_image_ids}; expected 0")
    if unused_audited:
        errors.append(f"unused audited usable prior count={len(unused_audited)}; expected 0")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "selected_row_count": len(rows),
        "modality_counts": dict(sorted(modality_counts.items())),
        "duplicate_image_id_count": duplicate_image_ids,
        "missing_audited_prior_count": 0,
        "unused_audited_usable_prior_count": len(unused_audited),
    }


def _validate_paths(rows: list[dict[str, str]], output_root: Path) -> dict[str, int]:
    missing: list[str] = []
    for index, row in enumerate(rows):
        for field_name in ("attention_map_path", "high_conf_mask_path", "trajectory_qc_path"):
            value = _clean(row.get(field_name))
            path = Path(value).expanduser()
            candidates = [path] if path.is_absolute() else [output_root / path, _project_root() / path]
            if not any(candidate.resolve().is_file() for candidate in candidates):
                missing.append(f"row {index} {field_name}={value}")
    if missing:
        raise FileNotFoundError("Missing audited prior path(s): " + "; ".join(missing[:20]))
    return {"missing_attention_mask_qc_path_count": 0}


def build_stage1_gaze_enabled_bundle_from_audited_priors(
    *,
    source_bundle_root: Path,
    audited_manifest_specs: list[AuditedManifestSpec],
    output_bundle_root: Path,
    overwrite: bool,
    expected_modality_counts: dict[str, int] | None = None,
    top_k: int = 20,
    allow_fixture_semantic_fallback: bool = False,
) -> dict[str, Any]:
    source_root = source_bundle_root.expanduser().resolve()
    output_root = output_bundle_root.expanduser().resolve()
    manifest_path = source_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source Stage 1 manifest: {manifest_path}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output bundle without --overwrite: {output_root}")
    if output_root.exists() and overwrite:
        clean_builder_managed_outputs(output_root)

    source_rows, source_fieldnames = read_csv(manifest_path)
    audited, audited_input_validation = _load_audited_manifests(audited_manifest_specs)
    selected_rows, excluded_rows, linkage_rows, used_audited, identity_linkage_summary = _selected_and_excluded_rows(
        source_rows,
        audited,
        manifest_path.parent,
        output_root,
    )
    expected_counts = expected_modality_counts or DEFAULT_EXPECTED_MODALITY_COUNTS
    count_summary = _validate_counts(selected_rows, expected_counts, audited, used_audited)
    path_summary = _validate_paths(selected_rows, output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_fieldnames = _manifest_fieldnames(source_fieldnames)
    excluded_fieldnames = list(source_fieldnames)
    if "exclusion_reason" not in excluded_fieldnames:
        excluded_fieldnames.append("exclusion_reason")
    write_csv(output_root / MANIFEST_NAME, manifest_fieldnames, selected_rows)
    write_csv(output_root / "manifest_excluded_diagnostic_only.csv", excluded_fieldnames, excluded_rows)
    write_csv(
        output_root / "gaze_enabled_bundle_linkage_audit.csv",
        [
            "source_row_index",
            "output_row_index",
            "image_id",
            "modality",
            "canonical_stage1_image_id",
            "audited_prior_image_id",
            "linkage_method",
            "source_image_path",
            "audited_image_path",
            "resolved_source_image_path",
            "resolved_audited_image_path",
            "linkage_unique",
            "audited_manifest_path",
            "audited_manifest_row_number",
            "attention_map_path",
            "high_conf_mask_path",
            "trajectory_qc_path",
        ],
        linkage_rows,
    )

    prompts, prompt_embedding_stats = filter_text_prompts_and_embeddings(
        source_root=source_root,
        output_root=output_root,
        rows=selected_rows,
        allow_fixture_semantic_fallback=allow_fixture_semantic_fallback,
    )
    prior_stats = filter_birads_prior_manifest_and_files(
        source_root=source_root,
        output_root=output_root,
        rows=selected_rows,
        allow_fixture_semantic_fallback=allow_fixture_semantic_fallback,
    )
    concept_stats = filter_case_concept_vectors(
        source_root=source_root,
        output_root=output_root,
        rows=selected_rows,
        allow_fixture_semantic_fallback=allow_fixture_semantic_fallback,
    )
    sparse_summary = _build_sparse_sidecar(output_root, top_k=top_k)

    summary = {
        "schema_version": "stage1_gaze_enabled_bundle_from_audited_priors_v1",
        "status": "pass",
        "source_bundle_root": str(source_root),
        "output_bundle_root": str(output_root),
        "audited_manifest_paths": [str(spec.path.expanduser().resolve()) for spec in audited_manifest_specs],
        "excluded_diagnostic_only_count": len(excluded_rows),
        "gaze_supervision_source": EXPECTED_GAZE_SUPERVISION_SOURCE,
        "gaze_prior_claim": "weak spatial attention prior, not real doctor gaze ground truth",
        "teacher_latent_mainline_enabled": False,
        "stage1_direct_graph_alignment_enabled": False,
        "sidecar_policy": "row_aligned_sidecars_regenerated_or_strictly_filtered",
        "semantic_soft_label_format": "sparse_topk",
        "allow_fixture_semantic_fallback": bool(allow_fixture_semantic_fallback),
        **audited_input_validation,
        **identity_linkage_summary,
        **prompt_embedding_stats,
        **prior_stats,
        **concept_stats,
        "sparse_semantic_summary": sparse_summary,
        **count_summary,
        **path_summary,
    }
    write_json(output_root / "gaze_enabled_bundle_build_summary.json", summary)
    return summary


__all__ = [
    "AuditedManifestSpec",
    "DEFAULT_EXPECTED_MODALITY_COUNTS",
    "build_stage1_gaze_enabled_bundle_from_audited_priors",
    "fill_concept_target_metadata",
    "read_csv",
    "write_csv",
    "write_json",
    "write_jsonl",
]
