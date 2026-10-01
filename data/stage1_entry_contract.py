from __future__ import annotations

# This legacy compatibility module is intentionally kept below the 1000-line
# split threshold.  V6 validation is isolated in stage1_v6_contract.py; the
# next legacy-only extraction targets prompt/embedding checks and registry
# readiness checks, without mixing either back into the V6 formal gate.
LEGACY_ENTRY_CONTRACT_LINE_COUNT_WARNING = (
    "stage1_entry_contract.py is a legacy compatibility boundary; keep it below "
    "1000 lines and keep all V6 formal checks in stage1_v6_contract.py"
)
LEGACY_ENTRY_CONTRACT_SPLIT_PLAN = (
    "extract legacy prompt_embedding validation when it changes independently",
    "extract registry readiness inspection when it changes independently",
    "do not reintroduce V6 checks into this legacy module",
)

import csv
import json
from pathlib import Path
from typing import Any

from breast_pretrain.data.semantic_soft_label_contract import validate_semantic_soft_labels
from breast_pretrain.data_registry.registry import DatasetRegistryEntry, load_dataset_registry
from breast_pretrain.text.clinical_concepts import (
    normalize_birads,
    normalize_density,
    normalize_laterality,
    normalize_view,
)


STAGE1_STANDARD_FILENAMES = {
    "image_manifest": "manifest_stage1_semantic.csv",
    "text_prompts": "text_prompts.jsonl",
    "prompt_embeddings": "stage1_prompt_embeddings.json",
    "semantic_soft_labels": "stage1_semantic_soft_labels.npy",
    "semantic_soft_labels_topk": "stage1_semantic_soft_labels_topk.npz",
    "semantic_soft_labels_topk_jsonl": "stage1_semantic_soft_labels_topk.jsonl",
    "semantic_soft_label_manifest": "stage1_semantic_soft_label_manifest.csv",
    "birads_prior_manifest": "stage1_birads_prior_manifest.csv",
    "v6_manifest": "manifest_stage1_v6.csv",
    "accepted_effective_reports": "accepted_effective_report_manifest.jsonl",
    "report_derived_graph_nodes": "report_derived_graph_nodes.jsonl",
    "final_rejected_cases": "final_rejected_case_manifest.jsonl",
    "discarded_images": "discarded_image_manifest.jsonl",
}

STAGE1_GAZE_SUPERVISION_SOURCES = (
    "observed_gaze",
    "diffeye_generated_gaze",
    "default_or_neutral_prior",
    "no_gaze",
)

_BASE_REQUIRED_MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "modality",
    "gaze_supervision_source",
)

_MAMMOGRAPHY_IDENTITY_COLUMNS = (
    "patient_id",
    "study_id",
    "laterality",
    "view",
)

_RECOMMENDED_MANIFEST_COLUMNS = (
    "breast_id",
    "density",
    "finding",
    "birads",
    "bbox_available",
    "bbox_path",
    "attention_map_path",
    "high_conf_mask_path",
    "patch_gaze_weight_path",
    "cancer_label",
    "benign_malignant_label",
    "study_description",
)

_PATH_COLUMNS = (
    "image_path",
    "bbox_path",
    "attention_map_path",
    "high_conf_mask_path",
    "patch_gaze_weight_path",
)

_MAMMOGRAPHY_MODALITIES = frozenset({"mammo", "mammography", "ffdm", "xray mammography"})
_RECOGNIZED_MODALITIES = _MAMMOGRAPHY_MODALITIES | {"mri", "ultrasound"}

_MODALITY_ALTERNATIVE_METADATA_FIELDS = (
    "sequence_type",
    "series_description",
    "series_id",
    "scan_plane",
    "acquisition_plane",
    "image_type",
    "protocol_name",
)


def _is_mammography_modality(modality: str) -> bool:
    return modality.lower() in _MAMMOGRAPHY_MODALITIES


_FINDING_OR_BBOX_COLUMNS = (
    "finding",
    "finding_categories",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
)

_FIELD_CANDIDATES = {
    "image_id": {
        "primary_manifest": ("image_id", "ImageID", "SOP Instance UID", "sop_uid"),
        "metadata_csv": ("SOP Instance UID", "image_id", "ImageID"),
        "breast_annotations": ("image_id", "ImageID"),
        "finding_annotations": ("image_id", "ImageID"),
    },
    "image_path": {
        "primary_manifest": ("image_path", "filepath", "file_path"),
    },
    "patient_id": {
        "primary_manifest": ("patient_id", "PatientID", "subject_id"),
        "metadata_csv": ("patient_id", "PatientID", "subject_id"),
    },
    "study_id": {
        "primary_manifest": ("study_id", "StudyInstanceUID", "exam_id"),
        "metadata_csv": ("study_id", "StudyInstanceUID", "exam_id"),
        "breast_annotations": ("study_id", "exam_id"),
    },
    "laterality": {
        "primary_manifest": ("laterality",),
        "metadata_csv": ("Image Laterality", "image_laterality", "Laterality", "laterality"),
        "breast_annotations": ("laterality", "Laterality"),
    },
    "view": {
        "primary_manifest": ("view", "view_position"),
        "metadata_csv": ("View Position", "view_position", "View", "view"),
        "breast_annotations": ("view_position", "view", "View"),
    },
    "density": {
        "metadata_csv": ("Density", "breast_density", "Breast Density"),
        "breast_annotations": ("breast_density", "density", "Density"),
    },
    "birads": {
        "breast_annotations": ("breast_birads", "birads", "BIRADS"),
        "finding_annotations": ("finding_birads", "BIRADS", "BI-RADS"),
    },
    "finding": {
        "primary_manifest": ("finding", "finding_categories"),
        "finding_annotations": _FINDING_OR_BBOX_COLUMNS,
    },
    "bbox": {
        "finding_annotations": ("xmin", "ymin", "xmax", "ymax"),
    },
}


def _status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "pass_with_warnings"
    return "pass"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_bool(value: Any) -> bool:
    return _clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def _is_explicit_unknown(value: Any) -> bool:
    return _clean_text(value).lower() in {"unknown", "unknown_not_provided"}


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(payload)
    return rows


def _resolve_path(raw_value: Any, base_dir: Path) -> Path | None:
    value = _clean_text(raw_value)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _warn_nonstandard_filename(
    actual_path: Path | None,
    expected_name: str,
    label: str,
    warnings: list[str],
) -> None:
    if actual_path is None:
        return
    if actual_path.name != expected_name:
        warnings.append(
            f"{label} uses non-standard filename '{actual_path.name}', expected '{expected_name}'."
        )


def _infer_legacy_gaze_source(row: dict[str, Any]) -> str:
    explicit = _clean_text(row.get("gaze_supervision_source")).lower()
    if explicit in STAGE1_GAZE_SUPERVISION_SOURCES:
        return explicit

    legacy_hints = " ".join(
        _clean_text(row.get(key)).lower()
        for key in (
            "prior_type",
            "prior_source",
            "prior_version",
            "gaze_source",
            "gaze_prior_status",
        )
    )
    if "observed" in legacy_hints or "eyetrack" in legacy_hints:
        return "observed_gaze"
    if "diffeye" in legacy_hints:
        return "diffeye_generated_gaze"
    if any(token in legacy_hints for token in ("center_prior", "random_prior", "neutral", "default")):
        return "default_or_neutral_prior"
    has_any_prior_path = any(_clean_text(row.get(column)) for column in _PATH_COLUMNS[2:])
    if "no_gaze" in legacy_hints:
        return "no_gaze"
    if has_any_prior_path:
        return "default_or_neutral_prior"
    return "no_gaze"


def _validate_manifest_rows(
    rows: list[dict[str, str]],
    manifest_path: Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    present_columns = set(rows[0].keys()) if rows else set()
    missing_required_columns = [
        column for column in _BASE_REQUIRED_MANIFEST_COLUMNS if column not in present_columns
    ]
    _has_mammography_rows = any(
        _is_mammography_modality(_clean_text(r.get("modality")).lower()) for r in rows
    )
    if _has_mammography_rows:
        missing_required_columns.extend(
            c for c in _MAMMOGRAPHY_IDENTITY_COLUMNS if c not in present_columns
        )
    missing_recommended_columns = [
        column for column in _RECOMMENDED_MANIFEST_COLUMNS if column not in present_columns
    ]
    if missing_required_columns:
        errors.append(
            "manifest is missing required Stage 1 columns: "
            + ", ".join(sorted(missing_required_columns))
        )
    if missing_recommended_columns:
        warnings.append(
            "manifest is missing recommended Stage 1 columns: "
            + ", ".join(sorted(missing_recommended_columns))
        )

    seen_image_ids: set[str] = set()
    duplicate_image_ids: list[str] = []
    explicit_gaze_count = 0
    legacy_inferred_gaze_count = 0
    gaze_source_counts = {key: 0 for key in STAGE1_GAZE_SUPERVISION_SOURCES}

    for row_index, row in enumerate(rows, start=1):
        image_id = _clean_text(row.get("image_id"))
        if image_id:
            if image_id in seen_image_ids:
                duplicate_image_ids.append(image_id)
            seen_image_ids.add(image_id)

        for column in ("image_id", "image_path", "modality"):
            if not _clean_text(row.get(column)):
                errors.append(f"row {row_index} is missing required value: {column}")

        modality = _clean_text(row.get("modality")).lower()
        if modality and modality not in _RECOGNIZED_MODALITIES:
            warnings.append(
                f"row {row_index} uses unrecognized modality='{modality}'. "
                "Stage 1 contract supports mammography, MRI, and ultrasound."
            )

        if _is_mammography_modality(modality):
            for column in ("patient_id", "study_id", "laterality", "view"):
                if not _clean_text(row.get(column)):
                    errors.append(f"row {row_index} is missing required mammography identity value: {column}")
        else:
            for column in ("patient_id", "study_id"):
                if not _clean_text(row.get(column)):
                    warnings.append(
                        f"row {row_index}: '{column}' is missing; "
                        f"adapters should supply this when available (modality={modality})."
                    )
            for column in ("laterality", "view"):
                if not _clean_text(row.get(column)):
                    alt_fields = [
                        f for f in _MODALITY_ALTERNATIVE_METADATA_FIELDS
                        if _clean_text(row.get(f))
                    ]
                    hint = f" modality-specific metadata present: {alt_fields}" if alt_fields else ""
                    warnings.append(
                        f"row {row_index}: '{column}' is missing for modality='{modality}'.{hint}"
                    )

        if _is_mammography_modality(modality):
            laterality = normalize_laterality(row.get("laterality"))
            if (
                _clean_text(row.get("laterality"))
                and not laterality
                and not _is_explicit_unknown(row.get("laterality"))
            ):
                errors.append(
                    f"row {row_index} has unsupported laterality value: {row.get('laterality')!r}"
                )

            view, _ = normalize_view(row.get("view"))
            if _clean_text(row.get("view")) and not view and not _is_explicit_unknown(row.get("view")):
                errors.append(f"row {row_index} has unsupported view value: {row.get('view')!r}")

        density = _clean_text(row.get("density"))
        if density and not normalize_density(density):
            warnings.append(f"row {row_index} has non-canonical density value: {density!r}")

        birads = _clean_text(row.get("birads"))
        if birads and not normalize_birads(birads):
            warnings.append(f"row {row_index} has non-canonical BI-RADS value: {birads!r}")

        explicit_gaze_source = _clean_text(row.get("gaze_supervision_source")).lower()
        inferred_gaze_source = _infer_legacy_gaze_source(row)
        if explicit_gaze_source:
            if explicit_gaze_source not in STAGE1_GAZE_SUPERVISION_SOURCES:
                errors.append(
                    "row %d has unsupported gaze_supervision_source %r. "
                    "Expected one of: %s."
                    % (
                        row_index,
                        row.get("gaze_supervision_source"),
                        ", ".join(STAGE1_GAZE_SUPERVISION_SOURCES),
                    )
                )
                gaze_source = ""
            else:
                gaze_source = explicit_gaze_source
                explicit_gaze_count += 1
        else:
            gaze_source = inferred_gaze_source
            legacy_inferred_gaze_count += 1
            errors.append(
                f"row {row_index} is missing explicit gaze_supervision_source; inferred legacy value='{gaze_source}'."
            )

        if gaze_source in gaze_source_counts:
            gaze_source_counts[gaze_source] += 1

        prior_path_count = sum(
            1 for column in ("attention_map_path", "high_conf_mask_path", "patch_gaze_weight_path")
            if _clean_text(row.get(column))
        )
        if gaze_source in {"observed_gaze", "diffeye_generated_gaze"} and prior_path_count == 0:
            errors.append(
                f"row {row_index} declares {gaze_source} but provides no prior path columns."
            )
        if gaze_source == "no_gaze" and prior_path_count > 0:
            warnings.append(
                f"row {row_index} declares no_gaze but still provides prior path columns."
            )

        bbox_available = _normalize_bool(row.get("bbox_available"))
        bbox_path = _clean_text(row.get("bbox_path"))
        if bbox_available and not bbox_path:
            warnings.append(
                f"row {row_index} marks bbox_available=true but bbox_path is empty."
            )

        for column in _PATH_COLUMNS:
            resolved_path = _resolve_path(row.get(column), manifest_path.parent)
            if column == "image_path":
                if resolved_path is None or not resolved_path.exists():
                    errors.append(
                        f"row {row_index} references missing image_path: {row.get(column)!r}"
                    )
            elif resolved_path is not None and not resolved_path.exists():
                warnings.append(
                    f"row {row_index} references missing optional path {column}: {row.get(column)!r}"
                )

    if duplicate_image_ids:
        errors.append(
            "manifest contains duplicate image_id values: "
            + ", ".join(sorted(set(duplicate_image_ids))[:20])
        )

    summary = {
        "row_count": len(rows),
        "present_columns": sorted(present_columns),
        "missing_required_columns": missing_required_columns,
        "missing_recommended_columns": missing_recommended_columns,
        "explicit_gaze_source_count": explicit_gaze_count,
        "legacy_inferred_gaze_source_count": legacy_inferred_gaze_count,
        "gaze_source_counts": gaze_source_counts,
    }
    return errors, warnings, summary


def _validate_text_prompts(
    manifest_rows: list[dict[str, str]],
    text_prompt_path: Path | None,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prompt_lookup: dict[str, str] = {}
    if text_prompt_path is None or not text_prompt_path.exists():
        errors.append("text_prompts.jsonl is missing; V6 formal entry does not permit a replacement prompt.")
        return errors, warnings, {"exists": False, "row_count": 0}, prompt_lookup

    rows = _read_jsonl_rows(text_prompt_path)
    duplicate_image_ids: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        image_id = _clean_text(row.get("image_id"))
        text_prompt = _clean_text(row.get("text_prompt"))
        if not image_id or not text_prompt:
            errors.append(
                f"text_prompts row {row_index} must include non-empty image_id and text_prompt."
            )
            continue
        if image_id in prompt_lookup:
            duplicate_image_ids.append(image_id)
        prompt_lookup[image_id] = text_prompt

    if duplicate_image_ids:
        errors.append(
            "text_prompts.jsonl contains duplicate image_id values: "
            + ", ".join(sorted(set(duplicate_image_ids))[:20])
        )

    manifest_image_ids = {
        _clean_text(row.get("image_id")) for row in manifest_rows if _clean_text(row.get("image_id"))
    }
    missing_prompt_image_ids = sorted(manifest_image_ids - set(prompt_lookup))
    if missing_prompt_image_ids:
        errors.append(
            "text_prompts.jsonl is missing Effective Reports for %d manifest rows."
            % len(missing_prompt_image_ids)
        )

    summary = {
        "exists": True,
        "row_count": len(rows),
        "missing_prompt_image_id_count": len(missing_prompt_image_ids),
        "missing_prompt_image_ids_preview": missing_prompt_image_ids[:20],
    }
    return errors, warnings, summary, prompt_lookup


def _validate_prompt_embeddings(
    prompt_lookup: dict[str, str],
    prompt_embedding_path: Path | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    if prompt_embedding_path is None or not prompt_embedding_path.exists():
        errors.append("stage1_prompt_embeddings.json is missing.")
        return errors, warnings, {"exists": False, "prompt_embedding_count": 0}

    payload = json.loads(prompt_embedding_path.read_text(encoding="utf-8"))
    prompt_embeddings = payload.get("prompt_embeddings")
    if not isinstance(prompt_embeddings, dict):
        errors.append("stage1_prompt_embeddings.json must contain a prompt_embeddings mapping.")
        return errors, warnings, {"exists": True, "prompt_embedding_count": 0}

    missing_prompt_count = 0
    for prompt in sorted(set(prompt_lookup.values())):
        if prompt not in prompt_embeddings:
            missing_prompt_count += 1
    if missing_prompt_count:
        errors.append(
            f"stage1_prompt_embeddings.json is missing {missing_prompt_count} prompt embedding entries."
        )

    text_dim = payload.get("text_dim")
    if text_dim is None:
        warnings.append("stage1_prompt_embeddings.json does not record text_dim.")

    summary = {
        "exists": True,
        "prompt_embedding_count": len(prompt_embeddings),
        "text_dim": text_dim,
        "missing_prompt_count": missing_prompt_count,
    }
    return errors, warnings, summary


def _validate_birads_prior_manifest(
    manifest_rows: list[dict[str, str]],
    birads_prior_manifest_path: Path | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    if birads_prior_manifest_path is None or not birads_prior_manifest_path.exists():
        errors.append("stage1_birads_prior_manifest.csv is missing.")
        return errors, warnings, {"exists": False, "row_count": 0}

    rows, fieldnames = _read_csv_rows(birads_prior_manifest_path)
    if not {"image_id", "prior_path"}.issubset(fieldnames):
        errors.append(
            "stage1_birads_prior_manifest.csv must include image_id and prior_path columns."
        )

    manifest_dir = birads_prior_manifest_path.parent
    covered_image_ids: set[str] = set()
    missing_prior_file_count = 0
    invalid_schema_count = 0
    for row in rows:
        image_id = _clean_text(row.get("image_id"))
        prior_path = _resolve_path(row.get("prior_path"), manifest_dir)
        if not image_id:
            warnings.append("stage1_birads_prior_manifest.csv contains a row with empty image_id.")
            continue
        covered_image_ids.add(image_id)
        if prior_path is None or not prior_path.exists():
            missing_prior_file_count += 1
            continue
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        if _clean_text(payload.get("schema_version")) != "stage1_birads_prior_v1":
            invalid_schema_count += 1

    manifest_image_ids = {
        _clean_text(row.get("image_id")) for row in manifest_rows if _clean_text(row.get("image_id"))
    }
    uncovered_image_ids = sorted(manifest_image_ids - covered_image_ids)
    if uncovered_image_ids:
        warnings.append(
            "stage1_birads_prior_manifest.csv does not cover %d manifest rows."
            % len(uncovered_image_ids)
        )
    if missing_prior_file_count:
        warnings.append(
            f"stage1_birads_prior_manifest.csv references {missing_prior_file_count} missing prior files."
        )
    if invalid_schema_count:
        errors.append(
            f"{invalid_schema_count} BI-RADS prior JSON files use an unsupported schema_version."
        )

    summary = {
        "exists": True,
        "row_count": len(rows),
        "missing_prior_file_count": missing_prior_file_count,
        "invalid_schema_count": invalid_schema_count,
        "uncovered_image_id_count": len(uncovered_image_ids),
        "uncovered_image_ids_preview": uncovered_image_ids[:20],
    }
    return errors, warnings, summary


def validate_stage1_manifest_bundle(
    manifest_path: str | Path,
    *,
    text_prompt_path: str | Path | None = None,
    prompt_embedding_path: str | Path | None = None,
    semantic_soft_label_path: str | Path | None = None,
    semantic_manifest_path: str | Path | None = None,
    semantic_soft_label_format: str = "dense",
    semantic_soft_label_topk_path: str | Path | None = None,
    birads_prior_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_text_prompt_path = (
        Path(text_prompt_path).expanduser().resolve() if text_prompt_path is not None else None
    )
    resolved_prompt_embedding_path = (
        Path(prompt_embedding_path).expanduser().resolve()
        if prompt_embedding_path is not None
        else None
    )
    resolved_semantic_soft_label_path = (
        Path(semantic_soft_label_path).expanduser().resolve()
        if semantic_soft_label_path is not None
        else None
    )
    resolved_semantic_manifest_path = (
        Path(semantic_manifest_path).expanduser().resolve()
        if semantic_manifest_path is not None
        else None
    )
    resolved_semantic_soft_label_topk_path = (
        Path(semantic_soft_label_topk_path).expanduser().resolve()
        if semantic_soft_label_topk_path is not None
        else None
    )
    resolved_birads_prior_manifest_path = (
        Path(birads_prior_manifest_path).expanduser().resolve()
        if birads_prior_manifest_path is not None
        else None
    )

    errors: list[str] = []
    warnings: list[str] = []
    sections: dict[str, Any] = {}

    _warn_nonstandard_filename(
        resolved_manifest_path,
        STAGE1_STANDARD_FILENAMES["image_manifest"],
        "manifest",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_text_prompt_path,
        STAGE1_STANDARD_FILENAMES["text_prompts"],
        "text prompt file",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_prompt_embedding_path,
        STAGE1_STANDARD_FILENAMES["prompt_embeddings"],
        "prompt embedding file",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_semantic_soft_label_path,
        STAGE1_STANDARD_FILENAMES["semantic_soft_labels"],
        "semantic soft-label matrix",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_semantic_soft_label_topk_path,
        STAGE1_STANDARD_FILENAMES["semantic_soft_labels_topk"],
        "semantic soft-label top-k",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_semantic_manifest_path,
        STAGE1_STANDARD_FILENAMES["semantic_soft_label_manifest"],
        "semantic soft-label manifest",
        warnings,
    )
    _warn_nonstandard_filename(
        resolved_birads_prior_manifest_path,
        STAGE1_STANDARD_FILENAMES["birads_prior_manifest"],
        "BI-RADS prior manifest",
        warnings,
    )

    if not resolved_manifest_path.exists():
        errors.append(f"manifest_stage1_semantic.csv is missing: {resolved_manifest_path}")
        return {
            "status": _status(errors, warnings),
            "errors": errors,
            "warnings": warnings,
            "paths": {
                "manifest_path": str(resolved_manifest_path),
                "text_prompt_path": str(resolved_text_prompt_path) if resolved_text_prompt_path else None,
                "prompt_embedding_path": str(resolved_prompt_embedding_path)
                if resolved_prompt_embedding_path
                else None,
                "semantic_soft_label_path": str(resolved_semantic_soft_label_path)
                if resolved_semantic_soft_label_path
                else None,
                "semantic_manifest_path": str(resolved_semantic_manifest_path)
                if resolved_semantic_manifest_path
                else None,
                "semantic_soft_label_topk_path": str(resolved_semantic_soft_label_topk_path)
                if resolved_semantic_soft_label_topk_path
                else None,
                "birads_prior_manifest_path": str(resolved_birads_prior_manifest_path)
                if resolved_birads_prior_manifest_path
                else None,
            },
            "sections": sections,
        }

    manifest_rows, _ = _read_csv_rows(resolved_manifest_path)
    manifest_errors, manifest_warnings, manifest_summary = _validate_manifest_rows(
        manifest_rows,
        resolved_manifest_path,
    )
    errors.extend(manifest_errors)
    warnings.extend(manifest_warnings)
    sections["manifest"] = manifest_summary

    prompt_errors, prompt_warnings, prompt_summary, prompt_lookup = _validate_text_prompts(
        manifest_rows,
        resolved_text_prompt_path,
    )
    errors.extend(prompt_errors)
    warnings.extend(prompt_warnings)
    sections["text_prompts"] = prompt_summary

    embedding_errors, embedding_warnings, embedding_summary = _validate_prompt_embeddings(
        prompt_lookup,
        resolved_prompt_embedding_path,
    )
    errors.extend(embedding_errors)
    warnings.extend(embedding_warnings)
    sections["prompt_embeddings"] = embedding_summary

    semantic_errors, semantic_warnings, semantic_summary = validate_semantic_soft_labels(
        manifest_rows,
        resolved_semantic_soft_label_path,
        resolved_semantic_manifest_path,
        semantic_soft_label_format=semantic_soft_label_format,
        semantic_soft_label_topk_path=resolved_semantic_soft_label_topk_path,
    )
    errors.extend(semantic_errors)
    warnings.extend(semantic_warnings)
    sections["semantic_soft_labels"] = semantic_summary

    birads_errors, birads_warnings, birads_summary = _validate_birads_prior_manifest(
        manifest_rows,
        resolved_birads_prior_manifest_path,
    )
    errors.extend(birads_errors)
    warnings.extend(birads_warnings)
    sections["birads_priors"] = birads_summary

    return {
        "status": _status(errors, warnings),
        "errors": errors,
        "warnings": warnings,
        "paths": {
            "manifest_path": str(resolved_manifest_path),
            "text_prompt_path": str(resolved_text_prompt_path) if resolved_text_prompt_path else None,
            "prompt_embedding_path": str(resolved_prompt_embedding_path)
            if resolved_prompt_embedding_path
            else None,
            "semantic_soft_label_path": str(resolved_semantic_soft_label_path)
            if resolved_semantic_soft_label_path
            else None,
            "semantic_manifest_path": str(resolved_semantic_manifest_path)
            if resolved_semantic_manifest_path
            else None,
            "semantic_soft_label_topk_path": str(resolved_semantic_soft_label_topk_path)
            if resolved_semantic_soft_label_topk_path
            else None,
            "birads_prior_manifest_path": str(resolved_birads_prior_manifest_path)
            if resolved_birads_prior_manifest_path
            else None,
        },
        "sections": sections,
    }


def _find_entry_by_name(
    registry_config_path: Path,
    dataset_name: str,
) -> DatasetRegistryEntry:
    entries = load_dataset_registry(registry_config_path)
    for entry in entries:
        if entry.name == dataset_name:
            return entry
    raise KeyError(f"Dataset not found in registry: {dataset_name}")


def _source_headers(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        _, headers = _read_csv_rows(path)
        return headers
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl_rows(path)
        if not rows:
            return []
        keys: set[str] = set()
        for row in rows[:20]:
            keys.update(str(key) for key in row.keys())
        return sorted(keys)
    return []


def _coverage_from_headers(headers_by_source: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for field_name, by_source in _FIELD_CANDIDATES.items():
        covered_sources: list[str] = []
        matched_headers: list[str] = []
        for source_key, candidates in by_source.items():
            headers = headers_by_source.get(source_key, [])
            for header in candidates:
                if header in headers:
                    covered_sources.append(source_key)
                    matched_headers.append(header)
                    break
        coverage[field_name] = {
            "covered": bool(covered_sources),
            "sources": covered_sources,
            "matched_headers": matched_headers,
        }
    return coverage


def inspect_stage1_dataset_entry_readiness(
    *,
    registry_config_path: str | Path,
    dataset_name: str,
) -> dict[str, Any]:
    resolved_registry_config_path = Path(registry_config_path).expanduser().resolve()
    entry = _find_entry_by_name(resolved_registry_config_path, dataset_name)

    errors: list[str] = []
    warnings: list[str] = []
    is_mammo = _is_mammography_modality(entry.modality)
    source_requirements = [
        ("primary_manifest", True, "minimum image-level entrypoint"),
        ("metadata_csv", is_mammo, "view/laterality image metadata"),
        ("breast_annotations", False, "breast-level BI-RADS and density"),
        ("finding_annotations", False, "finding and bbox annotations"),
        ("text_prompt_jsonl", False, "optional prompt source for preprocessing"),
        ("gaze_manifest", False, "optional gaze prior source"),
    ]

    source_presence: list[dict[str, Any]] = []
    headers_by_source: dict[str, list[str]] = {}
    for source_key, required, purpose in source_requirements:
        raw_path = entry.source.get(source_key)
        path = raw_path if isinstance(raw_path, Path) else None
        exists = bool(path is not None and path.exists())
        source_presence.append(
            {
                "source_key": source_key,
                "required": required,
                "purpose": purpose,
                "path": str(path) if path is not None else "",
                "exists": exists,
            }
        )
        headers_by_source[source_key] = _source_headers(path)
        if required and not exists:
            errors.append(f"{dataset_name}: missing required source {source_key}: {path}")
        elif not exists and path is not None:
            warnings.append(f"{dataset_name}: optional source is missing: {source_key}: {path}")

    coverage = _coverage_from_headers(headers_by_source)
    _critical_identity_fields = ["image_id", "image_path", "patient_id", "study_id"]
    if _is_mammography_modality(entry.modality):
        _critical_identity_fields.extend(["laterality", "view"])
    missing_critical_fields = [
        field_name
        for field_name in _critical_identity_fields
        if not coverage[field_name]["covered"]
    ]
    if missing_critical_fields:
        errors.append(
            f"{dataset_name}: raw sources cannot yet materialize critical Stage 1 fields: "
            + ", ".join(missing_critical_fields)
        )

    for field_name in ("density", "birads", "finding"):
        if not coverage[field_name]["covered"]:
            warnings.append(
                f"{dataset_name}: raw sources are missing semantic enrichment field: {field_name}"
            )
    if not coverage["bbox"]["covered"]:
        warnings.append(
            f"{dataset_name}: no bbox coordinate headers detected; localization metadata is unavailable."
        )

    dataset_root_exists = entry.dataset_root.exists()
    if not dataset_root_exists:
        errors.append(f"{dataset_name}: dataset_root does not exist: {entry.dataset_root}")

    recommended_bundle_paths = {
        name: str((entry.dataset_root / file_name).resolve())
        for name, file_name in STAGE1_STANDARD_FILENAMES.items()
    }
    existing_bundle_files = {
        name: Path(path).exists() for name, path in recommended_bundle_paths.items()
    }

    if entry.name == "RSNA Breast Cancer Detection":
        warnings.append(
            "RSNA is limited to metadata/readiness inspection here; do not connect it to Stage 1 training until DICOM ingestion is complete."
        )

    return {
        "status": _status(errors, warnings),
        "dataset_name": entry.name,
        "adapter": entry.adapter,
        "dataset_root": str(entry.dataset_root),
        "dataset_root_exists": dataset_root_exists,
        "source_presence": source_presence,
        "source_headers": headers_by_source,
        "field_coverage": coverage,
        "missing_critical_fields": missing_critical_fields,
        "recommended_stage1_bundle_paths": recommended_bundle_paths,
        "existing_stage1_bundle_files": existing_bundle_files,
        "errors": errors,
        "warnings": warnings,
    }
