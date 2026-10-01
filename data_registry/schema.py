from __future__ import annotations

from pathlib import Path
from typing import Any


UNKNOWN = "unknown"
MISSING_TEXT_VALUES = {"", "na", "nan", "none", "null", "n/a", "unknown", "missing"}
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off", ""}

MANIFEST_FIELDS = [
    "global_sample_id",
    "dataset_name",
    "modality",
    "sub_modality",
    "data_role",
    "patient_id",
    "study_id",
    "exam_id",
    "series_id",
    "image_id",
    "breast_id",
    "image_path",
    "mask_path",
    "bbox_path",
    "report_path",
    "metadata_path",
    "split",
    "official_split",
    "custom_split",
    "view",
    "laterality",
    "projection",
    "image_type",
    "age",
    "sex",
    "birads",
    "density",
    "finding",
    "pathology",
    "cancer_label",
    "benign_malignant_label",
    "molecular_subtype",
    "bbox_available",
    "mask_available",
    "segmentation_available",
    "report_available",
    "structured_prompt_available",
    "clinical_concept_available",
    "gaze_available",
    "gaze_source",
    "gaze_heatmap_path",
    "high_conf_mask_path",
    "patch_gaze_weight_path",
    "gaze_prior_status",
    "gaze_prior_qc_pass",
    "gaze_inside_ratio",
    "gaze_coverage_ratio",
    "teacher_latent_available",
    "teacher_latent_path",
    "teacher_model_name",
    "pretrain_eligible",
    "downstream_eligible",
    "localization_eligible",
    "segmentation_eligible",
    "classification_eligible",
    "retrieval_eligible",
    "report_generation_eligible",
    "exclude_reason",
    "quality_status",
    "notes",
]

EXTRA_MANIFEST_FIELDS = [
    "assigned_split",
    "leakage_risk",
    "split_source",
    "split_unit",
    "source_manifest_path",
]

ALL_MANIFEST_FIELDS = MANIFEST_FIELDS + EXTRA_MANIFEST_FIELDS

BOOLEAN_FIELDS = {
    "bbox_available",
    "mask_available",
    "segmentation_available",
    "report_available",
    "structured_prompt_available",
    "clinical_concept_available",
    "gaze_available",
    "gaze_prior_qc_pass",
    "teacher_latent_available",
    "pretrain_eligible",
    "downstream_eligible",
    "localization_eligible",
    "segmentation_eligible",
    "classification_eligible",
    "retrieval_eligible",
    "report_generation_eligible",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_missing(value: Any) -> bool:
    return clean_text(value).lower() in MISSING_TEXT_VALUES


def normalize_text(value: Any, default: str = "") -> str:
    text = clean_text(value)
    if not text:
        return default
    return "" if text.lower() in {"none", "null", "nan"} else text


def normalize_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = clean_text(value).lower()
    if text in TRUE_VALUES:
        return "true"
    if text in FALSE_VALUES:
        return "false"
    return "false"


def normalize_number(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        if any(character in text for character in (".", "e", "E")):
            return str(float(text))
        return str(int(text))
    except ValueError:
        return text


def normalize_path(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value.resolve())
    text = clean_text(value)
    return text


def build_global_sample_id(row: dict[str, Any], row_index: int = 0) -> str:
    dataset_name = normalize_text(row.get("dataset_name"), default="dataset")
    for field_name in ("image_id", "series_id", "exam_id", "study_id", "patient_id"):
        value = normalize_text(row.get(field_name))
        if value:
            return f"{dataset_name}::{value}"
    return f"{dataset_name}::row_{row_index}"


def empty_manifest_row() -> dict[str, str]:
    return {field_name: "" for field_name in ALL_MANIFEST_FIELDS}


def ensure_manifest_row(
    row: dict[str, Any],
    row_index: int = 0,
) -> dict[str, str]:
    normalized = empty_manifest_row()
    for field_name in ALL_MANIFEST_FIELDS:
        raw_value = row.get(field_name, "")
        if field_name in BOOLEAN_FIELDS:
            normalized[field_name] = normalize_bool(raw_value)
        elif field_name.endswith("_path") or field_name == "source_manifest_path":
            normalized[field_name] = normalize_path(raw_value)
        elif field_name in {"gaze_inside_ratio", "gaze_coverage_ratio"}:
            normalized[field_name] = normalize_number(raw_value)
        else:
            normalized[field_name] = normalize_text(raw_value)

    if not normalized["global_sample_id"]:
        normalized["global_sample_id"] = build_global_sample_id(normalized, row_index=row_index)
    if not normalized["split"] and normalized["official_split"]:
        normalized["split"] = normalized["official_split"]
    if not normalized["quality_status"]:
        normalized["quality_status"] = "ok"
    if not normalized["leakage_risk"]:
        normalized["leakage_risk"] = "unknown"
    if not normalized["split_source"]:
        normalized["split_source"] = "unassigned"
    if not normalized["split_unit"]:
        normalized["split_unit"] = "unassigned"
    return normalized
