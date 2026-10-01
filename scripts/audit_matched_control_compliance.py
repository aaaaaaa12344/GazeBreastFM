from __future__ import annotations

"""M2/M3 matched-control and MRI/US compliance checks extracted from
audit_final_model_compliance.py to keep that file under the size threshold."""

import csv
import re
from pathlib import Path
from typing import Any

from breast_pretrain.configs import load_stage1_joint_pretrain_bundle  # noqa: E402

GAZE_RELATED_FIELDS = {
    "gaze_supervision_source",
    "attention_map_path",
    "high_conf_mask_path",
    "patch_gaze_weight_path",
    "prior_status",
    "audit_status",
    "trajectory_consensus_summary_path",
    "gaze_prior_qc_metrics_path",
}
MRI_REJECT_TERMS = (
    "scout",
    "locator",
    "localizer",
    "survey",
    "calibration",
    "segmentation",
    "mask",
    "label",
    "adc-only",
    "adc_only",
    "non-diagnostic",
    "non_diagnostic",
    "non-breast",
    "non_breast",
    "pelvic",
)
MRI_REJECT_FIELDS = (
    "diagnostic_sequence_status",
    "rejected_reason",
    "sequence",
    "sequence_type",
    "sequence_or_view",
    "series_description",
    "protocol_name",
    "study_description",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mri_reject_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace("_", "[-_ ]").replace("\\-", "[-_ ]")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


_MRI_REJECT_PATTERNS = tuple(_mri_reject_pattern(term) for term in MRI_REJECT_TERMS)
MRI_ACCEPTED_STATUS_VALUES = {"", "diagnostic", "accepted", "included", "pass", "usable", "confirmed"}
MRI_REJECTED_STATUS_VALUES = {"rejected", "reject", "non_diagnostic", "non-diagnostic", "not_diagnostic"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(raw_value: Any, base_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path_obj = Path(text).expanduser()
    if not path_obj.is_absolute():
        path_obj = (base_dir / path_obj).resolve()
    return path_obj


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _mri_row_has_rejected_sequence(row: dict[str, str]) -> bool:
    status = str(row.get("diagnostic_sequence_status", "")).strip().lower()
    if status in MRI_REJECTED_STATUS_VALUES:
        return True
    rejected_reason = str(row.get("rejected_reason", "")).strip().lower()
    if rejected_reason and rejected_reason not in {"none", "n/a", "na", "not_applicable"}:
        return True
    if status and status not in MRI_ACCEPTED_STATUS_VALUES:
        return True

    text = " ".join(
        str(row.get(key, "")).lower()
        for key in MRI_REJECT_FIELDS
        if key not in {"diagnostic_sequence_status", "rejected_reason"}
    )
    return any(pattern.search(text) for pattern in _MRI_REJECT_PATTERNS)


def audit_m2_m3_match(
    raw_config: dict[str, Any],
    config_path: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    model_role = str(raw_config.get("model_role", raw_config.get("metadata", {}).get("model_role", ""))).lower()
    is_m3 = "m3" in model_role or config_path.name.lower().startswith("m3_")
    if not is_m3:
        _add(checks, "m3_has_m2_clean_matched_control", "pass", "Not applicable for non-M3 config.", "info")
        return None

    comparison = raw_config.get("matched_control")
    if not isinstance(comparison, dict):
        _add(checks, "m3_has_m2_clean_matched_control", "warn", "No matched_control block present.", "warn")
        return None
    project_root = _resolve(raw_config.get("project_root", PROJECT_ROOT), config_path.parent) or PROJECT_ROOT
    # Relative matched_control.config paths are project_root-relative, matching other config path fields.
    control_path = _resolve(comparison.get("config"), project_root)
    if control_path is None or not control_path.exists():
        _add(checks, "m3_has_m2_clean_matched_control", "fail", f"Matched M2-clean config missing: {control_path}", "hard")
        return None

    m3 = load_stage1_joint_pretrain_bundle(config_path)
    m2 = load_stage1_joint_pretrain_bundle(control_path)
    m3_rows = _rows(m3.trainer.data.image_manifest_path)
    m2_rows = _rows(m2.trainer.data.image_manifest_path)
    details: dict[str, Any] = {
        "m2_config": str(control_path),
        "m2_manifest": str(m2.trainer.data.image_manifest_path),
        "m3_manifest": str(m3.trainer.data.image_manifest_path),
        "row_count_m2": len(m2_rows),
        "row_count_m3": len(m3_rows),
    }
    m2_ids = [row.get("image_id", "") for row in m2_rows]
    m3_ids = [row.get("image_id", "") for row in m3_rows]
    if m2_ids != m3_ids:
        _add(checks, "m2_m3_image_id_order_match", "fail", "M2-clean and M3 image_id row order differs.", "hard")
        return details

    non_gaze_diffs = []
    for left, right in zip(m2_rows, m3_rows):
        keys = sorted((set(left) | set(right)) - GAZE_RELATED_FIELDS)
        for key in keys:
            if str(left.get(key, "")) != str(right.get(key, "")):
                non_gaze_diffs.append({"image_id": left.get("image_id", ""), "field": key})
                break
    if non_gaze_diffs:
        _add(checks, "m2_m3_only_gaze_fields_differ", "fail", f"Non-gaze field differences found: {non_gaze_diffs[:10]}", "hard")
    else:
        _add(checks, "m2_m3_only_gaze_fields_differ", "pass", "Only gaze-related manifest fields differ.", "info")

    m2_semantic = list(csv.DictReader(m2.trainer.semantic.semantic_manifest_path.open("r", encoding="utf-8-sig", newline="")))
    m3_semantic = list(csv.DictReader(m3.trainer.semantic.semantic_manifest_path.open("r", encoding="utf-8-sig", newline="")))
    if m2_semantic != m3_semantic:
        _add(checks, "m2_m3_semantic_rows_match", "fail", "Semantic manifest rows differ.", "hard")
    else:
        _add(checks, "m2_m3_semantic_rows_match", "pass", "Semantic manifest rows match exactly.", "info")
    return details


def audit_mri_us(
    raw_config: dict[str, Any],
    bundle_report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    paths = bundle_report.get("paths", {})
    manifest_path = Path(str(paths.get("manifest_path", "")))
    if not manifest_path.exists():
        return
    rows = _rows(manifest_path)
    mri_rows = [row for row in rows if str(row.get("modality", "")).lower() == "mri"]
    us_rows = [row for row in rows if str(row.get("modality", "")).lower() == "ultrasound"]
    rejected_mri = []
    for row in mri_rows:
        if _mri_row_has_rejected_sequence(row):
            rejected_mri.append(row.get("image_id", ""))
    if rejected_mri:
        _add(checks, "mri_formal_filter_no_non_diagnostic_sequences", "fail", f"MRI formal branch includes rejected terms: {rejected_mri[:10]}", "hard")
    elif mri_rows:
        _add(checks, "mri_formal_filter_no_non_diagnostic_sequences", "pass", "MRI rows do not include scout/locator/pelvic terms.", "info")
    if us_rows:
        bad_us_labels = [row.get("image_id", "") for row in us_rows if not str(row.get("benign_malignant_label", row.get("cancer_label", ""))).strip()]
        if bad_us_labels:
            _add(checks, "ultrasound_label_boundary", "warn", f"Ultrasound rows lack explicit benign/malignant labels: {bad_us_labels[:10]}", "warn")
        else:
            _add(checks, "ultrasound_label_boundary", "pass", "Ultrasound rows have explicit label fields.", "info")
