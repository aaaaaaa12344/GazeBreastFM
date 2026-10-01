from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_MAMMOGRAPHY_MODALITIES = frozenset({"mammo", "mammography", "ffdm", "xray mammography"})


def _is_mammography(modality: str) -> bool:
    return modality.strip().lower() in _MAMMOGRAPHY_MODALITIES


def _clean(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class PatientSplitAuditResult:
    status: str = "pass"  # pass / fail / blocked_template / pass_with_warnings
    total_rows: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    patient_count: int = 0
    leaking_patient_count: int = 0
    leaking_breast_side_count: int = 0
    leaking_study_count: int = 0
    leaking_examples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)


def audit_patient_split(
    manifest_path: str | Path,
    *,
    run_tier: str = "development",
    is_fixture: bool = False,
    is_final_config_template: bool = False,
) -> PatientSplitAuditResult:
    """Audit a Stage 1 manifest for patient-level split leakage.

    Args:
        manifest_path: Path to manifest_stage1_semantic.csv.
        run_tier: One of development / formal / final / final_config_template.
        is_fixture: True if manifest is a fixture/smoke asset.
        is_final_config_template: True if the config is final_config_template.

    Returns:
        PatientSplitAuditResult with status and leakage details.
    """
    result = PatientSplitAuditResult()
    manifest_path = Path(manifest_path)
    is_formal = run_tier in ("formal", "final")

    if not manifest_path.exists():
        result.errors.append(f"Manifest not found: {manifest_path}")
        result.status = "fail" if is_formal else "blocked_template"
        _add_check(result, "manifest_exists", "fail", str(manifest_path))
        return result

    rows, columns = _read_csv(manifest_path)
    result.total_rows = len(rows)
    _add_check(result, "manifest_exists", "pass", f"{len(rows)} rows, {len(columns)} columns")

    # Gate 1: split column must exist
    if "split" not in columns:
        msg = "manifest missing 'split' column; patient-level split cannot be enforced."
        if is_formal and not is_fixture:
            result.errors.append(msg)
            result.status = "fail"
            _add_check(result, "split_column_exists", "fail", msg)
        elif is_final_config_template or is_fixture:
            result.warnings.append(msg)
            result.status = "blocked_template"
            _add_check(result, "split_column_exists", "blocked_template", msg + " fixture/template accepted with blocking_warning.")
        else:
            result.warnings.append(msg)
            _add_check(result, "split_column_exists", "warn", msg)
        return result
    _add_check(result, "split_column_exists", "pass", f"split column present with values: {sorted(set(_clean(r.get('split')) for r in rows))}")

    # Gate 2: patient_id must exist (required for all modalities)
    if "patient_id" not in columns:
        msg = "manifest missing 'patient_id' column."
        if is_formal:
            result.errors.append(msg)
            result.status = "fail"
            _add_check(result, "patient_id_column_exists", "fail", msg)
        else:
            result.warnings.append(msg)
            _add_check(result, "patient_id_column_exists", "warn", msg)
        return result
    _add_check(result, "patient_id_column_exists", "pass", "patient_id column present")

    # Build split index
    split_index: dict[str, set[str]] = {}  # split -> set of patient_ids
    patient_index: dict[str, dict[str, set[str]]] = {}  # patient_id -> {split: set of row indices}
    breast_index: dict[str, dict[str, set[str]]] = {}  # (patient_id, laterality) -> {split: set}
    study_index: dict[str, dict[str, set[str]]] = {}  # (patient_id, study_id) -> {split: set}

    has_study_id = "study_id" in columns
    has_laterality = "laterality" in columns

    for i, row in enumerate(rows):
        split = _clean(row.get("split"))
        patient_id = _clean(row.get("patient_id"))
        modality = _clean(row.get("modality")).lower()
        study_id = _clean(row.get("study_id")) if has_study_id else ""
        laterality = _clean(row.get("laterality")) if has_laterality else ""

        if not split or not patient_id:
            continue

        split_index.setdefault(split, set()).add(patient_id)
        patient_index.setdefault(patient_id, {}).setdefault(split, set()).add(str(i))

        if _is_mammography(modality):
            if laterality:
                breast_key = f"{patient_id}|{laterality}"
                breast_index.setdefault(breast_key, {}).setdefault(split, set()).add(str(i))
            if study_id:
                study_key = f"{patient_id}|{study_id}"
                study_index.setdefault(study_key, {}).setdefault(split, set()).add(str(i))

    result.patient_count = len(patient_index)
    result.split_counts = {s: len(pids) for s, pids in split_index.items()}
    _add_check(result, "split_index_built", "pass",
               f"{result.patient_count} unique patients across {len(split_index)} splits: {result.split_counts}")

    # Leakage checks
    valid_splits = {"train", "val", "test"}

    # Check 1: same patient_id across splits
    patient_leaks = _find_cross_split_leaks(patient_index, valid_splits)
    result.leaking_patient_count = len(patient_leaks)
    if patient_leaks:
        examples = []
        for pid, splits in list(patient_leaks.items())[:5]:
            examples.append({"patient_id": pid, "splits": sorted(splits.keys()),
                             "row_indices": {s: sorted(list(v)[:3]) for s, v in splits.items()}})
        result.leaking_examples = examples
        msg = f"{len(patient_leaks)} patient(s) appear in multiple splits."
        if is_formal:
            result.errors.append(msg)
            result.status = "fail"
            _add_check(result, "patient_split_leakage", "fail", msg)
        elif is_final_config_template or is_fixture:
            result.warnings.append(msg)
            _add_check(result, "patient_split_leakage", "blocking_warning",
                       msg + " fixture/template: split violations must be resolved before formal training.")
        else:
            result.warnings.append(msg)
            _add_check(result, "patient_split_leakage", "warn", msg)
    else:
        _add_check(result, "patient_split_leakage", "pass", "no patient-level split leakage detected.")

    # Check 2: same patient_id + laterality across splits (mammography only)
    breast_leaks = _find_cross_split_leaks(breast_index, valid_splits)
    result.leaking_breast_side_count = len(breast_leaks)
    if breast_leaks:
        msg = f"{len(breast_leaks)} patient+laterality pair(s) appear in multiple splits."
        if is_formal:
            result.errors.append(msg)
            if result.status != "fail":
                result.status = "fail"
            _add_check(result, "breast_side_split_leakage", "fail", msg)
        elif is_final_config_template or is_fixture:
            result.warnings.append(msg)
            _add_check(result, "breast_side_split_leakage", "blocking_warning", msg)
        else:
            result.warnings.append(msg)
            _add_check(result, "breast_side_split_leakage", "warn", msg)
    else:
        _add_check(result, "breast_side_split_leakage", "pass", "no breast-side split leakage detected.")

    # Check 3: same patient_id + study_id across splits
    study_leaks = _find_cross_split_leaks(study_index, valid_splits)
    result.leaking_study_count = len(study_leaks)
    if study_leaks:
        msg = f"{len(study_leaks)} patient+study pair(s) appear in multiple splits."
        if is_formal:
            result.errors.append(msg)
            if result.status != "fail":
                result.status = "fail"
            _add_check(result, "study_split_leakage", "fail", msg)
        elif is_final_config_template or is_fixture:
            result.warnings.append(msg)
            _add_check(result, "study_split_leakage", "blocking_warning", msg)
        else:
            result.warnings.append(msg)
            _add_check(result, "study_split_leakage", "warn", msg)
    else:
        _add_check(result, "study_split_leakage", "pass", "no study-level split leakage detected.")

    # Check 4: mammography identity columns present
    mammo_rows = [r for r in rows if _is_mammography(_clean(r.get("modality")))]
    if mammo_rows:
        missing_mammo = []
        for col in ("patient_id", "study_id", "laterality", "view"):
            if col not in columns:
                missing_mammo.append(col)
        if missing_mammo:
            msg = f"mammography rows present but missing identity columns: {missing_mammo}"
            if is_formal:
                result.errors.append(msg)
                result.status = "fail"
                _add_check(result, "mammography_identity_columns", "fail", msg)
            else:
                result.warnings.append(msg)
                _add_check(result, "mammography_identity_columns", "warn", msg)
        else:
            _add_check(result, "mammography_identity_columns", "pass",
                       f"all mammography identity columns present for {len(mammo_rows)} mammography rows")

    # Check 5: MRI/US at minimum have patient_id
    non_mammo = [r for r in rows if not _is_mammography(_clean(r.get("modality")))]
    if non_mammo:
        missing_pid = sum(1 for r in non_mammo if not _clean(r.get("patient_id")))
        if missing_pid:
            msg = f"{missing_pid}/{len(non_mammo)} non-mammography rows missing patient_id"
            if is_formal:
                result.errors.append(msg)
                result.status = "fail"
                _add_check(result, "non_mammography_patient_id", "fail", msg)
            else:
                result.warnings.append(msg)
                _add_check(result, "non_mammography_patient_id", "warn", msg)
        else:
            _add_check(result, "non_mammography_patient_id", "pass",
                       f"all {len(non_mammo)} non-mammography rows have patient_id")
        extra_cols = [c for c in ("study_id", "exam_id", "case_id") if c in columns]
        if extra_cols:
            _add_check(result, "non_mammography_extra_identity", "pass",
                       f"non-mammography rows have additional identity columns: {extra_cols}")

    # Final status resolution
    if result.status not in ("fail", "blocked_template"):
        if result.warnings:
            result.status = "pass_with_warnings"
        else:
            result.status = "pass"

    return result


def _find_cross_split_leaks(
    index: dict[str, dict[str, set[str]]],
    valid_splits: set[str],
) -> dict[str, dict[str, set[str]]]:
    """Find keys that appear in more than one valid split."""
    leaks: dict[str, dict[str, set[str]]] = {}
    for key, split_map in index.items():
        active_splits = {s for s in split_map if s in valid_splits}
        if len(active_splits) > 1:
            leaks[key] = {s: split_map[s] for s in active_splits}
    return leaks


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _add_check(result: PatientSplitAuditResult, check_id: str, status: str, message: str) -> None:
    result.checks.append({"id": check_id, "status": status, "message": message})


def build_patient_split_report(
    manifest_path: str | Path,
    *,
    run_tier: str = "development",
    is_fixture: bool = False,
    is_final_config_template: bool = False,
) -> dict[str, Any]:
    """Run patient split audit and return a JSON-serializable report dict."""
    audit = audit_patient_split(
        manifest_path,
        run_tier=run_tier,
        is_fixture=is_fixture,
        is_final_config_template=is_final_config_template,
    )
    return {
        "schema_version": "patient_split_audit_v1",
        "manifest_path": str(manifest_path),
        "run_tier": run_tier,
        "is_fixture": is_fixture,
        "is_final_config_template": is_final_config_template,
        "status": audit.status,
        "total_rows": audit.total_rows,
        "split_counts": audit.split_counts,
        "patient_count": audit.patient_count,
        "leaking_patient_count": audit.leaking_patient_count,
        "leaking_breast_side_count": audit.leaking_breast_side_count,
        "leaking_study_count": audit.leaking_study_count,
        "leaking_examples": audit.leaking_examples,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "checks": audit.checks,
    }
