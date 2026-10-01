"""Identity, split, and modality-specific formal identity checks for candidate rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


_MAMMOGRAPHY_MODALITIES = frozenset({"mammo", "mammography", "ffdm", "dbt", "cesm"})
_ULTRASOUND_MODALITIES = frozenset({"ultrasound", "us"})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_mammography(record: Mapping[str, Any]) -> bool:
    return _text(record.get("modality")).lower() in _MAMMOGRAPHY_MODALITIES


def _append(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)


def validate_ready_modality_identity(record: Mapping[str, Any]) -> list[str]:
    """Validate only a formal-ready row; pending/excluded records may remain incomplete."""

    if record.get("candidate_ready") is not True:
        return []
    errors: list[str] = []
    modality = _text(record.get("modality")).lower()
    if modality in _MAMMOGRAPHY_MODALITIES:
        if not (_text(record.get("study_id")) or _text(record.get("exam_id"))):
            _append(errors, "formal mammography/DBT/CESM row requires study_id or exam_id.")
        for field in ("laterality", "view"):
            if not _text(record.get(field)):
                _append(errors, f"formal mammography/DBT/CESM row requires {field}.")
    elif modality in _ULTRASOUND_MODALITIES:
        if not (_text(record.get("lesion_id")) or _text(record.get("image_observation_id"))):
            _append(errors, "formal ultrasound row requires lesion_id or image_observation_id.")
    elif modality == "mri":
        for field in ("study_id", "series_group_id", "timepoint_id"):
            if not _text(record.get(field)):
                _append(errors, f"formal MRI row requires {field}.")
        if not (_text(record.get("sequence_phase")) or (_text(record.get("sequence")) and _text(record.get("phase")))):
            _append(errors, "formal MRI row requires explicit sequence_phase or both sequence and phase.")
    return errors


def candidate_has_ready_modality_identity(record: Mapping[str, Any]) -> bool:
    """Use source fields to gate readiness before a frozen candidate field exists."""

    candidate_view = dict(record)
    candidate_view["candidate_ready"] = True
    return not validate_ready_modality_identity(candidate_view)


def validate_candidate_identity_consistency(candidate_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Fail closed on split conflicts; never infer any identity field from a path."""

    errors: list[str] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_study_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    mammography_laterality_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        patient_id = _text(row.get("patient_id"))
        split = _text(row.get("patient_split"))
        if patient_id and split:
            patient_splits[patient_id].add(split)
            study_id = _text(row.get("study_id"))
            if study_id:
                patient_study_splits[(patient_id, study_id)].add(split)
            laterality = _text(row.get("laterality"))
            if laterality and _is_mammography(row):
                mammography_laterality_splits[(patient_id, laterality)].add(split)
        case_key = tuple(_text(row.get(field)) for field in ("dataset_id", "case_id", "report_unit_id", "report_id"))
        if all(case_key):
            case_groups[case_key].append(row)
    for patient_id, splits in sorted(patient_splits.items()):
        if len(splits) > 1:
            _append(errors, f"patient_id={patient_id!r} maps to multiple splits: {sorted(splits)}.")
    for key, splits in sorted(patient_study_splits.items()):
        if len(splits) > 1:
            _append(errors, f"patient_id+study_id={key!r} maps to multiple splits: {sorted(splits)}.")
    for key, splits in sorted(mammography_laterality_splits.items()):
        if len(splits) > 1:
            _append(errors, f"mammography patient_id+laterality={key!r} maps to multiple splits: {sorted(splits)}.")
    for key, rows in sorted(case_groups.items()):
        patient_ids = {_text(row.get("patient_id")) for row in rows}
        splits = {_text(row.get("patient_split")) for row in rows}
        if len(patient_ids) > 1:
            _append(errors, f"case/report group={key!r} maps to multiple patient_id values: {sorted(patient_ids)}.")
        if len(splits) > 1:
            _append(errors, f"case/report group={key!r} maps to multiple splits: {sorted(splits)}.")
    return errors
