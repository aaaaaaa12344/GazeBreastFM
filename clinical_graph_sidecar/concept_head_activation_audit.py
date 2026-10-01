from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from breast_pretrain.clinical_graph_sidecar.activation_audit import (
    HEAD_CLASS_LABELS,
    NOT_APPLICABLE_STATUS_VALUES,
    PROJECT_ROOT,
    RAW_UNCONFIRMED_STATUS_VALUES,
    _clinical_graph_paths,
    _config_image_manifest_path,
    _load_resolved_config,
    _node_sample_status,
    _truthy_mask,
    build_node_id_head_mapping,
    read_csv_rows,
    read_jsonl_rows,
)
from breast_pretrain.text.clinical_concepts import SUPPORTED_STAGE1_CONCEPT_HEADS


DIAG500_PRIORITY_HEADS = (
    "benign_malignant_label",
    "mri_sequence",
    "mri_treatment_response",
)
DEFAULT_MIN_CONFIRMED_COUNT = 2
DEFAULT_MIN_COVERAGE_RATIO_BY_APPLICABLE_MODALITY = 0.5
DEFAULT_MIN_CLASS_DIVERSITY = 2
TRIAL_ACTIVE_CANDIDATE_HEADS = frozenset(DIAG500_PRIORITY_HEADS)
HEAD_APPLICABLE_MODALITIES = {
    "mri_sequence": {"mri"},
    "mri_enhancement": {"mri"},
    "mri_kinetic_curve": {"mri"},
    "mri_treatment_response": {"mri"},
    "us_shape": {"ultrasound"},
    "us_margin": {"ultrasound"},
    "us_echogenicity": {"ultrasound"},
    "us_orientation": {"ultrasound"},
    "us_posterior_feature": {"ultrasound"},
    "us_vascularity": {"ultrasound"},
    "view": {"mammography"},
    "density": {"mammography"},
    "birads": {"mammography"},
    "finding": {"mammography"},
}


def _manifest_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_head_status(row: dict[str, str], head_name: str) -> str:
    status = str(row.get(f"{head_name}_status", "")).strip().lower()
    value = str(row.get(head_name, "")).strip().lower()
    if status in RAW_UNCONFIRMED_STATUS_VALUES:
        return "raw_unconfirmed"
    if status in NOT_APPLICABLE_STATUS_VALUES or value in NOT_APPLICABLE_STATUS_VALUES:
        return "not_applicable"
    if _truthy_mask(row.get(f"{head_name}_observed_mask")):
        return "confirmed"
    return "missing"


def _is_applicable_modality(row: dict[str, Any], head_name: str) -> bool:
    modalities = HEAD_APPLICABLE_MODALITIES.get(head_name)
    if not modalities:
        return True
    modality = str(row.get("modality", "")).strip().lower()
    if modality == "mammo":
        modality = "mammography"
    if modality == "us":
        modality = "ultrasound"
    return modality in modalities


def _manifest_confirmed_class(row: dict[str, str], head_name: str) -> str | None:
    if _manifest_head_status(row, head_name) != "confirmed":
        return None
    value = str(row.get(head_name, "")).strip().lower()
    return value or None


def _sidecar_head_status(
    *,
    row: dict[str, Any],
    head_name: str,
    nodes_by_id: dict[str, dict[str, str]],
    node_head_bindings: dict[str, dict[str, Any]],
) -> str:
    statuses: list[str] = []
    for node_id, binding in node_head_bindings.items():
        if str(binding.get("head_name")) != head_name:
            continue
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        statuses.append(_node_sample_status(row=row, node=node))
    if "confirmed" in statuses:
        return "confirmed"
    if "raw_unconfirmed" in statuses:
        return "raw_unconfirmed"
    if statuses and all(status == "not_applicable" for status in statuses):
        return "not_applicable"
    return "missing"


def _sidecar_head_confirmed_classes(
    *,
    row: dict[str, Any],
    head_name: str,
    nodes_by_id: dict[str, dict[str, str]],
    node_head_bindings: dict[str, dict[str, Any]],
) -> set[str]:
    classes: set[str] = set()
    for node_id, binding in node_head_bindings.items():
        if str(binding.get("head_name")) != head_name:
            continue
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        if _node_sample_status(row=row, node=node) != "confirmed":
            continue
        labels = HEAD_CLASS_LABELS.get(head_name, ())
        class_index = int(binding.get("class_index", -1))
        if 0 <= class_index < len(labels):
            classes.add(str(labels[class_index]))
        else:
            classes.add(node_id)
    return classes


def _activation_decision(
    *,
    counts: Counter[str],
    head_name: str,
    active_heads: set[str],
    pending_heads: set[str],
    concept_weights: dict[str, Any],
) -> str:
    confirmed = int(counts.get("confirmed", 0))
    weight = float(concept_weights.get(head_name, 0.0))
    if confirmed > 0 and head_name in active_heads and weight > 0.0:
        return "active_confirmed_supervised"
    if confirmed > 0 and head_name in active_heads:
        return "active_confirmed_no_concept_loss_weight"
    if confirmed > 0 and head_name in pending_heads:
        return "confirmed_pending_head_trial_candidate"
    if confirmed > 0:
        return "confirmed_available_but_not_configured"
    return "inactive_no_confirmed_labels"


def _recommended_action(
    *,
    thresholds_met: bool,
    head_name: str,
    configured_active: bool,
    configured_pending: bool,
    loss_weight: float,
) -> str:
    if configured_active and loss_weight > 0.0 and thresholds_met:
        return "active_confirmed_supervised"
    if thresholds_met and head_name in TRIAL_ACTIVE_CANDIDATE_HEADS and not configured_active:
        return "eligible_trial_active"
    if configured_pending or not configured_active:
        return "keep_pending"
    return "keep_pending"


def _audit_status(head_reports: dict[str, dict[str, Any]]) -> str:
    if any(report["recommended_action"] == "eligible_trial_active" for report in head_reports.values()):
        return "eligible_trial_active_heads_present"
    if any(report["recommended_action"] == "active_confirmed_supervised" for report in head_reports.values()):
        return "active_confirmed_supervised_heads_present"
    return "no_trial_active_eligible_heads"


def build_concept_head_activation_audit_from_resolved_config(
    resolved_config_path: Path,
    *,
    min_confirmed_count: int = DEFAULT_MIN_CONFIRMED_COUNT,
    min_coverage_ratio_by_applicable_modality: float = DEFAULT_MIN_COVERAGE_RATIO_BY_APPLICABLE_MODALITY,
    min_class_diversity: int = DEFAULT_MIN_CLASS_DIVERSITY,
) -> dict[str, Any]:
    resolved = Path(resolved_config_path).expanduser().resolve()
    config = _load_resolved_config(resolved)
    semantic = config.get("semantic") if isinstance(config.get("semantic"), dict) else {}
    losses = config.get("losses") if isinstance(config.get("losses"), dict) else {}
    active_heads = set(str(item) for item in semantic.get("active_concept_heads", []))
    pending_heads = set(str(item) for item in semantic.get("pending_concept_heads", []))
    concept_weights = losses.get("concept_head_weights") if isinstance(losses.get("concept_head_weights"), dict) else {}
    manifest_path = _config_image_manifest_path(config, resolved.parent)
    graph_paths = _clinical_graph_paths(config, PROJECT_ROOT)
    nodes_path = graph_paths.get("nodes_path")
    sidecar_path = graph_paths.get("sidecar_case_concept_vector_path")

    heads = tuple(dict.fromkeys(DIAG500_PRIORITY_HEADS + SUPPORTED_STAGE1_CONCEPT_HEADS + tuple(active_heads) + tuple(pending_heads)))
    counters = {head: Counter({"confirmed": 0, "missing": 0, "raw_unconfirmed": 0, "not_applicable": 0}) for head in heads}
    confirmed_classes: dict[str, set[str]] = {head: set() for head in heads}

    source = "manifest"
    if sidecar_path is not None and Path(sidecar_path).is_file() and nodes_path is not None and Path(nodes_path).is_file():
        source = "sidecar"
        nodes = read_csv_rows(Path(nodes_path))
        nodes_by_id = {str(row.get("node_id", "")).strip(): row for row in nodes}
        node_ids = list(nodes_by_id)
        bindings = build_node_id_head_mapping(set(HEAD_CLASS_LABELS) | {"cancer_label"}, node_ids)
        for row in read_jsonl_rows(Path(sidecar_path)):
            for head in heads:
                if not _is_applicable_modality(row, head):
                    counters[head]["not_applicable"] += 1
                    continue
                counters[head][
                    _sidecar_head_status(
                        row=row,
                        head_name=head,
                        nodes_by_id=nodes_by_id,
                        node_head_bindings=bindings,
                    )
                ] += 1
                confirmed_classes[head].update(
                    _sidecar_head_confirmed_classes(
                        row=row,
                        head_name=head,
                        nodes_by_id=nodes_by_id,
                        node_head_bindings=bindings,
                    )
                )
    else:
        for row in _manifest_rows(manifest_path):
            for head in heads:
                if not _is_applicable_modality(row, head):
                    counters[head]["not_applicable"] += 1
                    continue
                counters[head][_manifest_head_status(row, head)] += 1
                confirmed_class = _manifest_confirmed_class(row, head)
                if confirmed_class is not None:
                    confirmed_classes[head].add(confirmed_class)

    head_reports = {}
    for head in heads:
        counts = counters[head]
        applicable_count = int(
            counts["confirmed"] + counts["missing"] + counts["raw_unconfirmed"]
        )
        coverage_ratio = (
            float(counts["confirmed"]) / float(applicable_count)
            if applicable_count > 0
            else 0.0
        )
        class_diversity = int(len(confirmed_classes[head]))
        thresholds_met = (
            int(counts["confirmed"]) >= int(min_confirmed_count)
            and coverage_ratio >= float(min_coverage_ratio_by_applicable_modality)
            and class_diversity >= int(min_class_diversity)
        )
        loss_weight = float(concept_weights.get(head, 0.0))
        configured_active = head in active_heads
        configured_pending = head in pending_heads
        head_reports[head] = {
            "head_name": head,
            "confirmed": int(counts["confirmed"]),
            "missing": int(counts["missing"]),
            "raw_unconfirmed": int(counts["raw_unconfirmed"]),
            "not_applicable": int(counts["not_applicable"]),
            "applicable_sample_count": applicable_count,
            "coverage_ratio_by_applicable_modality": coverage_ratio,
            "class_diversity": class_diversity,
            "confirmed_classes": sorted(confirmed_classes[head]),
            "min_confirmed_count": int(min_confirmed_count),
            "min_coverage_ratio_by_applicable_modality": float(
                min_coverage_ratio_by_applicable_modality
            ),
            "min_class_diversity": int(min_class_diversity),
            "thresholds_met": bool(thresholds_met),
            "activation_decision": _activation_decision(
                counts=counts,
                head_name=head,
                active_heads=active_heads,
                pending_heads=pending_heads,
                concept_weights=concept_weights,
            ),
            "recommended_action": _recommended_action(
                thresholds_met=thresholds_met,
                head_name=head,
                configured_active=configured_active,
                configured_pending=configured_pending,
                loss_weight=loss_weight,
            ),
            "configured_active": configured_active,
            "configured_pending": configured_pending,
            "loss_weight": loss_weight,
        }

    return {
        "schema_version": "stage1_concept_head_activation_audit_v1",
        "concept_head_activation_audit_status": _audit_status(head_reports),
        "resolved_config_path": str(resolved),
        "source": source,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "sidecar_case_concept_vector_path": str(sidecar_path) if sidecar_path is not None else None,
        "diag500_priority_heads": list(DIAG500_PRIORITY_HEADS),
        "thresholds": {
            "min_confirmed_count": int(min_confirmed_count),
            "min_coverage_ratio_by_applicable_modality": float(
                min_coverage_ratio_by_applicable_modality
            ),
            "min_class_diversity": int(min_class_diversity),
        },
        "active_concept_heads": sorted(active_heads),
        "pending_concept_heads": sorted(pending_heads),
        "heads": head_reports,
        "policy": {
            "does_not_modify_config": True,
            "confirmed_labels_only_for_supervised_loss": True,
            "missing_raw_unconfirmed_not_applicable_are_not_negative": True,
        },
    }


__all__ = [
    "DEFAULT_MIN_CLASS_DIVERSITY",
    "DEFAULT_MIN_CONFIRMED_COUNT",
    "DEFAULT_MIN_COVERAGE_RATIO_BY_APPLICABLE_MODALITY",
    "DIAG500_PRIORITY_HEADS",
    "build_concept_head_activation_audit_from_resolved_config",
]
