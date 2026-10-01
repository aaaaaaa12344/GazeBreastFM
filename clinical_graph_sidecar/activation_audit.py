from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.text.clinical_concepts import (
    BENIGN_MALIGNANT_LABELS,
    BIRADS_LABELS,
    DENSITY_LABELS,
    FINDING_LABELS,
    LATERALITY_LABELS,
    MRI_ENHANCEMENT_LABELS,
    MRI_KINETIC_CURVE_LABELS,
    MRI_SEQUENCE_LABELS,
    MRI_TREATMENT_RESPONSE_LABELS,
    US_ECHOGENICITY_LABELS,
    US_MARGIN_LABELS,
    US_ORIENTATION_LABELS,
    US_POSTERIOR_FEATURE_LABELS,
    US_SHAPE_LABELS,
    US_VASCULARITY_LABELS,
    VIEW_LABELS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "configs" / "clinical_graph" / "tri_modal_clinical_graph_v1"
CONFIRMED_STATUS_VALUES = {"confirmed", "observed", "present", "verified", "structured_label"}
RAW_UNCONFIRMED_STATUS_VALUES = {"raw_unconfirmed", "unconfirmed"}
NOT_APPLICABLE_STATUS_VALUES = {"not_applicable", "not applicable", "na", "n/a"}
SUPPORTED_RULE_TYPES = {
    "conflicts_with",
    "implies",
    "maps_to_shared_concept",
    "risk_order",
}
HEAD_CLASS_LABELS: dict[str, tuple[str, ...]] = {
    "view": tuple(label.lower() for label in VIEW_LABELS),
    "laterality": LATERALITY_LABELS,
    "density": tuple(label.lower() for label in DENSITY_LABELS),
    "birads": tuple(label.lower() for label in BIRADS_LABELS),
    "finding": FINDING_LABELS,
    "benign_malignant_label": BENIGN_MALIGNANT_LABELS,
    "mri_sequence": MRI_SEQUENCE_LABELS,
    "mri_enhancement": MRI_ENHANCEMENT_LABELS,
    "mri_kinetic_curve": MRI_KINETIC_CURVE_LABELS,
    "mri_treatment_response": MRI_TREATMENT_RESPONSE_LABELS,
    "us_shape": US_SHAPE_LABELS,
    "us_margin": US_MARGIN_LABELS,
    "us_echogenicity": US_ECHOGENICITY_LABELS,
    "us_orientation": US_ORIENTATION_LABELS,
    "us_posterior_feature": US_POSTERIOR_FEATURE_LABELS,
    "us_vascularity": US_VASCULARITY_LABELS,
}
HEAD_TYPES: dict[str, str] = {
    "cancer_label": "binary",
    "finding": "multilabel",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            rows.append(payload)
    return rows


def _resolve_path(raw_value: Any, base_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value or value.startswith("FORMAL_STAGE1_BUNDLE"):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _truthy_mask(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return float(raw_value) > 0.5
    normalized = str(raw_value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "observed", "confirmed", "present"}


def _lower_lookup(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value is None:
        return ""
    return str(value).strip().lower()


def graph_node_to_concept_head(node_id: str, active_heads: set[str] | None = None) -> str | None:
    binding = graph_node_to_concept_binding(node_id, active_heads)
    return str(binding["head_name"]) if binding is not None else None


def _binding(head_name: str, class_index: int, *, head_type: str | None = None) -> dict[str, Any]:
    return {
        "head_name": head_name,
        "class_index": int(class_index),
        "head_type": head_type or HEAD_TYPES.get(head_name, "multiclass"),
    }


def _label_binding(head_name: str, label: str, *, head_type: str | None = None) -> dict[str, Any] | None:
    labels = HEAD_CLASS_LABELS.get(head_name)
    if labels is None:
        return None
    normalized = str(label).strip().lower()
    if normalized not in labels:
        return None
    return _binding(head_name, labels.index(normalized), head_type=head_type)


def graph_node_to_concept_binding(
    node_id: str,
    active_heads: set[str] | None = None,
) -> dict[str, Any] | None:
    active = active_heads or set()
    candidates: list[tuple[str, str, str]] = [
        ("mammography.view.", "view", ""),
        ("laterality.", "laterality", ""),
        ("mammography.density.", "density", ""),
        ("mammography.finding.", "finding", "multilabel"),
        ("mammography.assessment.birads_", "birads", ""),
        ("mri.sequence.", "mri_sequence", ""),
        ("mri.enhancement.", "mri_enhancement", ""),
        ("mri.kinetic_curve.", "mri_kinetic_curve", ""),
        ("mri.treatment_response.", "mri_treatment_response", ""),
        ("ultrasound.shape.", "us_shape", ""),
        ("ultrasound.margin.", "us_margin", ""),
        ("ultrasound.echogenicity.", "us_echogenicity", ""),
        ("ultrasound.orientation.", "us_orientation", ""),
        ("ultrasound.posterior_feature.", "us_posterior_feature", ""),
        ("ultrasound.vascularity.", "us_vascularity", ""),
    ]
    if node_id == "finding.no_finding":
        return _label_binding("finding", "no_finding", head_type="multilabel")
    if node_id in {"benign_malignant", "benign_malignant.benign", "benign_malignant.malignant"}:
        label = "malignant" if node_id.endswith(".malignant") else "benign"
        if "benign_malignant_label" in active:
            return _label_binding("benign_malignant_label", label)
        if "cancer_label" in active:
            return _binding("cancer_label", 1 if label == "malignant" else 0, head_type="binary")
        return _label_binding("benign_malignant_label", label)
    for prefix, head_name, head_type in candidates:
        if node_id == prefix or node_id.startswith(prefix):
            raw_label = node_id.removeprefix(prefix)
            if head_name == "birads":
                raw_label = raw_label.replace("birads_", "")
            if head_name == "view":
                raw_label = raw_label.upper()
            binding = _label_binding(head_name, raw_label, head_type=head_type or None)
            if binding is not None:
                return binding
    return None


def build_node_id_head_mapping(
    active_heads: set[str] | tuple[str, ...],
    node_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    active = set(active_heads)
    ids = list(node_ids or ())
    static_ids = [
        "finding.no_finding",
        "benign_malignant",
        "benign_malignant.benign",
        "benign_malignant.malignant",
    ]
    for index in range(10):
        ids.append(f"mammography.assessment.birads_{index}")
    for suffix in ("cc", "mlo"):
        ids.append(f"mammography.view.{suffix}")
    for suffix in ("a", "b", "c", "d"):
        ids.append(f"mammography.density.{suffix}")
    ids.extend(static_ids)

    mapping: dict[str, dict[str, Any]] = {}
    for node_id in ids:
        binding = graph_node_to_concept_binding(node_id, active)
        if binding is not None and str(binding["head_name"]) in active:
            mapping[node_id] = binding
    return mapping


def _load_resolved_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved config must be a mapping: {path}")
    return payload


def _default_graph_path(filename: str) -> Path:
    return DEFAULT_GRAPH_DIR / filename


def _clinical_graph_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Path | None]:
    block = config.get("clinical_graph") if isinstance(config.get("clinical_graph"), dict) else {}
    image_manifest_path = _config_image_manifest_path(config, base_dir)
    sidecar_path = _resolve_path(block.get("sidecar_case_concept_vector_path"), base_dir)
    if sidecar_path is None:
        sidecar_path = _infer_sidecar_path_from_manifest(image_manifest_path)
    return {
        "nodes_path": _resolve_path(block.get("nodes_path"), base_dir)
        or _default_graph_path("breast_multimodal_nodes_v1.csv"),
        "edges_path": _resolve_path(block.get("edges_path"), base_dir)
        or _default_graph_path("breast_multimodal_edges_v1.csv"),
        "mapping_rules_path": _resolve_path(block.get("mapping_rules_path"), base_dir)
        or _default_graph_path("breast_multimodal_mapping_rules_v1.yaml"),
        "consistency_rules_path": _resolve_path(block.get("consistency_rules_path"), base_dir)
        or _default_graph_path("graph_consistency_rules_v1.yaml"),
        "prompt_templates_path": _resolve_path(block.get("prompt_templates_path"), base_dir)
        or _default_graph_path("concept_prompt_templates_v1.yaml"),
        "sidecar_case_concept_vector_path": sidecar_path,
    }


def _config_image_manifest_path(config: dict[str, Any], base_dir: Path) -> Path | None:
    data_block = config.get("data") if isinstance(config.get("data"), dict) else {}
    raw = data_block.get("image_manifest_path") or config.get("image_manifest_path")
    return _resolve_path(raw, base_dir)


def _infer_sidecar_path_from_manifest(image_manifest_path: Path | None) -> Path | None:
    if image_manifest_path is None:
        return None
    manifest_dir = image_manifest_path.parent
    candidates = (
        manifest_dir / "tri_modal_case_concept_vector.jsonl",
        manifest_dir / "stage1_case_concept_vector.jsonl",
        manifest_dir / "clinical_graph_sidecar" / "tri_modal_case_concept_vector.jsonl",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def graph_activation_audit_status(
    *,
    sample_count: int,
    sidecar_case_concept_vector_path: Path | None,
    require_sidecar: bool,
    is_formal_tier: bool,
) -> str:
    if sidecar_case_concept_vector_path is not None and sidecar_case_concept_vector_path.is_file() and sample_count > 0:
        return "ok"
    if require_sidecar:
        return "missing_sidecar_required"
    if is_formal_tier:
        return "missing_or_empty_sidecar_formal_audit_incomplete"
    return "missing_or_empty_sidecar_allowed"


def _is_formal_resolved_config(config: dict[str, Any]) -> bool:
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    values = (
        str(metadata.get("run_tier", "")).lower(),
        str(metadata.get("model_role", "")).lower(),
        str(metadata.get("compliance_status", "")).lower(),
        str(config.get("config_path", "")).lower(),
    )
    return any("formal" in value or "production" in value for value in values)


def _sidecar_node_maps(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    values = row.get("concept_values") if isinstance(row.get("concept_values"), dict) else {}
    masks = row.get("observed_mask")
    if not isinstance(masks, dict):
        masks = row.get("observed_masks") if isinstance(row.get("observed_masks"), dict) else {}
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return values, masks, status, source


def _node_sample_status(
    *,
    row: dict[str, Any],
    node: dict[str, str],
) -> str:
    node_id = str(node.get("node_id", "")).strip()
    node_modality = str(node.get("modality_scope", "")).strip().lower()
    sample_modality = str(row.get("modality", "")).strip().lower()
    if node_modality and node_modality != "all" and sample_modality and node_modality != sample_modality:
        return "not_applicable"

    values, masks, status, source = _sidecar_node_maps(row)
    status_text = _lower_lookup(status, node_id)
    source_text = _lower_lookup(source, node_id)
    value_text = str(values.get(node_id) or "").strip().lower()
    if status_text in RAW_UNCONFIRMED_STATUS_VALUES or source_text in RAW_UNCONFIRMED_STATUS_VALUES:
        return "raw_unconfirmed"
    if status_text in NOT_APPLICABLE_STATUS_VALUES or value_text in NOT_APPLICABLE_STATUS_VALUES:
        return "not_applicable"
    if _truthy_mask(masks.get(node_id)):
        return "confirmed"
    return "missing"


def _row_has_confirmed_head(
    *,
    row: dict[str, Any],
    head_name: str,
    node_lookup: dict[str, dict[str, str]],
    node_bindings: dict[str, dict[str, Any]],
) -> bool:
    for node_id, binding in node_bindings.items():
        if str(binding.get("head_name")) != head_name:
            continue
        node = node_lookup.get(node_id)
        if node is not None and _node_sample_status(row=row, node=node) == "confirmed":
            return True
    return False


def _load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _rule_entries(path: Path | None) -> list[dict[str, Any]]:
    payload = _load_yaml_mapping(path)
    rules = payload.get("consistency_rules", [])
    return [rule for rule in rules if isinstance(rule, dict)]


def _concept_label_fields(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        return {"manifest_path": str(manifest_path) if manifest_path is not None else None, "fields": []}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
    concept_fields = sorted(
        field for field in fields
        if field.endswith("_observed_mask") or field.endswith("_status") or field.endswith("_source")
    )
    return {"manifest_path": str(manifest_path), "fields": concept_fields}


def build_graph_activation_audit(
    *,
    resolved_config_path: Path,
    nodes_path: Path,
    edges_path: Path,
    mapping_rules_path: Path,
    consistency_rules_path: Path,
    prompt_templates_path: Path,
    sidecar_case_concept_vector_path: Path | None,
    require_sidecar: bool = True,
) -> dict[str, Any]:
    config = _load_resolved_config(resolved_config_path)
    nodes = read_csv_rows(nodes_path)
    node_lookup = {str(row.get("node_id", "")).strip(): row for row in nodes}
    node_ids = [str(row.get("node_id", "")).strip() for row in nodes]
    rules = _rule_entries(consistency_rules_path)
    semantic = config.get("semantic") if isinstance(config.get("semantic"), dict) else {}
    losses = config.get("losses") if isinstance(config.get("losses"), dict) else {}
    active_heads = set(str(item) for item in semantic.get("active_concept_heads", []))
    pending_heads = set(str(item) for item in semantic.get("pending_concept_heads", []))
    concept_weights = losses.get("concept_head_weights") if isinstance(losses.get("concept_head_weights"), dict) else {}
    base_dir = resolved_config_path.parent
    image_manifest_path = _config_image_manifest_path(config, base_dir)
    is_formal_tier = _is_formal_resolved_config(config)

    if sidecar_case_concept_vector_path is None or not sidecar_case_concept_vector_path.is_file():
        if require_sidecar:
            raise FileNotFoundError(f"Missing tri_modal_case_concept_vector.jsonl: {sidecar_case_concept_vector_path}")
        sidecar_rows: list[dict[str, Any]] = []
    else:
        sidecar_rows = read_jsonl_rows(sidecar_case_concept_vector_path)
    audit_status = graph_activation_audit_status(
        sample_count=len(sidecar_rows),
        sidecar_case_concept_vector_path=sidecar_case_concept_vector_path,
        require_sidecar=require_sidecar,
        is_formal_tier=is_formal_tier,
    )

    node_reports: dict[str, dict[str, Any]] = {}
    aggregate_counts = Counter({"confirmed": 0, "missing": 0, "raw_unconfirmed": 0, "not_applicable": 0})
    by_modality: dict[str, Counter[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", "")).strip()
        counts = Counter({"confirmed": 0, "missing": 0, "raw_unconfirmed": 0, "not_applicable": 0})
        for row in sidecar_rows:
            status = _node_sample_status(row=row, node=node)
            counts[status] += 1
            modality = str(row.get("modality", "")).strip().lower() or "unknown"
            by_modality.setdefault(modality, Counter())
            if status == "confirmed":
                by_modality[modality][node_id] += 1
        aggregate_counts.update(counts)
        head_name = graph_node_to_concept_head(node_id, active_heads)
        head_active = head_name in active_heads if head_name is not None else False
        head_pending = head_name in pending_heads if head_name is not None else False
        head_weight = float(concept_weights.get(head_name, 0.0)) if head_name is not None else 0.0
        if counts["confirmed"] > 0 and head_active and head_weight > 0.0:
            activation_decision = "active_confirmed_supervised"
            loss_role = "concept_target_loss"
            skipped_reason = ""
        elif counts["confirmed"] > 0 and head_active:
            activation_decision = "active_confirmed_no_concept_loss_weight"
            loss_role = "semantic_prior"
            skipped_reason = "concept_head_weight_zero"
        elif counts["confirmed"] > 0 and head_pending:
            activation_decision = "confirmed_but_pending_head"
            loss_role = "prompt_only_or_semantic_prior"
            skipped_reason = "concept_head_pending"
        elif counts["confirmed"] > 0:
            activation_decision = "confirmed_without_runtime_head"
            loss_role = "prompt_only_or_semantic_prior"
            skipped_reason = "no_active_concept_head"
        else:
            activation_decision = "inactive_no_confirmed_labels"
            loss_role = "prompt_only"
            skipped_reason = "no_confirmed_labels"
        node_reports[node_id] = {
            "node_id": node_id,
            "modality_scope": str(node.get("modality_scope", "")).strip(),
            "concept_head": head_name,
            "confirmed": int(counts["confirmed"]),
            "missing": int(counts["missing"]),
            "raw_unconfirmed": int(counts["raw_unconfirmed"]),
            "not_applicable": int(counts["not_applicable"]),
            "activation_decision": activation_decision,
            "loss_role": loss_role,
            "skipped_reason": skipped_reason,
        }

    node_bindings = build_node_id_head_mapping(active_heads, node_ids)
    rule_reports: dict[str, dict[str, Any]] = {}
    graph_loss_node_ids: set[str] = set()
    for rule in rules:
        rule_id = str(rule.get("rule_id", "")).strip()
        source_id = str(rule.get("source_node_id", "")).strip()
        target_id = str(rule.get("target_node_id", "")).strip()
        source_node = node_lookup.get(source_id)
        target_node = node_lookup.get(target_id)
        source_binding = node_bindings.get(source_id)
        target_binding = node_bindings.get(target_id)
        source_head = str(source_binding["head_name"]) if source_binding is not None else None
        target_head = str(target_binding["head_name"]) if target_binding is not None else None
        rule_type = str(rule.get("rule_type", ""))
        reason_counts: Counter[str] = Counter()
        matched = 0
        if source_node is None or target_node is None:
            reason_counts["missing_rule_endpoint"] += len(sidecar_rows)
        elif source_binding is None and target_binding is None:
            reason_counts["source_and_target_head_unmapped_or_inactive"] += len(sidecar_rows)
        elif source_binding is None:
            reason_counts["source_head_unmapped_or_inactive"] += len(sidecar_rows)
        elif target_binding is None:
            reason_counts["target_head_unmapped_or_inactive"] += len(sidecar_rows)
        elif rule_type not in SUPPORTED_RULE_TYPES:
            reason_counts["unsupported_rule_type"] += len(sidecar_rows)
        else:
            for row in sidecar_rows:
                source_confirmed = _row_has_confirmed_head(
                    row=row,
                    head_name=str(source_binding["head_name"]),
                    node_lookup=node_lookup,
                    node_bindings=node_bindings,
                )
                target_confirmed = _row_has_confirmed_head(
                    row=row,
                    head_name=str(target_binding["head_name"]),
                    node_lookup=node_lookup,
                    node_bindings=node_bindings,
                )
                if source_head == target_head:
                    if source_confirmed or target_confirmed:
                        matched += 1
                    else:
                        reason_counts["no_confirmed_observed_mask_overlap"] += 1
                    continue
                if source_confirmed and target_confirmed:
                    matched += 1
                    continue
                if not source_confirmed and not target_confirmed:
                    reason_counts["source_and_target_unconfirmed_or_missing"] += 1
                elif not source_confirmed:
                    reason_counts["source_unconfirmed_or_missing"] += 1
                else:
                    reason_counts["target_unconfirmed_or_missing"] += 1

        heads_active = source_binding is not None and target_binding is not None
        can_contribute = matched > 0 and heads_active and rule_type in SUPPORTED_RULE_TYPES
        if can_contribute:
            graph_loss_node_ids.update({source_id, target_id})
        skipped = max(0, len(sidecar_rows) - matched)
        rule_reports[rule_id] = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "matched_sample_count": int(matched),
            "skipped_count": int(skipped),
            "skipped_reason_counts": dict(sorted(reason_counts.items())),
            "source_concept_head": source_head,
            "target_concept_head": target_head,
            "source_binding": source_binding,
            "target_binding": target_binding,
            "can_contribute_loss": bool(can_contribute),
        }

    for node_id in graph_loss_node_ids:
        if node_id in node_reports and node_reports[node_id]["loss_role"] != "concept_target_loss":
            node_reports[node_id]["loss_role"] = "graph_consistency_regularizer"

    activation_by_modality = {
        modality: {
            "active_graph_node_count": int(sum(1 for value in counts.values() if value > 0)),
            "confirmed_node_counts": dict(sorted(counts.items())),
        }
        for modality, counts in sorted(by_modality.items())
    }
    concept_head_decision = {
        head: {
            "activation_decision": (
                "active_confirmed_supervised"
                if any(
                    report["concept_head"] == head
                    and report["confirmed"] > 0
                    and float(concept_weights.get(head, 0.0)) > 0.0
                    for report in node_reports.values()
                )
                else "configured_but_no_confirmed_supervised_nodes"
            ),
            "loss_weight": float(concept_weights.get(head, 0.0)),
        }
        for head in sorted(active_heads)
    }

    return {
        "schema_version": "stage1_graph_activation_audit_v1",
        "graph_activation_audit_status": audit_status,
        "resolved_config_path": str(resolved_config_path),
        "graph_paths": {
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "mapping_rules_path": str(mapping_rules_path),
            "consistency_rules_path": str(consistency_rules_path),
            "prompt_templates_path": str(prompt_templates_path),
            "sidecar_case_concept_vector_path": str(sidecar_case_concept_vector_path)
            if sidecar_case_concept_vector_path is not None
            else None,
        },
        "sample_count": len(sidecar_rows),
        "active_concept_heads": sorted(active_heads),
        "pending_concept_heads": sorted(pending_heads),
        "concept_label_fields": _concept_label_fields(image_manifest_path),
        "graph_node_status_counts": dict(sorted(aggregate_counts.items())),
        "active_graph_node_count": int(
            sum(1 for report in node_reports.values() if int(report["confirmed"]) > 0)
        ),
        "graph_node_activation_by_modality": activation_by_modality,
        "concept_head_activation_decision": concept_head_decision,
        "nodes": node_reports,
        "rules": rule_reports,
        "rule_summary": {
            "graph_rule_term_count": int(sum(1 for report in rule_reports.values() if report["can_contribute_loss"])),
            "graph_rule_skipped_count": int(sum(report["skipped_count"] for report in rule_reports.values())),
            "graph_rule_skipped_reason_counts": dict(
                sorted(
                    sum((Counter(report["skipped_reason_counts"]) for report in rule_reports.values()), Counter()).items()
                )
            ),
        },
        "v5_2_policy": {
            "stage1_only_formal_pretraining": True,
            "stage0_gaze_prior_construction_only": True,
            "clinical_graph_stage1_side_constraint_only": True,
            "no_independent_stage2_graph_pretraining": True,
            "no_image_region_to_graph_node_direct_alignment": True,
            "diffeye_gaze_claim": "weak spatial attention prior, not physician gaze ground truth",
            "missing_raw_unconfirmed_not_applicable_are_not_negative": True,
        },
    }


def build_graph_activation_audit_from_resolved_config(
    resolved_config_path: Path,
    *,
    require_sidecar: bool = True,
) -> dict[str, Any] | None:
    resolved = Path(resolved_config_path).expanduser().resolve()
    if not resolved.is_file():
        if require_sidecar:
            raise FileNotFoundError(f"Resolved config not found: {resolved}")
        return None
    config = _load_resolved_config(resolved)
    paths = _clinical_graph_paths(config, PROJECT_ROOT)
    required_keys = (
        "nodes_path",
        "edges_path",
        "mapping_rules_path",
        "consistency_rules_path",
        "prompt_templates_path",
    )
    if any(paths[key] is None or not Path(paths[key]).is_file() for key in required_keys):
        if require_sidecar:
            missing = [key for key in required_keys if paths[key] is None or not Path(paths[key]).is_file()]
            raise FileNotFoundError(f"Missing clinical graph config path(s): {missing}")
        return None
    sidecar = paths["sidecar_case_concept_vector_path"]
    if sidecar is not None and not Path(sidecar).is_file() and require_sidecar:
        raise FileNotFoundError(f"Missing tri_modal_case_concept_vector.jsonl: {sidecar}")
    return build_graph_activation_audit(
        resolved_config_path=resolved,
        nodes_path=Path(paths["nodes_path"]),
        edges_path=Path(paths["edges_path"]),
        mapping_rules_path=Path(paths["mapping_rules_path"]),
        consistency_rules_path=Path(paths["consistency_rules_path"]),
        prompt_templates_path=Path(paths["prompt_templates_path"]),
        sidecar_case_concept_vector_path=Path(sidecar) if sidecar is not None else None,
        require_sidecar=require_sidecar,
    )


__all__ = [
    "build_graph_activation_audit",
    "build_graph_activation_audit_from_resolved_config",
    "build_node_id_head_mapping",
    "graph_node_to_concept_binding",
    "graph_node_to_concept_head",
]
