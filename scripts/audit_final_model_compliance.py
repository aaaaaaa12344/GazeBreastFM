from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breast_pretrain.configs import load_stage1_joint_pretrain_bundle  # noqa: E402
from audit_clinical_graph_sidecar import (  # noqa: E402
    audit_graph_consistency_integration,
    audit_modality_embedding_wiring,
    audit_no_fixture_in_final,
    audit_sidecar_existence_and_coverage,
)
from audit_matched_control_compliance import (  # noqa: E402
    audit_m2_m3_match,
    audit_mri_us,
)
from audit_final_backbone_bundle_gate import (  # noqa: E402
    FINAL_OR_TEMPLATE_TIERS,
    FINAL_TEMPLATE_TIERS,
    FINAL_TIERS,
    FORMAL_PRODUCTION_TIERS,
    _is_final,
    _is_formal_production,
    _metadata_value,
    _read_yaml,
    audit_backbone_bundle_gate,
)
from audit_final_patient_split_gate import audit_patient_split_gate  # noqa: E402
from audit_final_graph_encoder_gate import (  # noqa: E402
    audit_clinical_graph_gate,
    audit_graph_encoder_gate,
)
from audit_final_adaptive_masking_gate import audit_adaptive_masking_gate  # noqa: E402
from audit_final_runtime_boundary_gate import (  # noqa: E402
    audit_codebase_boundary_gate,
    audit_evaluation_config_gate,
)
from audit_dynamic_loss_weighting_compliance import audit_dynamic_loss_weighting  # noqa: E402
from breast_pretrain.train.stage1_joint.config_validation import FORMAL_SOURCE_INTENSITY_TOLERANCE  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_model_compliance"
REPORT_JSON = OUTPUT_DIR / "final_model_compliance_report.json"
REPORT_MD = OUTPUT_DIR / "final_model_compliance_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final V5.2 model compliance for Stage 1 configs.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 config to audit.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _audit_source_intensity_requirement(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    resolved_config_path: Path,
) -> None:
    """For Mammo-FM formal_production configs, source_intensity_audit.required must be true."""
    model_block = raw_config.get("model") if isinstance(raw_config.get("model"), dict) else {}
    backbone = str(model_block.get("vision_encoder_name", "")).strip()

    if "mammo_fm_timm_efficientnet_b5" not in backbone:
        return

    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    is_formal = run_tier in FORMAL_PRODUCTION_TIERS or run_tier in FINAL_TIERS

    audit_block = raw_config.get("source_intensity_audit")
    required = bool(audit_block.get("required", False)) if isinstance(audit_block, dict) else False

    if is_formal and not required:
        _add(
            checks,
            "source_intensity_audit_required",
            "fail",
            "Mammo-FM formal_production config must set source_intensity_audit.required=true. "
            "This cannot be bypassed by changing run_tier or removing known_limitations.",
            "hard",
        )
        return

    if required:
        approved = float(audit_block.get("approved_tolerance", -1)) if isinstance(audit_block, dict) else 0.0
        if not math.isfinite(approved) or approved <= 0:
            _add(
                checks,
                "source_intensity_audit_approved_tolerance",
                "fail",
                f"source_intensity_audit.approved_tolerance={approved} is invalid.",
                "hard",
            )
        elif approved != FORMAL_SOURCE_INTENSITY_TOLERANCE:
            _add(
                checks,
                "source_intensity_audit_approved_tolerance",
                "fail",
                "source_intensity_audit.approved_tolerance must be exactly "
                f"{FORMAL_SOURCE_INTENSITY_TOLERANCE:g} for the formal run, got {approved:g}.",
                "hard",
            )
        else:
            _add(
                checks,
                "source_intensity_audit_required",
                "pass",
                f"source_intensity_audit.required=true, approved_tolerance={approved}.",
                "info",
            )


def _resolve_config_path(raw_config: dict[str, Any], value: Any, config_path: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    project_root_raw = str(raw_config.get("project_root", "")).strip()
    if project_root_raw:
        project_root = Path(project_root_raw)
        if not project_root.is_absolute():
            project_root = (config_path.parent / project_root).resolve()
    else:
        project_root = config_path.parent
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path.resolve()


def _audit_formal_resolved_path_consistency(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    resolved_config_path: Path,
) -> None:
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if run_tier not in FORMAL_PRODUCTION_TIERS:
        return

    model = raw_config.get("model") if isinstance(raw_config.get("model"), dict) else {}
    formal_backbone = _resolve_config_path(
        raw_config,
        raw_config.get("formal_backbone_weight_path"),
        resolved_config_path,
    )
    runtime_backbone = _resolve_config_path(
        raw_config,
        model.get("pretrained_weight_path", model.get("pretrained_model_path")),
        resolved_config_path,
    )
    if formal_backbone is None:
        _add(checks, "formal_backbone_weight_path_present", "fail", "Resolved formal config is missing formal_backbone_weight_path.", "hard")
    elif runtime_backbone != formal_backbone:
        _add(
            checks,
            "formal_backbone_path_consistency",
            "fail",
            f"Runtime pretrained_weight_path {runtime_backbone} != formal_backbone_weight_path {formal_backbone}.",
            "hard",
        )
    else:
        _add(checks, "formal_backbone_path_consistency", "pass", str(formal_backbone), "info")

    bundle_root = _resolve_config_path(raw_config, raw_config.get("formal_bundle_root"), resolved_config_path)
    if bundle_root is None:
        _add(checks, "formal_bundle_root_present", "fail", "Resolved formal config is missing formal_bundle_root.", "hard")
        return

    bundle_keys = {
        "image_manifest_path": raw_config.get("image_manifest_path"),
        "text_prompt_path": raw_config.get("text_prompt_path"),
        "semantic.prompt_embedding_path": (raw_config.get("semantic") or {}).get("prompt_embedding_path"),
        "semantic.semantic_soft_label_path": (raw_config.get("semantic") or {}).get("semantic_soft_label_path"),
        "semantic.semantic_soft_label_topk_path": (raw_config.get("semantic") or {}).get("semantic_soft_label_topk_path"),
        "semantic.semantic_manifest_path": (raw_config.get("semantic") or {}).get("semantic_manifest_path"),
        "semantic.birads_prior_manifest_path": (raw_config.get("semantic") or {}).get("birads_prior_manifest_path"),
        "clinical_graph.sidecar_case_concept_vector_path": (raw_config.get("clinical_graph") or {}).get("sidecar_case_concept_vector_path"),
    }
    mismatches: list[str] = []
    for key, value in bundle_keys.items():
        resolved = _resolve_config_path(raw_config, value, resolved_config_path)
        if resolved is None:
            continue
        try:
            resolved.relative_to(bundle_root)
        except ValueError:
            mismatches.append(f"{key}={resolved}")
    if mismatches:
        _add(
            checks,
            "formal_bundle_path_consistency",
            "fail",
            "Formal bundle paths must resolve under formal_bundle_root. " + "; ".join(mismatches[:8]),
            "hard",
        )
    else:
        _add(checks, "formal_bundle_path_consistency", "pass", str(bundle_root), "info")


def build_report(config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.expanduser().resolve()
    raw_config = _read_yaml(resolved_config_path)
    checks: list[dict[str, Any]] = []

    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    contract_version = str(raw_config.get("entry_contract_version", "")).strip().lower()
    if run_tier == "formal_production" and contract_version == "v6":
        # The legacy V5 bundle audit requires manifest_stage1_semantic.csv and
        # performs row-wise remote-path stats. Formal V6 is instead closed by
        # its exact receipt-only authorization and PASS_FORMAL parent lineage.
        from breast_pretrain.train.stage1_joint.formal_v6_entry_gate import assert_v6_formal_entry

        authorization = assert_v6_formal_entry(resolved_config_path)
        authorities = authorization.get("bound_authorities", {}) if isinstance(authorization, dict) else {}
        _add(checks, "formal_v6_entry", "pass", "V6 entry and authorization bindings verified.", "hard")
        _add(
            checks,
            "formal_v6_parent_pass_formal",
            "pass" if authorities.get("final_entry_status") == "PASS_FORMAL" else "fail",
            f"final_entry_status={authorities.get('final_entry_status')!r}",
            "hard",
        )
        failures = [item for item in checks if item["status"] == "fail"]
        return {
            "schema_version": "final_model_compliance_v2",
            "config_path": str(resolved_config_path),
            "status": "fail" if failures else "pass",
            "hard_failure_count": len(failures),
            "blocking_warning_count": 0,
            "warning_count": 0,
            "is_final_config": True,
            "bundle_validation": {
                "status": "PASS_FORMAL" if not failures else "FAIL",
                "authority": "RECEIPT_ONLY_AUTHORIZATION_REBIND_V1",
                "formal_universe_rows": authorities.get("formal_universe_rows"),
                "final_bundle_sha256": authorities.get("final_bundle_sha256"),
            },
            "matched_control": {"status": "FROZEN_PARENT_AUTHORITY"},
            "checks": checks,
        }

    # Gate 1: Backbone + Bundle validation
    bundle = load_stage1_joint_pretrain_bundle(resolved_config_path)
    bundle_report = audit_backbone_bundle_gate(raw_config, bundle, checks, config_path=resolved_config_path)

    # Gate 2: Patient split enforcement
    audit_patient_split_gate(raw_config, checks, bundle, resolved_config_path)

    # Gate 3: M2/M3 matched control and MRI/US validation
    matched_control = audit_m2_m3_match(raw_config, resolved_config_path, checks)
    audit_mri_us(raw_config, bundle_report, checks)

    # Gate 4: Clinical graph block validation
    audit_clinical_graph_gate(raw_config, checks, resolved_config_path)
    audit_sidecar_existence_and_coverage(raw_config, resolved_config_path, checks)
    audit_graph_consistency_integration(raw_config, checks)
    audit_modality_embedding_wiring(raw_config, checks)
    audit_no_fixture_in_final(raw_config, checks)

    # Gate 5: Graph encoder vs sidecar-only (with no_graph_ablation exemption)
    audit_graph_encoder_gate(raw_config, checks, resolved_config_path)

    # Gate 6: Adaptive masking enforcement
    audit_adaptive_masking_gate(raw_config, checks, resolved_config_path)

    # Gate 7: Evaluation config and codebase boundary
    audit_evaluation_config_gate(raw_config, checks)
    audit_codebase_boundary_gate(checks)

    # Gate 8: Dynamic loss weighting compliance
    audit_dynamic_loss_weighting(raw_config, checks, resolved_config_path)

    # Gate 9: Source intensity audit compliance (formal_production + mammo_fm)
    _audit_source_intensity_requirement(raw_config, checks, resolved_config_path)
    _audit_formal_resolved_path_consistency(raw_config, checks, resolved_config_path)

    # Tier-downgrade: final_config_template structural hard failures → blocking_warning
    # formal_production: structural checks stay HARD; no downgrade.
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if run_tier in FINAL_TEMPLATE_TIERS:
        _structural_ids = {
            "final_no_minimal_patch_encoder",
            "final_no_wrapper_backend",
            "final_pretrained_backend",
            "final_pretrained_weight_exists",
            "clinical_graph_block_present",
            "clinical_graph_version",
            "clinical_graph_nodes_path_present",
            "clinical_graph_nodes_path_exists",
            "clinical_graph_edges_path_present",
            "clinical_graph_edges_path_exists",
            "clinical_graph_mapping_rules_path_present",
            "clinical_graph_mapping_rules_path_exists",
            "clinical_graph_consistency_rules_path_present",
            "clinical_graph_consistency_rules_path_exists",
            "clinical_graph_prompt_templates_path_present",
            "clinical_graph_prompt_templates_path_exists",
            "graph_encoder_not_sidecar_only",
            "evaluation_layer_config_present",
            "evaluation_layer_entrypoint_present",
            "no_fixture_backbone_in_formal",
            "no_fixture_bundle_in_formal",
            "no_allow_missing_pretrained_fallback",
            "patient_split_enforcement",
            "stage1_bundle_contract",
            "trainer_reads_stage1_bundle",
            "gaze_supervision_source_allowed",
            "code_size_gate",
            "clinical_graph_no_active_stage2_graph_pretraining",
            "m3_pass193_no_usable189_bundle",
            "dw_code_fn",
            "dw_code_total_loss",
            "dw_code_metrics",
            "dw_config_gaze_enabled",
            "dw_config_targets",
            "dw_config_weight_bounds",
            "source_intensity_audit_required",
        }
        for item in checks:
            if item["status"] == "fail" and item["severity"] == "hard":
                if item["id"] in _structural_ids:
                    item["severity"] = "blocking_warning"
                    item["message"] = f"[final_config_template:blocking] {item['message']}"
                else:
                    item["severity"] = "warn"
                    item["message"] = f"[final_config_template:hard→warn] {item['message']}"

    failures = [item for item in checks if item["status"] == "fail" and item["severity"] == "hard"]
    blocking = [item for item in checks if item["severity"] == "blocking_warning"]
    warnings = [
        item for item in checks
        if item["status"] == "warn" or item["severity"] == "warn"
    ]

    if run_tier in FINAL_TEMPLATE_TIERS and blocking:
        overall = "blocked_template"
    elif failures:
        overall = "fail"
    elif warnings:
        overall = "pass_with_warnings"
    else:
        overall = "pass"

    return {
        "schema_version": "final_model_compliance_v2",
        "config_path": str(resolved_config_path),
        "status": overall,
        "hard_failure_count": len(failures),
        "blocking_warning_count": len(blocking),
        "warning_count": len(warnings),
        "is_final_config": _is_final(raw_config),
        "bundle_validation": bundle_report,
        "matched_control": matched_control,
        "checks": checks,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON.name
    md_path = output_dir / REPORT_MD.name
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    blocking_count = report.get("blocking_warning_count", 0)
    lines = [
        "# Final Model Compliance Report",
        "",
        f"Config: `{report['config_path']}`",
        f"Status: `{report['status']}`",
        f"Hard failures: `{report['hard_failure_count']}`",
        f"Blocking warnings: `{blocking_count}`",
        f"Warnings: `{report['warning_count']}`",
        "",
        "## Checks",
        "| Check | Status | Severity | Message |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["checks"]:
        message = str(item["message"]).replace("|", "\\|")
        lines.append(f"| `{item['id']}` | `{item['status']}` | `{item['severity']}` | {message} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    report = build_report(args.config)
    json_path, md_path = write_report(report, args.output_dir)
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True))
    if report["status"] in ("fail", "blocked_template"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
