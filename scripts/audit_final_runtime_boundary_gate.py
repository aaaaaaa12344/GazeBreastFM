from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INACTIVE_STAGE2_CONTEXT_MARKERS = (
    "forbid",
    "forbidden",
    "do_not_claim",
    "not ",
    "no ",
    " no ",
    "no_",
    "no-",
    "false",
    "disabled",
)


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _resolve(raw_value: Any, base_dir: Path = PROJECT_ROOT) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _has_active_stage2_surface(
    text: str,
    forbidden_graph_terms: tuple[str, ...],
) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.lower()
        if not any(term in line for term in forbidden_graph_terms):
            continue
        if any(marker in line for marker in INACTIVE_STAGE2_CONTEXT_MARKERS):
            continue
        return True
    return False


def audit_evaluation_config_gate(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Validate evaluation config presence and entrypoint."""
    eval_config = _resolve(raw_config.get("evaluation_config"))
    if eval_config is None:
        _add(checks, "evaluation_layer_config_present", "fail", "Final config must point to an evaluation_config.", "hard")
    elif not eval_config.exists():
        _add(checks, "evaluation_layer_config_present", "fail", f"evaluation_config missing: {eval_config}", "hard")
    else:
        _add(checks, "evaluation_layer_config_present", "pass", str(eval_config), "info")
    script_path = PROJECT_ROOT / "scripts" / "evaluate_stage1_foundation_model.py"
    _add(
        checks,
        "evaluation_layer_entrypoint_present",
        "pass" if script_path.exists() else "fail",
        str(script_path),
        "hard" if not script_path.exists() else "info",
    )


def audit_codebase_boundary_gate(checks: list[dict[str, Any]]) -> None:
    """Enforce code size limits and scan for forbidden Stage 2 graph/region-text surfaces."""
    from audit_code_size import build_code_size_report  # noqa: E402

    code_size = build_code_size_report(max_warn_lines=800, max_fail_lines=1000)
    if code_size["status"] == "fail":
        _add(checks, "code_size_gate", "fail", "Files exceed 1000 lines: " + ", ".join(code_size["failing_files"]), "hard")
    else:
        _add(checks, "code_size_gate", "pass" if code_size["status"] == "pass" else "warn",
             f"code_size_status={code_size['status']}", "warn" if code_size["status"] != "pass" else "info")

    active_stage2_hits = []
    forbidden_graph_terms = (
        "stage2_graph", "region_text_alignment",
        "image_region_to_graph_node", "graph_node_alignment",
        "image_to_graph_node_contrastive", "region_node_alignment",
    )
    legacy_markers_variants = (
        "legacy/reference only", "legacy_ablation_only",
        "not_final_mainline", "legacy_ablation",
    )
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "configs"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".md"}:
                continue
            if path.name in {
                "audit_final_model_compliance.py",
                "audit_final_backbone_bundle_gate.py",
                "audit_final_patient_split_gate.py",
                "audit_final_graph_encoder_gate.py",
                "audit_final_adaptive_masking_gate.py",
                "audit_final_runtime_boundary_gate.py",
                "audit_repo_cleanup.py",
                "materialize_mri_stage1_primary_manifests.py",
                "audit_clinical_graph_sidecar.py",
                "validate_clinical_graph_schema.py",
                "build_tri_modal_clinical_graph_sidecars.py",
                "build_tri_modal_mini_stage1_bundle.py",
                "graph_tensor_builder.py",
                "semantic_prior.py",
                "audit_graph_encoder_compliance.py",
                "audit_adaptive_masking_compliance.py",
                "audit_experiment_matrix_compliance.py",
                "audit_m4_preflight_profiles.py",
                "rule_engine.py",
                "prompt_builder.py",
                "matrix_export.py",
                "audit_matched_control_compliance.py",
                "audit_code_size.py",
            }:
                continue
            if str(path).endswith("breast_multimodal_mapping_rules_v0.yaml"):
                continue
            if "tri_modal_clinical_graph_v1" in str(path):
                continue
            if "clinical_graph_encoder" in str(path):
                continue
            if "clinical_graph_sidecar" in str(path):
                continue
            if "stage1_joint" in str(path):
                continue
            if "m3_embed_aggregated_gaze_semantic" in str(path):
                continue
            if "m4_tri_modal_final" in str(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            has_forbidden = _has_active_stage2_surface(lowered, forbidden_graph_terms)
            has_legacy_marker = any(marker in lowered for marker in legacy_markers_variants)
            if has_forbidden and not has_legacy_marker:
                active_stage2_hits.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    if active_stage2_hits:
        _add(checks, "clinical_graph_no_active_stage2_graph_pretraining", "fail",
             "Unmarked Stage 2 graph/region-text surfaces: " + ", ".join(active_stage2_hits[:20]), "hard")
    else:
        _add(checks, "clinical_graph_no_active_stage2_graph_pretraining", "pass",
             "Stage 2 graph/region-text surfaces are legacy-marked or absent.", "info")
