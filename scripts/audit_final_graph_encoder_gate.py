from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_TEMPLATE_TIERS = {"final_config_template"}
FINAL_OR_TEMPLATE_TIERS = {"final", "final_config_template"}


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _metadata_value(raw_config: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = raw_config.get("metadata") if isinstance(raw_config.get("metadata"), dict) else {}
    return raw_config.get(key, metadata.get(key, default))


def _resolve(raw_value: Any, base_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _severity_for_tier(raw_config: dict[str, Any], hard_severity: str = "hard") -> str:
    tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if tier in {"final"}:
        return hard_severity
    if tier in FINAL_TEMPLATE_TIERS:
        return "warn" if hard_severity == "hard" else hard_severity
    return "info"


def audit_clinical_graph_gate(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Validate clinical_graph block structure and completeness."""
    cg_block = raw_config.get("clinical_graph")
    if cg_block is None or not isinstance(cg_block, dict):
        _add(checks, "clinical_graph_block_present", "fail", "Final M3/M4 config must have clinical_graph block.", "hard")
        return

    _add(checks, "clinical_graph_block_present", "pass", "clinical_graph block present.", "info")

    project_root_raw = raw_config.get("project_root", ".")
    proj_root = (config_path.parent / Path(str(project_root_raw))).resolve()

    version = str(cg_block.get("version", ""))
    if version != "tri_modal_clinical_graph_v1":
        _add(checks, "clinical_graph_version", "fail", f"clinical_graph.version must be tri_modal_clinical_graph_v1, got {version!r}.", "hard")
    else:
        _add(checks, "clinical_graph_version", "pass", f"version={version}", "info")

    required_paths = [
        ("nodes_path", "nodes CSV"),
        ("edges_path", "edges CSV"),
        ("mapping_rules_path", "mapping rules YAML"),
        ("consistency_rules_path", "consistency rules YAML"),
        ("prompt_templates_path", "prompt templates YAML"),
    ]
    for path_key, label in required_paths:
        raw_path = cg_block.get(path_key)
        if raw_path is None:
            _add(checks, f"clinical_graph_{path_key}_present", "fail", f"clinical_graph.{path_key} is required.", "hard")
        else:
            resolved = (proj_root / str(raw_path)).resolve()
            if not resolved.exists():
                _add(checks, f"clinical_graph_{path_key}_exists", "fail", f"clinical_graph.{path_key} not found: {resolved}", "hard")
            else:
                _add(checks, f"clinical_graph_{path_key}_exists", "pass", str(raw_path), "info")

    for flag_key in ("use_for_structured_prompt", "use_for_semantic_soft_labels", "use_for_concept_targets", "use_for_concept_consistency"):
        val = cg_block.get(flag_key)
        if val is not True:
            _add(checks, f"clinical_graph_{flag_key}", "fail", f"clinical_graph.{flag_key} must be true.", "hard")
        else:
            _add(checks, f"clinical_graph_{flag_key}", "pass", f"{flag_key}=true", "info")

    forbid = cg_block.get("forbid_direct_graph_node_alignment")
    if forbid is not True:
        _add(checks, "clinical_graph_forbid_direct_graph_node_alignment", "fail", "clinical_graph.forbid_direct_graph_node_alignment must be true.", "hard")
    else:
        _add(checks, "clinical_graph_forbid_direct_graph_node_alignment", "pass", "forbid_direct_graph_node_alignment=true", "info")

    sidecar_path = cg_block.get("sidecar_case_concept_vector_path")
    if sidecar_path:
        _add(checks, "clinical_graph_sidecar_path_set", "pass", str(sidecar_path), "info")
    else:
        _add(checks, "clinical_graph_sidecar_path_set", "warn", "sidecar_case_concept_vector_path not set.", "warn")

    schema_script = PROJECT_ROOT / "scripts" / "validate_clinical_graph_schema.py"
    if schema_script.exists():
        _add(checks, "clinical_graph_schema_script_exists", "pass", "validate_clinical_graph_schema.py exists.", "info")
    else:
        _add(checks, "clinical_graph_schema_script_exists", "fail", "validate_clinical_graph_schema.py missing.", "hard")

    cg_scope = str(raw_config.get("semantic", {}).get("clinical_graph_scope", "")).strip()
    valid_scopes = {"stage1_semantic_prior_sidecar_only", "stage1_semantic_prior_and_graph_encoder"}
    if not cg_scope:
        _add(checks, "clinical_graph_scope_not_empty", "fail", "semantic.clinical_graph_scope is empty or missing.", "hard")
    elif cg_scope in valid_scopes:
        _add(checks, "clinical_graph_scope_not_empty", "pass", f"clinical_graph_scope={cg_scope}", "info")
    else:
        _add(checks, "clinical_graph_scope_not_empty", "warn", f"clinical_graph_scope={cg_scope!r} — expected one of {valid_scopes}.", "warn")


def audit_graph_encoder_gate(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Hard-fail if M3/M4 final uses sidecar-only clinical_graph without graph_encoder.
    Exempts no_graph_ablation configs (graph_encoder.enabled=false is intentional there)."""
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if run_tier not in FINAL_OR_TEMPLATE_TIERS:
        _add(checks, "graph_encoder_not_sidecar_only", "pass",
             "Not a final or final_config_template config; sidecar-only check skipped.", "info")
        return

    model_role = str(_metadata_value(raw_config, "model_role", "")).strip().lower()
    config_name = config_path.name.lower() if config_path else ""
    is_m3_m4 = (
        "m3" in model_role or "m4" in model_role
        or config_name.startswith("m3_") or config_name.startswith("m4_")
    )
    if not is_m3_m4:
        _add(checks, "graph_encoder_not_sidecar_only", "pass",
             "Not M3/M4; sidecar-only check skipped.", "info")
        return

    # ── no_graph_ablation exemption ──
    is_no_graph_ablation = "no_graph_ablation" in config_name or "no_graph_ablation" in model_role
    cg_block = raw_config.get("clinical_graph")
    ge_block = raw_config.get("graph_encoder")
    has_clinical_graph = cg_block is not None and isinstance(cg_block, dict)
    has_graph_encoder = (
        ge_block is not None
        and isinstance(ge_block, dict)
        and ge_block.get("enabled") is True
    )

    if is_no_graph_ablation and has_clinical_graph and not has_graph_encoder:
        # Verify the ablation is correctly configured
        gc_weight = float(raw_config.get("losses", {}).get("graph_consistency_weight", 0.0))
        forbid = cg_block.get("forbid_direct_graph_node_alignment")
        if gc_weight != 0.0:
            _add(checks, "graph_encoder_not_sidecar_only", "fail",
                 "no_graph_ablation config has graph_consistency_weight != 0. "
                 "Set graph_consistency_weight=0 for valid no-graph ablation.", "hard")
        elif forbid is not True:
            _add(checks, "graph_encoder_not_sidecar_only", "fail",
                 "no_graph_ablation config must have forbid_direct_graph_node_alignment=true.", "hard")
        else:
            _add(checks, "graph_encoder_not_sidecar_only", "pass",
                 "no_graph_ablation config accepted: graph_encoder.enabled=false, "
                 "graph_consistency_weight=0, forbid_direct_graph_node_alignment=true. "
                 "This is an ablation, not a full model.", "info")
        return

    severity = _severity_for_tier(raw_config, "hard")
    if has_clinical_graph and not has_graph_encoder:
        _add(checks, "graph_encoder_not_sidecar_only", "fail",
             "M3/M4 has clinical_graph enabled but graph_encoder.enabled is missing or false. "
             "graph_encoder must be enabled — sidecar-only clinical graph is not acceptable.",
             severity)
    elif has_clinical_graph and has_graph_encoder:
        _add(checks, "graph_encoder_not_sidecar_only", "pass",
             "graph_encoder.enabled=true alongside clinical_graph.", "info")
    else:
        _add(checks, "graph_encoder_not_sidecar_only", "fail",
             "M3/M4 is missing clinical_graph and/or graph_encoder blocks.", severity)

    if run_tier in FINAL_TEMPLATE_TIERS:
        cs = str(_metadata_value(raw_config, "compliance_status", "")).strip()
        if cs != "fixture_references_must_be_replaced_before_training":
            _add(checks, "final_config_template_compliance_status", "fail",
                 f"final_config_template must have "
                 f"compliance_status=fixture_references_must_be_replaced_before_training, "
                 f"got {cs!r}.", "warn")
