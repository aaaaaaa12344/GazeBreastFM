from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def audit_adaptive_masking_gate(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """M3/M4 gaze-enabled configs must use adaptive_gaze_masking, never mixed_random_gaze.
    M2/M4 no-gaze configs are explicitly allowed to use random masking."""
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if run_tier not in FINAL_OR_TEMPLATE_TIERS:
        _add(checks, "adaptive_masking_gate", "pass",
             "Not a final or final_config_template config; adaptive masking gate skipped.", "info")
        return

    model_role = str(_metadata_value(raw_config, "model_role", "")).strip().lower()
    config_name = config_path.name.lower() if config_path else ""
    is_m3_m4 = (
        "m3" in model_role or "m4" in model_role
        or config_name.startswith("m3_") or config_name.startswith("m4_")
    )
    if not is_m3_m4:
        _add(checks, "adaptive_masking_gate", "pass",
             "Not M3/M4; adaptive masking gate skipped.", "info")
        return

    masking_block = raw_config.get("masking") or {}
    mask_strategy = str(masking_block.get("mask_strategy", "")).strip()
    gaze_loss_mode = str(masking_block.get("gaze_loss_mode", "")).strip().lower()
    severity = _severity_for_tier(raw_config, "hard")

    # M2/M4 no-gaze and ablation configs explicitly allowed to use random masking
    is_gaze_disabled = gaze_loss_mode in {"no_gaze", ""} or not gaze_loss_mode
    if is_gaze_disabled:
        is_nogaze_config = (
            config_name.startswith("m2_")
            or "no_gaze" in config_name
            or "no_graph_ablation" in config_name
            or "semantic_only" in config_name
            or (config_name.startswith("m4_") and "no_gaze" in config_name)
            or (config_name.startswith("m4_") and "no_graph_ablation" in config_name)
        )
        if is_nogaze_config:
            _add(checks, "adaptive_masking_gate", "pass",
                 f"M2/M4 no-gaze/ablation config with mask_strategy={mask_strategy!r} is allowed.", "info")
            return

    if mask_strategy == "mixed_random_gaze":
        _add(checks, "adaptive_masking_gate_no_mixed_random_gaze", "fail",
             f"M3/M4 gaze-enabled config must not use mask_strategy=mixed_random_gaze. "
             f"Current: {mask_strategy!r}. Use adaptive_gaze_masking.",
             severity)
        return

    if mask_strategy != "adaptive_gaze_masking":
        _add(checks, "adaptive_masking_gate_no_mixed_random_gaze", "fail",
             f"M3/M4 gaze-enabled config must use mask_strategy=adaptive_gaze_masking. "
             f"Current: {mask_strategy!r}.",
             severity)
        return

    _add(checks, "adaptive_masking_gate_no_mixed_random_gaze", "pass",
         f"mask_strategy=adaptive_gaze_masking for M3/M4.", "info")

    # Summary metrics verification
    output_dir = _resolve(raw_config.get("output_dir"), Path("."))
    if output_dir is not None:
        summary_path = output_dir / "stage1_joint_train_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                required_fields = [
                    "mask_policy_used",
                    "adaptive_gaze_quota",
                    "adaptive_random_fraction",
                    "fallback_reason",
                    "masked_gaze_coverage",
                    "visible_gaze_coverage",
                ]
                missing_fields = [f for f in required_fields if f not in summary]
                if missing_fields:
                    _add(checks, "adaptive_masking_summary_fields", "warn",
                         f"Summary missing top-level fields: {missing_fields}. "
                         "Run training to populate.", "warn")
                else:
                    _add(checks, "adaptive_masking_summary_fields", "pass",
                         f"mask_policy_used={summary.get('mask_policy_used')} "
                         f"adaptive_gaze_quota={summary.get('adaptive_gaze_quota')} "
                         f"adaptive_random_fraction={summary.get('adaptive_random_fraction')} "
                         f"masked_gaze_coverage={summary.get('masked_gaze_coverage')} "
                         f"visible_gaze_coverage={summary.get('visible_gaze_coverage')}",
                         "info")
                fallback_entries = summary.get("fallback_reason", {})
                if fallback_entries:
                    _add(checks, "adaptive_masking_fallback_present", "pass",
                         f"fallback_reason distribution: {fallback_entries}", "info")
                else:
                    _add(checks, "adaptive_masking_fallback_present", "warn",
                         "No fallback_reason in summary. Run training to populate.", "warn")
            except (json.JSONDecodeError, OSError):
                _add(checks, "adaptive_masking_summary_fields", "warn",
                     "Could not read summary; top-level fields unverified.", "warn")
        else:
            _add(checks, "adaptive_masking_summary_fields", "warn",
                 f"Summary not found at {summary_path}. Run training to populate.", "warn")
