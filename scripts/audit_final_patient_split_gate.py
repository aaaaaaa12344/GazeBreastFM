from __future__ import annotations

from pathlib import Path
from typing import Any

from breast_pretrain.data.patient_split_audit import audit_patient_split

FINAL_TIERS = {"final"}
FINAL_TEMPLATE_TIERS = {"final_config_template"}


def _metadata_value(raw_config: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = raw_config.get("metadata") if isinstance(raw_config.get("metadata"), dict) else {}
    return raw_config.get(key, metadata.get(key, default))


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def audit_patient_split_gate(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    bundle: Any,
    config_path: Path,
) -> None:
    """Audit patient-level split enforcement on the Stage 1 manifest."""
    trainer = bundle.trainer
    manifest_path = trainer.data.image_manifest_path
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    is_fixture = "final_compliance_fixtures" in str(manifest_path).replace("\\", "/")
    is_template = run_tier in FINAL_TEMPLATE_TIERS

    if not manifest_path.exists():
        msg = f"Manifest not found for patient split audit: {manifest_path}"
        if run_tier in FINAL_TIERS:
            _add(checks, "patient_split_audit", "fail", msg, "hard")
        elif is_template:
            _add(checks, "patient_split_audit", "blocking_warning", msg, "blocking_warning")
        else:
            _add(checks, "patient_split_audit", "warn", msg, "warn")
        return

    result = audit_patient_split(
        manifest_path,
        run_tier=run_tier,
        is_fixture=is_fixture,
        is_final_config_template=is_template,
    )

    for c in result.checks:
        check_id = f"patient_split_{c['id']}"
        if c["status"] == "fail":
            severity = "hard" if run_tier in FINAL_TIERS else "blocking_warning" if is_template else "warn"
            _add(checks, check_id, "fail", c["message"], severity)
        elif c["status"] == "blocked_template":
            _add(checks, check_id, "blocking_warning", c["message"], "blocking_warning")
        elif c["status"] == "warn":
            _add(checks, check_id, "warn", c["message"], "warn")
        else:
            _add(checks, check_id, "pass", c["message"], "info")

    # Summary check
    if result.status == "fail":
        _add(checks, "patient_split_enforcement", "fail",
             f"Patient split audit FAILED: {result.leaking_patient_count} leaking patients, "
             f"{result.leaking_breast_side_count} leaking breast sides, "
             f"{result.leaking_study_count} leaking studies. "
             f"Errors: {result.errors}", "hard")
    elif result.status == "blocked_template":
        _add(checks, "patient_split_enforcement", "blocking_warning",
             f"Patient split audit BLOCKED (template/fixture): {result.warnings[:3]}",
             "blocking_warning")
    elif result.status == "pass_with_warnings":
        _add(checks, "patient_split_enforcement", "pass",
             f"Patient split audit passed with warnings: {result.warnings[:3]}", "info")
    else:
        _add(checks, "patient_split_enforcement", "pass",
             f"Patient split audit passed: {result.patient_count} patients, "
             f"{result.total_rows} rows, no leakage.", "info")
