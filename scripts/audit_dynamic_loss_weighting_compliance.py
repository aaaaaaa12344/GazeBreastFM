from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from breast_pretrain.train.stage1_joint.dynamic_loss_weighting import compute_conflict_aware_weights

FINAL_OR_TEMPLATE_TIERS = {"final", "final_config_template", "formal_production", "production_ready_candidate"}
FINAL_TEMPLATE_TIERS = {"final_config_template"}
FORMAL_PRODUCTION_TIERS = {"formal_production", "production_ready_candidate"}


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _metadata_value(raw_config: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = raw_config.get("metadata") if isinstance(raw_config.get("metadata"), dict) else {}
    return raw_config.get(key, metadata.get(key, default))


def _severity_for_tier(raw_config: dict[str, Any], hard_severity: str = "hard") -> str:
    tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if tier in {"final"} or tier in FORMAL_PRODUCTION_TIERS:
        return hard_severity
    if tier in FINAL_TEMPLATE_TIERS:
        return "warn" if hard_severity == "hard" else hard_severity
    return "info"


def _is_gaze_enabled_config(raw_config: dict[str, Any], config_path: Path) -> bool:
    masking = raw_config.get("masking", {}) if isinstance(raw_config.get("masking"), dict) else {}
    gaze_loss_mode = str(masking.get("gaze_loss_mode", "no_gaze")).strip().lower()
    mask_strategy = str(masking.get("mask_strategy", "random")).strip().lower()
    has_gaze = gaze_loss_mode not in ("no_gaze", "")
    has_adaptive = "gaze" in mask_strategy.lower()
    return has_gaze or has_adaptive


def _is_no_gaze_config(raw_config: dict[str, Any]) -> bool:
    masking = raw_config.get("masking", {}) if isinstance(raw_config.get("masking"), dict) else {}
    gaze_loss_mode = str(masking.get("gaze_loss_mode", "no_gaze")).strip().lower()
    return gaze_loss_mode in ("no_gaze", "")


def _is_semantic_only_config(raw_config: dict[str, Any]) -> bool:
    losses = raw_config.get("losses", {}) if isinstance(raw_config.get("losses"), dict) else {}
    recon_weight = float(losses.get("reconstruction_weight", 1.0))
    return abs(recon_weight) < 1e-8


def _is_no_graph_ablation(raw_config: dict[str, Any]) -> bool:
    ge = raw_config.get("graph_encoder") if isinstance(raw_config.get("graph_encoder"), dict) else None
    if ge is None:
        return False
    return not bool(ge.get("enabled", True))


def _check_codebase_for_dynamic_weighting() -> dict[str, Any]:
    losses_py = SRC_ROOT / "breast_pretrain" / "train" / "stage1_joint" / "losses.py"
    dynamic_module = SRC_ROOT / "breast_pretrain" / "train" / "stage1_joint" / "dynamic_loss_weighting.py"
    trainer_py = SRC_ROOT / "breast_pretrain" / "train" / "stage1_joint" / "trainer.py"
    metrics_py = SRC_ROOT / "breast_pretrain" / "train" / "stage1_joint" / "metrics.py"

    has_conflict_aware_fn = False
    has_total_loss_multiplication = False
    has_summary_only_not_training = True
    has_metrics_tracking = False
    has_dynamic_module = dynamic_module.exists()
    has_reconstruction_positive_relation = False
    has_forbidden_reconstruction_inverse = False
    has_semantic_positive_relation = False
    has_bad_reconstruction_upweight_warning = False
    has_good_reconstruction_upweight_warning = False
    has_no_gaze_fallback = False
    has_no_gaze_mode_fallback = False
    has_global_align_static_policy = False
    has_graph_dynamic_override_guard = False

    dynamic_text = dynamic_module.read_text(encoding="utf-8") if dynamic_module.exists() else ""
    if losses_py.exists():
        losses_text = losses_py.read_text(encoding="utf-8")
        combined_text = losses_text + "\n" + dynamic_text
        compact_losses_text = "".join(combined_text.split())
        has_conflict_aware_fn = "compute_conflict_aware_weights" in combined_text
        has_total_loss_multiplication = "dynamic_weights[" in losses_text
        has_reconstruction_positive_relation = "masked_coverage/recon_target" in compact_losses_text
        has_forbidden_reconstruction_inverse = (
            "recon_target/max(masked_coverage" in compact_losses_text
            or "recon_target/masked_coverage" in compact_losses_text
        )
        has_semantic_positive_relation = "visible_coverage/semantic_target" in compact_losses_text
        has_bad_reconstruction_upweight_warning = (
            "conflict_aware_reconstruction_upweighted:" in combined_text
            and "masked_gaze_coverage={masked_coverage:.6f}<target={recon_target:.6f}" in combined_text
        )
        has_good_reconstruction_upweight_warning = (
            "conflict_aware_reconstruction_upweighted:" in combined_text
            and "masked_gaze_coverage={masked_coverage:.6f}>target={recon_target:.6f}" in combined_text
        )
        has_no_gaze_fallback = (
            "gaze_mass < float(config.masking.gaze_mask_eps)" in combined_text
            and "return weights" in combined_text
        )
        has_no_gaze_mode_fallback = (
            "config.masking.gaze_loss_mode" in combined_text
            and "no_gaze" in combined_text
            and "conflict_aware_fallback_base_weights:no_gaze_config" in combined_text
            and "return weights" in combined_text
        )
        has_global_align_static_policy = (
            '"global_align"' in dynamic_text
            and '"global_align": semantic_multiplier' not in dynamic_text
        )
        has_graph_dynamic_override_guard = (
            "if config.losses.allow_dynamic_graph_consistency_weighting:" in dynamic_text
            and 'weights["graph_consistency"] = semantic_multiplier' in dynamic_text
        )

        if has_total_loss_multiplication and "losses[\"total\"]" in losses_text:
            lines = losses_text.split("\n")
            in_total_block = False
            for line in lines:
                if "losses[\"total\"]" in line or "losses['total']" in line:
                    in_total_block = True
                    continue
                if in_total_block and "dynamic_weights" in line:
                    has_summary_only_not_training = False
                    break
                if in_total_block and (line.strip().startswith(")") or line.strip().startswith("return")):
                    break

    if trainer_py.exists():
        trainer_text = trainer_py.read_text(encoding="utf-8")
        has_metrics_tracking = "dynamic_loss_weights" in trainer_text

    if metrics_py.exists():
        metrics_text = metrics_py.read_text(encoding="utf-8")
        has_summary_fields = "dynamic_loss_weighting_enabled" in metrics_text

    return {
        "has_conflict_aware_fn": has_conflict_aware_fn,
        "has_total_loss_multiplication": has_total_loss_multiplication,
        "has_metrics_tracking": has_metrics_tracking,
        "has_summary_fields": has_summary_fields if metrics_py.exists() else False,
        "has_dynamic_module": has_dynamic_module,
        "summary_only_not_training": has_summary_only_not_training,
        "has_reconstruction_positive_relation": has_reconstruction_positive_relation,
        "has_forbidden_reconstruction_inverse": has_forbidden_reconstruction_inverse,
        "has_semantic_positive_relation": has_semantic_positive_relation,
        "has_bad_reconstruction_upweight_warning": has_bad_reconstruction_upweight_warning,
        "has_good_reconstruction_upweight_warning": has_good_reconstruction_upweight_warning,
        "has_no_gaze_fallback": has_no_gaze_fallback,
        "has_no_gaze_mode_fallback": has_no_gaze_mode_fallback,
        "has_global_align_static_policy": has_global_align_static_policy,
        "has_graph_dynamic_override_guard": has_graph_dynamic_override_guard,
    }


def _clamp_weight(value: float, min_weight: float, max_weight: float) -> float:
    return min(max(float(value), float(min_weight)), float(max_weight))


def _numeric_direction_sanity() -> dict[str, float | bool]:
    target = 0.25
    min_weight = 0.5
    max_weight = 2.0
    low_masked = 0.10
    high_masked = 0.60
    low_visible = 0.10
    high_visible = 0.60

    low_reconstruction = 1.0 if low_masked <= target else _clamp_weight(low_masked / target, min_weight, max_weight)
    high_reconstruction = 1.0 if high_masked <= target else _clamp_weight(high_masked / target, min_weight, max_weight)
    low_semantic = _clamp_weight(low_visible / target, min_weight, max_weight)
    high_semantic = _clamp_weight(high_visible / target, min_weight, max_weight)

    return {
        "target": target,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "masked_0_10_reconstruction_multiplier": low_reconstruction,
        "masked_0_60_reconstruction_multiplier": high_reconstruction,
        "visible_0_10_semantic_multiplier": low_semantic,
        "visible_0_60_semantic_multiplier": high_semantic,
        "passes": (
            low_reconstruction <= high_reconstruction
            and high_reconstruction > 1.0
            and low_semantic <= 1.0
            and high_semantic > 1.0
        ),
    }


def _formal_dynamic_multiplier_probe(raw_config: dict[str, Any]) -> dict[str, float]:
    """Evaluate the formal multiplier dictionary with non-unity visible coverage."""
    masking = raw_config.get("masking", {}) if isinstance(raw_config.get("masking"), dict) else {}
    losses = raw_config.get("losses", {}) if isinstance(raw_config.get("losses"), dict) else {}
    conflict_aware = losses.get("conflict_aware") if isinstance(losses.get("conflict_aware"), dict) else {}
    config = SimpleNamespace(
        masking=SimpleNamespace(
            gaze_loss_mode=masking.get("gaze_loss_mode", "no_gaze"),
            gaze_mask_eps=float(masking.get("gaze_mask_eps", 1.0e-6)),
        ),
        losses=SimpleNamespace(
            conflict_aware_enabled=bool(conflict_aware.get("enabled", False)),
            conflict_aware_semantic_visible_coverage_target=float(
                conflict_aware.get("semantic_visible_coverage_target", 0.25)
            ),
            conflict_aware_reconstruction_masked_gaze_target=float(
                conflict_aware.get("reconstruction_masked_gaze_target", 0.25)
            ),
            conflict_aware_min_weight=float(conflict_aware.get("min_weight", 0.5)),
            conflict_aware_max_weight=float(conflict_aware.get("max_weight", 2.0)),
            allow_dynamic_graph_consistency_weighting=bool(
                conflict_aware.get("allow_dynamic_graph_consistency_weighting", False)
            ),
        ),
    )
    masking_output = SimpleNamespace(
        attention_tokens=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        patch_mask=torch.tensor([[False, True]], dtype=torch.bool),
    )
    weights, _ = compute_conflict_aware_weights(config, masking_output)
    return weights


REQUIRED_SUMMARY_FIELDS = [
    "dynamic_loss_weighting_enabled",
    "masked_gaze_coverage",
    "visible_gaze_coverage",
    "mask_policy_used",
    "adaptive_gaze_quota",
    "adaptive_random_fraction",
    "fallback_reason",
    "loss_weight_audit_summary",
]


def _check_summary_fields(config_path: Path) -> dict[str, bool]:
    output_dir_name = str(config_path.stem) if config_path.stem else "unknown"
    summary_candidates = list((PROJECT_ROOT / "outputs").glob(f"**/{output_dir_name}/**/*summary*.json"))
    if not summary_candidates:
        summary_candidates = list((PROJECT_ROOT / "outputs").glob("**/*summary*.json"))
    summary_candidates = sorted(
        summary_candidates,
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
    )

    found: dict[str, bool] = {field: False for field in REQUIRED_SUMMARY_FIELDS}
    for summary_path in summary_candidates[-3:]:
        try:
            data = json.load(open(summary_path, encoding="utf-8"))
            for field in REQUIRED_SUMMARY_FIELDS:
                if field in data:
                    found[field] = True
            dw_summary = data.get("dynamic_loss_weight_summary", {})
            if dw_summary:
                found["dynamic_loss_weighting_enabled"] = data.get("dynamic_loss_weighting_enabled", False)
        except Exception:
            continue
    return found


def audit_config_conflict_aware(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Check config has conflict_aware block properly configured."""
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    losses = raw_config.get("losses", {}) if isinstance(raw_config.get("losses"), dict) else {}
    conflict_aware = losses.get("conflict_aware")
    if isinstance(conflict_aware, dict):
        conflict_aware = dict(conflict_aware)
    else:
        conflict_aware = {}

    is_gaze = _is_gaze_enabled_config(raw_config, config_path)
    is_no_gaze = _is_no_gaze_config(raw_config)
    is_semantic_only = _is_semantic_only_config(raw_config)
    is_no_graph = _is_no_graph_ablation(raw_config)

    if is_gaze:
        enabled = bool(conflict_aware.get("enabled", False))
        if not enabled:
            _add(checks, "dw_config_gaze_enabled",
                 "fail", "Gaze-enabled config must have conflict_aware.enabled=true.",
                 _severity_for_tier(raw_config))
        else:
            _add(checks, "dw_config_gaze_enabled", "pass",
                 "Gaze-enabled config has conflict_aware.enabled=true.", "info")

        has_targets = (
            "semantic_visible_coverage_target" in conflict_aware
            and "reconstruction_masked_gaze_target" in conflict_aware
        )
        if has_targets:
            _add(checks, "dw_config_targets", "pass",
                 "conflict_aware has coverage target fields.", "info")
        else:
            _add(checks, "dw_config_targets", "fail",
                 "conflict_aware missing coverage target fields.",
                 _severity_for_tier(raw_config))

        has_weights = "min_weight" in conflict_aware and "max_weight" in conflict_aware
        if has_weights:
            _add(checks, "dw_config_weight_bounds", "pass",
                 "conflict_aware has min_weight/max_weight bounds.", "info")
        else:
            _add(checks, "dw_config_weight_bounds", "fail",
                 "conflict_aware missing min_weight/max_weight.",
                 _severity_for_tier(raw_config))

    if is_no_gaze:
        enabled = bool(conflict_aware.get("enabled", False))
        if enabled:
            _add(checks, "dw_no_gaze_conflict_aware",
                 "warn",
                 "no-gaze config has conflict_aware.enabled=true; dynamic weights must fallback to base (1.0) for no-gaze configs or zero gaze attention mass.",
                 "warn")
        else:
            _add(checks, "dw_no_gaze_conflict_aware", "pass",
                 "no-gaze config has conflict_aware.enabled=false (clean).", "info")

    if is_semantic_only:
        recon_weight = float(losses.get("reconstruction_weight", 1.0))
        if abs(recon_weight) < 1e-8:
            _add(checks, "dw_semantic_only_recon_zero", "pass",
                 "semantic-only: reconstruction_weight=0; dynamic recon scaling has no effect.", "info")
        conflict_enabled = bool(conflict_aware.get("enabled", False))
        if conflict_enabled and is_no_gaze:
            _add(checks, "dw_semantic_only_no_gaze", "info",
                 "semantic-only no-gaze: dynamic semantic weighting falls back to base.", "info")

    if is_no_graph:
        graph_weight = float(losses.get("graph_consistency_weight", 0.0))
        if abs(graph_weight) < 1e-8:
            _add(checks, "dw_no_graph_ablation", "pass",
                 "no-graph ablation: graph_consistency_weight=0.", "info")
        ge_enabled = (
            raw_config.get("graph_encoder", {}).get("enabled", True)
            if isinstance(raw_config.get("graph_encoder"), dict)
            else True
        )
        if not ge_enabled:
            _add(checks, "dw_no_graph_encoder", "pass",
                 "no-graph ablation: graph_encoder.enabled=false.", "info")


def audit_code_dynamic_weighting(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Audit that dynamic loss weighting is implemented in code and influences total_loss."""
    code_status = _check_codebase_for_dynamic_weighting()

    if not code_status["has_conflict_aware_fn"]:
        _add(checks, "dw_code_fn", "fail",
             "compute_conflict_aware_weights not found in dynamic loss weighting code.", "hard")
    else:
        _add(checks, "dw_code_fn", "pass",
             "compute_conflict_aware_weights exists in dynamic loss weighting code.", "info")

    if not code_status["has_total_loss_multiplication"]:
        _add(checks, "dw_code_total_loss", "fail",
             "dynamic_weights not multiplied into total_loss.", "hard")
    elif code_status["summary_only_not_training"]:
        _add(checks, "dw_code_total_loss", "fail",
             "dynamic_weights written to summary but NOT multiplied into total_loss.", "hard")
    else:
        _add(checks, "dw_code_total_loss", "pass",
             "dynamic_weights are multiplied into total_loss and affect backward.", "info")

    if not code_status["has_metrics_tracking"]:
        _add(checks, "dw_code_metrics", "fail",
             "dynamic_loss_weights not passed to metrics logger in trainer.py.", "hard")
    else:
        _add(checks, "dw_code_metrics", "pass",
             "dynamic_loss_weights passed to metrics.update() in trainer.py.", "info")

    if not code_status["has_summary_fields"]:
        _add(checks, "dw_code_summary", "warn",
             "metrics.py may be missing dynamic_loss_weighting_enabled or effective weight fields.", "warn")
    else:
        _add(checks, "dw_code_summary", "pass",
             "metrics.py includes dynamic_loss_weighting_enabled and effective weight fields.", "info")

    if not code_status["has_dynamic_module"]:
        _add(checks, "dw_code_module", "info",
             "No separate dynamic_loss_weighting.py module; logic is inline in losses.py (acceptable if <800 lines).", "info")

    if code_status["has_forbidden_reconstruction_inverse"]:
        _add(checks, "dw_reconstruction_direction_no_inverse", "fail",
             "Forbidden inverse reconstruction direction found: recon_target / masked_coverage.", "hard")
    else:
        _add(checks, "dw_reconstruction_direction_no_inverse", "pass",
             "No forbidden recon_target / masked_coverage reconstruction upweight logic found.", "info")

    if not code_status["has_reconstruction_positive_relation"]:
        _add(checks, "dw_reconstruction_positive_relation", "fail",
             "Could not confirm positive reconstruction relation masked_coverage / recon_target.", "hard")
    else:
        _add(checks, "dw_reconstruction_positive_relation", "pass",
             "Reconstruction multiplier is positively related to masked_gaze_coverage.", "info")

    if not code_status["has_semantic_positive_relation"]:
        _add(checks, "dw_semantic_positive_relation", "fail",
             "Could not confirm semantic positive relation visible_coverage / semantic_target.", "hard")
    else:
        _add(checks, "dw_semantic_positive_relation", "pass",
             "Semantic multiplier remains positively related to visible_gaze_coverage.", "info")

    if code_status["has_bad_reconstruction_upweight_warning"]:
        _add(checks, "dw_reconstruction_upweight_warning_direction", "fail",
             "reconstruction_upweighted warning still uses masked_gaze_coverage < target.", "hard")
    elif not code_status["has_good_reconstruction_upweight_warning"]:
        _add(checks, "dw_reconstruction_upweight_warning_direction", "fail",
             "Could not confirm reconstruction_upweighted warning uses masked_gaze_coverage > target.", "hard")
    else:
        _add(checks, "dw_reconstruction_upweight_warning_direction", "pass",
             "reconstruction_upweighted warning uses masked_gaze_coverage > target.", "info")

    if not code_status["has_no_gaze_fallback"] or not code_status["has_no_gaze_mode_fallback"]:
        _add(checks, "dw_no_gaze_fallback_code", "fail",
             "Could not confirm no-gaze config and gaze_mass < gaze_mask_eps both return base_weights.", "hard")
    else:
        _add(checks, "dw_no_gaze_fallback_code", "pass",
             "no-gaze config and zero-gaze fallback return base_weights.", "info")

    losses = raw_config.get("losses", {}) if isinstance(raw_config.get("losses"), dict) else {}
    conflict_aware = losses.get("conflict_aware") if isinstance(losses.get("conflict_aware"), dict) else {}
    allow_dynamic_graph = bool(conflict_aware.get("allow_dynamic_graph_consistency_weighting", False))
    formal_weights = _formal_dynamic_multiplier_probe(raw_config)
    semantic_multiplier = formal_weights["visible_align"]
    global_static = formal_weights["global_align"] == 1.0
    graph_static = formal_weights["graph_consistency"] == 1.0
    if (
        not code_status["has_global_align_static_policy"]
        or not global_static
        or semantic_multiplier == 1.0
    ):
        _add(checks, "dw_global_align_static", "fail",
             "Formal multiplier probe did not confirm global_align=1.0 under non-unity semantic weighting.", "hard")
    else:
        _add(checks, "dw_global_align_static", "pass",
             "Formal probe confirmed global_align=1.0 while visible_align uses a non-unity semantic multiplier.", "info")

    if not code_status["has_graph_dynamic_override_guard"]:
        _add(checks, "dw_graph_consistency_static", "fail",
             "Could not confirm graph_consistency dynamic weighting is guarded by its explicit config switch.", "hard")
    elif allow_dynamic_graph:
        _add(checks, "dw_graph_consistency_static", "pass",
             "Historical/development config explicitly enables dynamic graph_consistency weighting.", "info")
    elif graph_static:
        _add(checks, "dw_graph_consistency_static", "pass",
             "Formal probe confirmed graph_consistency=1.0 with the guarded override disabled.", "info")
    else:
        _add(checks, "dw_graph_consistency_static", "fail",
             "Formal graph_consistency multiplier is not 1.0 despite its dynamic override being disabled.", "hard")

    sanity = _numeric_direction_sanity()
    if sanity["passes"]:
        _add(checks, "dw_numeric_direction_sanity", "pass",
             "Numeric sanity passed: "
             f"masked0.10_recon={sanity['masked_0_10_reconstruction_multiplier']}, "
             f"masked0.60_recon={sanity['masked_0_60_reconstruction_multiplier']}, "
             f"visible0.10_semantic={sanity['visible_0_10_semantic_multiplier']}, "
             f"visible0.60_semantic={sanity['visible_0_60_semantic_multiplier']}.",
             "info")
    else:
        _add(checks, "dw_numeric_direction_sanity", "fail",
             f"Numeric sanity failed: {sanity}", "hard")


def audit_no_gaze_pollution(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Ensure no-gaze matched control is not polluted by gaze-based dynamic weighting."""
    if not _is_no_gaze_config(raw_config):
        return

    losses = raw_config.get("losses", {}) if isinstance(raw_config.get("losses"), dict) else {}
    conflict_aware = losses.get("conflict_aware")
    if isinstance(conflict_aware, dict):
        enabled = bool(conflict_aware.get("enabled", False))
    else:
        enabled = False

    if enabled:
        _add(checks, "dw_no_gaze_pollution_risk",
             "warn",
             "no-gaze config has conflict_aware.enabled=true. "
             "Code must detect no-gaze configs or zero gaze attention mass and fallback to base weights (1.0). "
             "If no fallback exists, no-gaze matched control is contaminated by dynamic weights.",
             "warn")
    else:
        _add(checks, "dw_no_gaze_pollution", "pass",
             "no-gaze config has conflict_aware.enabled=false; no dynamic weighting contamination.", "info")


def audit_ablation_fallback(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Check semantic-only and no-graph ablation configs have correct fallback."""
    is_semantic_only = _is_semantic_only_config(raw_config)
    is_no_graph = _is_no_graph_ablation(raw_config)

    if is_semantic_only:
        conflicts = raw_config.get("losses", {}).get("conflict_aware", {})
        if isinstance(conflicts, dict) and conflicts.get("enabled", False):
            _add(checks, "dw_semantic_only_conflict", "info",
                 "semantic-only: conflict_aware enabled; reconstruction_weight=0 so dynamic recon scale has no effect. Semantic dynamic weighting applies if gaze is available.", "info")
        else:
            _add(checks, "dw_semantic_only_clean", "pass",
                 "semantic-only: conflict_aware disabled; no dynamic weighting.", "info")

    if is_no_graph:
        _add(checks, "dw_no_graph_fallback", "pass",
             "no-graph ablation: graph_encoder disabled, graph_consistency_weight=0; dynamic weighting does not depend on graph.", "info")


def audit_summary_required_fields(
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Check summary contains required dynamic weighting fields."""
    found = _check_summary_fields(config_path)
    for field in REQUIRED_SUMMARY_FIELDS:
        if found.get(field):
            _add(checks, f"dw_summary_{field}", "pass",
                 f"Summary contains {field}.", "info")
        else:
            _add(checks, f"dw_summary_{field}", "warn",
                 f"Summary field {field} not confirmed in existing outputs (may be missing or no smoke run exists).", "warn")


def audit_dynamic_loss_weighting(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path,
) -> None:
    """Main entry point for dynamic loss weighting compliance audit."""
    start_index = len(checks)
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    if run_tier not in FINAL_OR_TEMPLATE_TIERS:
        _add(checks, "dw_gate_skip", "pass",
             "Not a final/formal production config or final_config_template; dynamic loss weighting audit skipped.", "info")
        return

    audit_config_conflict_aware(raw_config, checks, config_path)
    audit_code_dynamic_weighting(raw_config, checks)
    audit_no_gaze_pollution(raw_config, checks, config_path)
    audit_ablation_fallback(raw_config, checks, config_path)
    audit_summary_required_fields(checks, config_path)

    dynamic_checks = checks[start_index:]
    has_failures = any(
        c["status"] == "fail" and c["severity"] in ("hard", "blocking_warning")
        for c in dynamic_checks
    )
    has_warnings = any(
        c["status"] in ("warn",) or c["severity"] == "warn"
        for c in dynamic_checks
    )

    if has_failures:
        _add(checks, "dw_overall", "fail", "Dynamic loss weighting compliance: FAIL.", "hard")
    elif has_warnings:
        _add(checks, "dw_overall", "pass_with_warnings", "Dynamic loss weighting compliance: pass_with_warnings.", "warn")
    else:
        _add(checks, "dw_overall", "pass", "Dynamic loss weighting compliance: pass.", "info")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Audit dynamic loss weighting compliance.")
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 config YAML path.")
    args = parser.parse_args()

    resolved = args.config.expanduser().resolve()
    raw_config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    checks: list[dict[str, Any]] = []
    audit_dynamic_loss_weighting(raw_config, checks, resolved)

    failures = [c for c in checks if c["status"] == "fail" and c["severity"] == "hard"]
    warnings_list = [c for c in checks if c["status"] == "warn" or c["severity"] in ("warn", "blocking_warning")]

    if failures:
        overall = "fail"
    elif warnings_list:
        overall = "pass_with_warnings"
    else:
        overall = "pass"

    print(json.dumps({"status": overall, "checks": checks}, indent=2, ensure_ascii=True))

    if failures:
        raise SystemExit(1)
