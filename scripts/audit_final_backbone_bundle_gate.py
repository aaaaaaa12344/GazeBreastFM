from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.data.stage1_entry_contract import (
    STAGE1_GAZE_SUPERVISION_SOURCES,
    validate_stage1_manifest_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

FINAL_TIERS = {"final"}
FINAL_TEMPLATE_TIERS = {"final_config_template"}
FORMAL_PRODUCTION_TIERS = {"formal_production", "production_ready_candidate"}
FINAL_OR_TEMPLATE_TIERS = FINAL_TIERS | FINAL_TEMPLATE_TIERS
PRODUCTION_HARD_TIERS = FINAL_TIERS | FORMAL_PRODUCTION_TIERS
SMOKE_TIERS = {"smoke", "development", "engineering_acceptance"}
VALID_TIERS = FINAL_OR_TEMPLATE_TIERS | FORMAL_PRODUCTION_TIERS | SMOKE_TIERS | {"legacy_ablation"}
GAZE_RELATED_FIELDS = {
    "gaze_supervision_source",
    "attention_map_path",
    "high_conf_mask_path",
    "patch_gaze_weight_path",
    "prior_status",
    "audit_status",
    "trajectory_consensus_summary_path",
    "gaze_prior_qc_metrics_path",
}
CONCEPT_FIELDS = (
    "view",
    "laterality",
    "density",
    "finding",
    "birads",
    "cancer_label",
    "benign_malignant_label",
)
WRAPPER_BACKENDS = {
    "pretrained_visual_encoder",
    "local_torch_checkpoint",
    "generic_torch_checkpoint",
}
REAL_BACKENDS = {
    "generic_timm_vit",
    "hf_clip_vit",
    "clip_vit_b16",
    "biomedclip",
    "medsiglip",
    "mammo_clip",
    "mammo_fm",
    "mammo_fm_timm_efficientnet_b5",
    "rad_dino",
}
HIGHRES_REAL_BACKENDS = {
    "mammo_fm_timm_efficientnet_b5",
    "timm_highres_hierarchical",
    "mammo_fm_like_highres_adapter",
}
HIGHRES_STUB_BACKENDS = {"preflight_highres_hierarchical_stub"}
PRETRAINED_BACKENDS = WRAPPER_BACKENDS | REAL_BACKENDS


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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _add(checks: list[dict[str, Any]], check_id: str, status: str, message: str, severity: str) -> None:
    checks.append({"id": check_id, "status": status, "severity": severity, "message": message})


def _is_final(raw_config: dict[str, Any]) -> bool:
    tier = str(raw_config.get("run_tier", raw_config.get("metadata", {}).get("run_tier", ""))).strip()
    return tier in PRODUCTION_HARD_TIERS


def _is_formal_production(raw_config: dict[str, Any]) -> bool:
    tier = str(raw_config.get("run_tier", raw_config.get("metadata", {}).get("run_tier", ""))).strip()
    return tier in FORMAL_PRODUCTION_TIERS


def _is_final_or_template(raw_config: dict[str, Any]) -> bool:
    tier = str(raw_config.get("run_tier", raw_config.get("metadata", {}).get("run_tier", ""))).strip()
    return tier in (FINAL_OR_TEMPLATE_TIERS | FORMAL_PRODUCTION_TIERS)


def _severity_for_tier(raw_config: dict[str, Any], hard_severity: str = "hard") -> str:
    tier = str(raw_config.get("run_tier", raw_config.get("metadata", {}).get("run_tier", ""))).strip()
    if tier in FINAL_TIERS:
        return hard_severity
    if tier in FINAL_TEMPLATE_TIERS:
        return "warn" if hard_severity == "hard" else hard_severity
    return "info"


def _metadata_value(raw_config: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = raw_config.get("metadata") if isinstance(raw_config.get("metadata"), dict) else {}
    return raw_config.get(key, metadata.get(key, default))


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def audit_backbone_bundle_gate(
    raw_config: dict[str, Any],
    bundle: Any,
    checks: list[dict[str, Any]],
    config_path: Path,
) -> dict[str, Any]:
    """Run backbone enforcement and Stage 1 bundle validation gates."""
    _audit_config_fields(raw_config, checks)
    bundle_report = _audit_bundle(bundle, raw_config, checks, config_path=config_path)
    return bundle_report


def _audit_config_fields(raw_config: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    model = raw_config.get("model") if isinstance(raw_config.get("model"), dict) else {}
    semantic = raw_config.get("semantic") if isinstance(raw_config.get("semantic"), dict) else {}
    audit = raw_config.get("audit") if isinstance(raw_config.get("audit"), dict) else {}
    references = raw_config.get("references") if isinstance(raw_config.get("references"), dict) else {}
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    hard = run_tier in PRODUCTION_HARD_TIERS

    if not run_tier:
        _add(checks, "run_tier_present", "fail", "Config is missing run_tier.", "hard")
    elif run_tier not in VALID_TIERS:
        _add(checks, "run_tier_known", "fail", f"Unsupported run_tier={run_tier!r}.", "hard")
    else:
        _add(checks, "run_tier_present", "pass", f"run_tier={run_tier}", "info")

    backend = str(model.get("vision_encoder_name", "minimal_patch_encoder")).strip().lower()
    if hard and backend == "minimal_patch_encoder":
        _add(checks, "final_no_minimal_patch_encoder", "fail", "Production config uses minimal_patch_encoder.", "hard")
    elif hard and backend in WRAPPER_BACKENDS:
        _add(checks, "final_no_wrapper_backend", "fail", f"Production config uses compliance-wrapper backend {backend!r} instead of a real pretrained backend.", "hard")
    elif hard and backend not in PRETRAINED_BACKENDS:
        _add(checks, "final_pretrained_backend", "fail", f"Production config uses unsupported backend {backend!r}.", "hard")
    elif backend in REAL_BACKENDS:
        _add(checks, "visual_encoder_backend", "pass", f"visual_encoder_backend={backend} (real pretrained backbone)", "info")
    else:
        _add(checks, "visual_encoder_backend", "pass", f"visual_encoder_backend={backend}", "info")

    weight_path = _resolve(model.get("pretrained_weight_path", model.get("pretrained_model_path")))
    if hard and (weight_path is None or not weight_path.exists()):
        _add(checks, "final_pretrained_weight_exists", "fail", f"Missing pretrained weight path: {weight_path}", "hard")
    elif weight_path is not None:
        _add(checks, "pretrained_weight_exists", "pass" if weight_path.exists() else "warn", str(weight_path), "warn")

    if bool(raw_config.get("require_teacher_latents", False)):
        severity = "hard" if hard else "warn"
        _add(checks, "teacher_latents_not_required", "fail" if hard else "warn", "require_teacher_latents=true.", severity)
    else:
        _add(checks, "teacher_latents_not_required", "pass", "require_teacher_latents=false.", "info")
    if str(semantic.get("reconstruction_teacher_source", "self_masked_reconstruction")).strip() != "self_masked_reconstruction" and hard:
        _add(checks, "final_no_teacher_source", "fail", "Production config uses teacher latent reconstruction.", "hard")
    else:
        _add(checks, "teacher_source_boundary", "pass", "Teacher latent is not final mainline.", "info")

    _allowed_policies = {"read_only_reference_only", "no_runtime_external_repo_dependency"}
    import_policy = str(references.get("import_policy", "read_only_reference_only")).strip() if references else ""
    if not import_policy:
        _add(checks, "external_reference_import_policy", "pass", "No references block; import policy N/A.", "info")
    elif import_policy not in _allowed_policies:
        _add(checks, "external_reference_import_policy", "fail",
             f"import_policy={import_policy!r} not in {sorted(_allowed_policies)}.", "hard")
    else:
        _add(checks, "external_reference_import_policy", "pass",
             f"import_policy={import_policy} — no runtime external repo dependency.", "info")

    if hard and bool(audit.get("warn_only", False)):
        _add(checks, "final_no_warn_only", "fail", "Production config sets audit.warn_only=true.", "hard")

    _FIXTURE_BACKBONE_DIR = "final_compliance_fixtures/pretrained_backbone"
    weight_path_str = str(model.get("pretrained_weight_path", "") or "")
    allow_fallback = bool(model.get("allow_missing_pretrained_fallback", False))
    is_fixture_backbone = _FIXTURE_BACKBONE_DIR in weight_path_str.replace("\\", "/")
    is_prod_or_final = run_tier in PRODUCTION_HARD_TIERS

    if is_prod_or_final and is_fixture_backbone:
        _add(checks, "no_fixture_backbone_in_formal", "fail",
             f"Production config uses fixture backbone: {weight_path_str}. "
             "Replace with real pretrained weight.", "hard")
    elif is_prod_or_final and allow_fallback:
        _add(checks, "no_allow_missing_pretrained_fallback", "fail",
             "Production config has allow_missing_pretrained_fallback=true.", "hard")
    elif is_fixture_backbone:
        _add(checks, "no_fixture_backbone_in_formal", "blocking_warning",
             f"Config uses fixture backbone: {weight_path_str}. "
             "Must be replaced with real pretrained backbone before formal training.",
             "blocking_warning")
    elif is_prod_or_final:
        _add(checks, "no_fixture_backbone_in_formal", "pass",
             f"Production config uses non-fixture backbone: {weight_path_str}.", "info")
    else:
        _add(checks, "no_fixture_backbone_in_formal", "pass",
             "Not production/final; fixture backbone check skipped.", "info")

    # -- highres encoder contract gate (R5) --
    is_highres = backend in HIGHRES_REAL_BACKENDS
    if is_prod_or_final and backend in HIGHRES_STUB_BACKENDS:
        _add(checks, "final_no_highres_stub", "fail",
             f"Production config uses highres stub backend {backend!r}. "
             "Use mammo_fm_timm_efficientnet_b5 or another real highres backend.",
             "hard")
    elif is_prod_or_final and is_highres:
        contract_status = "formal_frozen" if backend == "mammo_fm_timm_efficientnet_b5" else "candidate"
        _add(checks, "highres_contract_status", "pass",
             f"highres_contract_status={contract_status}, "
             f"multiscale_status=derived_local_multiscale, "
             f"encoder_backend={backend}",
             "info")
    elif is_highres:
        _add(checks, "highres_contract_status", "pass",
             f"highres_contract_status=candidate, "
             f"encoder_backend={backend} (non-production tier)",
             "info")

    # -- Mammo-FM weight path gate --
    if backend == "mammo_fm_timm_efficientnet_b5" and is_prod_or_final:
        mammo_path = _resolve(
            model.get("pretrained_weight_path") or model.get("pretrained_model_path")
        )
        if mammo_path is None or not mammo_path.exists():
            _add(checks, "mammo_fm_weight_missing", "fail",
                 f"Mammo-FM pretrained weight not found: {mammo_path}. "
                 "Formal mammo_fm_timm_efficientnet_b5 config requires real weights.",
                 "hard")
        else:
            _add(checks, "mammo_fm_weight_exists", "pass",
                 f"Mammo-FM pretrained weight found: {mammo_path}", "info")

    if is_prod_or_final and allow_fallback:
        pass
    elif allow_fallback:
        _add(checks, "allow_missing_pretrained_fallback", "warn",
             "allow_missing_pretrained_fallback=true; formal training should disable this.", "warn")
    else:
        _add(checks, "allow_missing_pretrained_fallback", "pass",
             "allow_missing_pretrained_fallback=false.", "info")

    _FIXTURE_BUNDLE_PREFIX = "final_compliance_fixtures/stage1_m"
    manifest_path_str = str(raw_config.get("data", {}).get("image_manifest_path",
                           raw_config.get("image_manifest_path", "")) or "")
    if not manifest_path_str:
        manifest_path_str = str(
            raw_config.get("model", {}).get("data", {}).get("image_manifest_path", "") or ""
        )
    is_fixture_bundle = _FIXTURE_BUNDLE_PREFIX in manifest_path_str.replace("\\", "/")

    if is_prod_or_final and is_fixture_bundle:
        _add(checks, "no_fixture_bundle_in_formal", "fail",
             f"Production config uses fixture bundle: {manifest_path_str}. "
             "Replace with real Stage 1 bundle.", "hard")
    elif is_prod_or_final:
        _add(checks, "no_fixture_bundle_in_formal", "pass",
             f"Production config uses non-fixture manifest: {manifest_path_str}.", "info")
    elif is_fixture_bundle:
        _add(checks, "no_fixture_bundle_in_formal", "blocking_warning",
             f"Config uses fixture bundle: {manifest_path_str}. "
             "Must be replaced with real Stage 1 bundle before formal training.",
             "blocking_warning")
    else:
        _add(checks, "no_fixture_bundle_in_formal", "pass",
             "Not formal/final; fixture bundle check skipped.", "info")


def _audit_bundle(
    config: Any,
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
    config_path: Path | None = None,
) -> dict[str, Any]:
    trainer = config.trainer
    report = validate_stage1_manifest_bundle(
        manifest_path=trainer.data.image_manifest_path,
        text_prompt_path=trainer.data.text_prompt_path,
        prompt_embedding_path=trainer.semantic.prompt_embedding_path,
        semantic_soft_label_path=trainer.semantic.semantic_soft_label_path,
        semantic_manifest_path=trainer.semantic.semantic_manifest_path,
        semantic_soft_label_format=trainer.semantic.semantic_soft_label_format,
        semantic_soft_label_topk_path=trainer.semantic.semantic_soft_label_topk_path,
        birads_prior_manifest_path=trainer.semantic.birads_prior_manifest_path,
    )
    if report["status"] == "fail":
        _add(checks, "stage1_bundle_contract", "fail", "Stage 1 bundle validation failed.", "hard")
    else:
        _add(checks, "stage1_bundle_contract", "pass", f"Stage 1 bundle status={report['status']}.", "info")

    manifest_path = trainer.data.image_manifest_path
    if manifest_path.name != "manifest_stage1_semantic.csv":
        _add(checks, "trainer_reads_stage1_bundle", "fail", "Trainer manifest is not a standard Stage 1 bundle manifest.", "hard")
        return report

    rows = _rows(manifest_path)
    total_rows = len(rows)
    gaze_counter = Counter(str(row.get("gaze_supervision_source", "")).strip() for row in rows)
    unsupported = sorted(source for source in gaze_counter if source not in STAGE1_GAZE_SUPERVISION_SOURCES)
    if unsupported:
        _add(checks, "gaze_supervision_source_allowed", "fail", f"Unsupported gaze sources: {unsupported}", "hard")
    else:
        _add(checks, "gaze_supervision_source_allowed", "pass", f"distribution={dict(gaze_counter)}", "info")

    missing_label_columns = []
    for concept in CONCEPT_FIELDS:
        for suffix in ("observed_mask", "source", "confidence", "status"):
            column = f"{concept}_{suffix}"
            if rows and column not in rows[0]:
                missing_label_columns.append(column)
    if missing_label_columns:
        _add(checks, "concept_target_metadata_fields", "fail", "Missing concept target metadata columns: " + ", ".join(missing_label_columns[:20]), "hard")
    else:
        _add(checks, "concept_target_metadata_fields", "pass", "Concept targets include value/source/confidence/status/observed_mask metadata.", "info")

    mislabeled_defaults = [
        row.get("image_id", "")
        for row in rows
        if str(row.get("gaze_supervision_source", "")).strip() in {"observed_gaze", "diffeye_generated_gaze"}
        and any(str(row.get(key, "")).lower().find(token) >= 0 for key in ("prior_status", "prior_version", "audit_status") for token in ("center", "default", "neutral"))
    ]
    if mislabeled_defaults:
        _add(checks, "no_default_prior_as_real_gaze", "fail", f"Default/center prior mislabeled as real gaze: {mislabeled_defaults[:10]}", "hard")
    else:
        _add(checks, "no_default_prior_as_real_gaze", "pass", "No default/center prior is labeled as observed/diffeye gaze.", "info")

    # Row-level gaze prior audit
    gaze_source_labels = {"observed_gaze", "diffeye_generated_gaze"}
    require_all_gaze = raw_config.get("require_attention_prior_paths", False)
    config_file_name = str(config_path.name if config_path else "").lower()
    is_m3 = config_file_name.startswith("m3_")
    is_m4 = config_file_name.startswith("m4_")

    priority_fields = ("attention_map_path", "high_conf_mask_path", "patch_gaze_weight_path")
    attention_map_exists = 0
    high_conf_mask_exists = 0
    patch_gaze_weight_exists = 0
    total_gaze_rows = 0
    invalid_gaze_prior_rows: list[dict[str, str]] = []

    for row in rows:
        source = str(row.get("gaze_supervision_source", "")).strip()
        is_gaze_row = source in gaze_source_labels
        if is_gaze_row:
            total_gaze_rows += 1

        row_issues: list[str] = []
        for field in priority_fields:
            field_val = str(row.get(field, "")).strip()
            if is_gaze_row:
                if not field_val:
                    row_issues.append(f"missing_{field}")
                else:
                    path = _resolve(field_val, manifest_path.parent)
                    if path is None or not path.exists():
                        row_issues.append(f"{field}_file_not_found")
                    else:
                        if field == "attention_map_path":
                            attention_map_exists += 1
                        elif field == "high_conf_mask_path":
                            high_conf_mask_exists += 1
                        elif field == "patch_gaze_weight_path":
                            patch_gaze_weight_exists += 1
            elif source in {"no_gaze", ""}:
                pass
            elif source not in gaze_source_labels:
                if not field_val:
                    row_issues.append(f"source={source}_missing_{field}")

        if row_issues:
            invalid_gaze_prior_rows.append({"image_id": str(row.get("image_id", "")), "source": source, "issues": row_issues})

    attn_rate = attention_map_exists / max(1, total_gaze_rows)
    hcm_rate = high_conf_mask_exists / max(1, total_gaze_rows)
    pgw_rate = patch_gaze_weight_exists / max(1, total_gaze_rows)

    gaze_prior_qc = {
        "gaze_supervision_source_distribution": dict(gaze_counter),
        "attention_map_exists_rate": round(attn_rate, 4),
        "high_conf_mask_exists_rate": round(hcm_rate, 4),
        "patch_gaze_weight_exists_rate": round(pgw_rate, 4),
        "total_gaze_rows": total_gaze_rows,
        "total_rows": total_rows,
        "invalid_gaze_prior_row_count": len(invalid_gaze_prior_rows),
    }

    if is_m3 and _is_final(raw_config):
        if total_gaze_rows > 0:
            if attn_rate < 1.0 or hcm_rate < 1.0 or pgw_rate < 1.0:
                _add(checks, "m3_final_all_gaze_rows_readable", "fail",
                     f"M3 final must have 100% readable gaze prior paths. "
                     f"attn={attn_rate:.3f} hcm={hcm_rate:.3f} pgw={pgw_rate:.3f}. "
                     f"Invalid rows: {[r['image_id'] for r in invalid_gaze_prior_rows[:10]]}",
                     "hard")
            else:
                _add(checks, "m3_final_all_gaze_rows_readable", "pass",
                     f"All {total_gaze_rows} gaze rows have readable prior paths.", "info")
        else:
            _add(checks, "m3_final_all_gaze_rows_readable", "fail",
                 "M3 final has no gaze rows (expected gaze rows for M3).", "hard")

    if is_m4 and _is_final(raw_config):
        if total_gaze_rows > 0:
            if attn_rate < 1.0 or hcm_rate < 1.0 or pgw_rate < 1.0:
                _add(checks, "m4_final_gaze_rows_readable", "fail",
                     f"M4 final gaze rows must have readable prior paths. "
                     f"attn={attn_rate:.3f} hcm={hcm_rate:.3f} pgw={pgw_rate:.3f}.",
                     "hard")
            else:
                _add(checks, "m4_final_gaze_rows_readable", "pass",
                     f"All {total_gaze_rows} gaze rows have readable prior paths.", "info")

    if invalid_gaze_prior_rows:
        _add(checks, "gaze_prior_row_level_qc", "pass" if not (require_all_gaze and invalid_gaze_prior_rows) else "fail",
             f"Gaze prior row-level QC: {gaze_prior_qc}",
             "hard" if (require_all_gaze and invalid_gaze_prior_rows) else "info")
    else:
        _add(checks, "gaze_prior_row_level_qc", "pass",
             f"Gaze prior row-level QC: {gaze_prior_qc}", "info")

    if raw_config.get("require_attention_prior_paths"):
        missing_prior_rows = []
        for row in rows:
            for field in ("attention_map_path", "high_conf_mask_path", "patch_gaze_weight_path"):
                path = _resolve(row.get(field), manifest_path.parent)
                if path is None or not path.exists():
                    missing_prior_rows.append(row.get("image_id", ""))
                    break
        if missing_prior_rows:
            _add(checks, "required_gaze_prior_paths_readable", "fail", f"Rows missing gaze prior paths: {missing_prior_rows[:10]}", "hard")
        else:
            _add(checks, "required_gaze_prior_paths_readable", "pass", "All required gaze prior path fields are populated.", "info")

    # M3 pass193 sensitivity check
    config_name = str(config_path or "").lower()
    is_pass193 = "pass193" in config_name
    run_tier = str(_metadata_value(raw_config, "run_tier", "")).strip()
    is_template = run_tier in FINAL_TEMPLATE_TIERS
    is_final_tier = run_tier in FINAL_TIERS
    if is_pass193 and (is_final_tier or is_template):
        manifest_str = str(manifest_path).lower()
        uses_usable189 = "usable189" in manifest_str
        if uses_usable189:
            severity = "hard" if is_final_tier else "hard"
            _add(checks, "m3_pass193_no_usable189_bundle", "fail",
                 f"Config name contains 'pass193' but manifest is from usable189 bundle.", severity)
        elif is_final_tier and total_rows != 193:
            _add(checks, "m3_pass193_row_count_193", "fail",
                 f"Pass193 final config must have 193 rows, got {total_rows}.", "hard")
        elif is_final_tier:
            unique_ids = len({str(row.get("image_id", "")).strip() for row in rows})
            if unique_ids != 193:
                _add(checks, "m3_pass193_unique_ids_193", "fail",
                     f"Pass193 final config must have 193 unique image_ids, got {unique_ids}.", "hard")
            else:
                _add(checks, "m3_pass193_row_count_193", "pass",
                     f"Pass193 manifest has {total_rows} rows with {unique_ids} unique image_ids.", "info")
        else:
            _add(checks, "m3_pass193_row_count_193", "pass",
                 f"Pass193 template check: {total_rows} rows (not yet enforcing 193 for template).", "info")

    return report
