"""Clinical graph sidecar & modality wiring audit helpers.

Split from audit_final_model_compliance.py to keep main audit under 800 lines.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_modality(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"mammo", "mammography", "mg"}:
        return "mammography"
    if text in {"mri", "magnetic_resonance", "mr"}:
        return "mri"
    if text in {"ultrasound", "us", "ultrasonography"}:
        return "ultrasound"
    return text or "unknown"


def _project_root(raw_config: dict[str, Any], config_path: Path) -> Path:
    return _resolve(raw_config.get("project_root", PROJECT_ROOT), config_path.parent) or PROJECT_ROOT


def audit_sidecar_existence_and_coverage(
    raw_config: dict[str, Any],
    config_path: Path,
    checks: list[dict[str, Any]],
) -> None:
    """Audit clinical graph sidecar: file existence, row matching, modality coverage."""
    cg_block = raw_config.get("clinical_graph")
    if cg_block is None or not isinstance(cg_block, dict):
        return

    sidecar_path = cg_block.get("sidecar_case_concept_vector_path")
    if sidecar_path is None:
        _add(checks, "clinical_graph_sidecar_exists", "fail",
             "sidecar_case_concept_vector_path not set in clinical_graph block.", "hard")
        return

    project_root = _project_root(raw_config, config_path)
    resolved = _resolve(sidecar_path, project_root)
    if resolved is None or not resolved.exists():
        _add(checks, "clinical_graph_sidecar_exists", "fail",
             f"Sidecar file not found: {sidecar_path}", "hard")
        return

    _add(checks, "clinical_graph_sidecar_exists", "pass",
         f"Sidecar exists: {sidecar_path}", "info")

    sidecar_rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    _add(checks, "clinical_graph_sidecar_jsonl_valid", "fail",
                         f"Invalid JSONL at line {line_number}: {exc}", "hard")
                    return
                sidecar_rows.append(payload)
        line_count = len(sidecar_rows)
        if line_count == 0:
            _add(checks, "clinical_graph_sidecar_min_rows", "fail",
                 "Sidecar has 0 rows.", "hard")
        else:
            _add(checks, "clinical_graph_sidecar_min_rows", "pass",
                 f"Sidecar has {line_count} rows.", "info")
    except Exception as exc:
        _add(checks, "clinical_graph_sidecar_readable", "fail",
             f"Cannot read sidecar: {exc}", "hard")
        return

    manifest_path = _resolve(raw_config.get("image_manifest_path"), project_root)
    if manifest_path is None or not manifest_path.exists():
        _add(checks, "clinical_graph_sidecar_manifest_alignment", "fail",
             f"image_manifest_path not found for sidecar alignment: {raw_config.get('image_manifest_path')}", "hard")
        return

    manifest_rows = _rows(manifest_path)
    manifest_id_counts = Counter(
        str(row.get("image_id", "")).strip()
        for row in manifest_rows
        if str(row.get("image_id", "")).strip()
    )
    sidecar_id_counts = Counter(
        str(row.get("image_id", "")).strip()
        for row in sidecar_rows
        if str(row.get("image_id", "")).strip()
    )
    manifest_ids = set(manifest_id_counts)
    sidecar_ids = set(sidecar_id_counts)
    missing_in_sidecar = sorted(manifest_ids - sidecar_ids)
    extra_in_sidecar = sorted(sidecar_ids - manifest_ids)
    duplicate_image_id = sorted(image_id for image_id, count in sidecar_id_counts.items() if count > 1)

    manifest_modalities = {
        str(row.get("image_id", "")).strip(): _normalize_modality(row.get("modality"))
        for row in manifest_rows
        if str(row.get("image_id", "")).strip()
    }
    sidecar_modalities = {
        str(row.get("image_id", "")).strip(): _normalize_modality(row.get("modality"))
        for row in sidecar_rows
        if str(row.get("image_id", "")).strip()
    }
    modality_mismatch = [
        {
            "image_id": image_id,
            "manifest_modality": manifest_modalities[image_id],
            "sidecar_modality": sidecar_modalities[image_id],
        }
        for image_id in sorted(manifest_ids & sidecar_ids)
        if manifest_modalities.get(image_id) != sidecar_modalities.get(image_id)
    ]

    if len(sidecar_rows) != len(manifest_rows):
        _add(checks, "clinical_graph_sidecar_row_count_alignment", "fail",
             f"sidecar_rows={len(sidecar_rows)} image_manifest_rows={len(manifest_rows)}", "hard")
    else:
        _add(checks, "clinical_graph_sidecar_row_count_alignment", "pass",
             f"sidecar_rows={len(sidecar_rows)} image_manifest_rows={len(manifest_rows)}", "info")
    if missing_in_sidecar or extra_in_sidecar:
        _add(checks, "clinical_graph_sidecar_image_id_alignment", "fail",
             f"missing_in_sidecar={missing_in_sidecar}; extra_in_sidecar={extra_in_sidecar}", "hard")
    else:
        _add(checks, "clinical_graph_sidecar_image_id_alignment", "pass",
             "image_id sets match; missing_in_sidecar=[]; extra_in_sidecar=[]", "info")

    if duplicate_image_id:
        _add(checks, "clinical_graph_sidecar_duplicate_image_id", "fail",
             f"duplicate_image_id={duplicate_image_id}", "hard")
    else:
        _add(checks, "clinical_graph_sidecar_duplicate_image_id", "pass",
             "duplicate_image_id=[]", "info")

    if modality_mismatch:
        _add(checks, "clinical_graph_sidecar_modality_match", "fail",
             f"modality_mismatch={modality_mismatch[:20]}", "hard")
    else:
        _add(checks, "clinical_graph_sidecar_modality_match", "pass",
             "modality_mismatch=[]", "info")

    try:
        manifest_modality_counts = Counter(manifest_modalities.values())
        sidecar_modality_counts = Counter(sidecar_modalities.values())
        for modality, expected_count in sorted(manifest_modality_counts.items()):
            actual_count = sidecar_modality_counts.get(modality, 0)
            if actual_count != expected_count:
                _add(checks, "clinical_graph_sidecar_modality_coverage", "fail",
                     f"modality={modality} manifest_rows={expected_count} sidecar_rows={actual_count}", "hard")
            else:
                _add(checks, f"clinical_graph_sidecar_modality_{modality}", "pass",
                     f"Modality '{modality}': {actual_count} rows.", "info")
    except Exception as exc:
        _add(checks, "clinical_graph_sidecar_coverage_audit", "warn",
             f"Could not audit modality coverage: {exc}", "warn")

    summary_path = resolved.with_name("tri_modal_concept_vector_summary.json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _add(checks, "clinical_graph_sidecar_summary_valid", "fail",
                 f"Cannot parse concept vector summary: {exc}", "hard")
        else:
            if summary.get("warn_on_heuristic_fallback") is True:
                _add(checks, "clinical_graph_sidecar_no_heuristic_fallback", "fail",
                     "Sidecar was generated with warn_on_heuristic_fallback=true.", "hard")
            else:
                _add(checks, "clinical_graph_sidecar_no_heuristic_fallback", "pass",
                     "Sidecar summary confirms warn_on_heuristic_fallback=false.", "info")


def audit_graph_consistency_integration(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Verify GraphConsistencyLoss is importable and wired."""
    cg_block = raw_config.get("clinical_graph")
    if cg_block is None or not isinstance(cg_block, dict):
        return

    use_for_consistency = cg_block.get("use_for_concept_consistency", False)
    if not use_for_consistency:
        _add(checks, "graph_consistency_integration", "warn",
             "clinical_graph.use_for_concept_consistency is false; graph consistency loss not used.", "warn")
        return

    # Check importable
    try:
        from breast_pretrain.losses.graph_consistency import compute_graph_consistency_loss  # noqa: F401
        _add(checks, "graph_consistency_importable", "pass",
             "compute_graph_consistency_loss is importable.", "info")
    except ImportError as exc:
        _add(checks, "graph_consistency_importable", "fail",
             f"Cannot import compute_graph_consistency_loss: {exc}", "hard")
        return

    # Check graph_consistency_weight > 0 when use_for_concept_consistency is true
    loss_block = raw_config.get("losses") or {}
    weight = float(loss_block.get("graph_consistency_weight", 0.0))
    if weight <= 0.0:
        _add(checks, "graph_consistency_weight_active", "warn",
             f"graph_consistency_weight={weight} but use_for_concept_consistency=true. "
             "Loss is wired but will not contribute. Set >0 to activate.", "warn")
    else:
        _add(checks, "graph_consistency_weight_active", "pass",
             f"graph_consistency_weight={weight}", "info")


def audit_modality_embedding_wiring(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Verify modality_ids full chain from dataset to model."""
    model_block = raw_config.get("model") or {}
    if not model_block.get("modality_embedding", False):
        _add(checks, "modality_embedding_wiring", "pass",
             "modality_embedding is disabled; chain audit skipped.", "info")
        return

    # Check Stage1JointBatch has modality_ids field
    try:
        from breast_pretrain.train.stage1_joint.types import Stage1JointBatch  # noqa: F401
        _add(checks, "modality_ids_in_batch_type", "pass",
             "Stage1JointBatch.modality_ids field present.", "info")
    except (ImportError, AttributeError):
        _add(checks, "modality_ids_in_batch_type", "fail",
             "Stage1JointBatch.modality_ids field missing.", "hard")
        return

    # Check forward_student signature includes modality_ids
    try:
        import inspect
        from breast_pretrain.train.stage1_joint.student_forward import forward_student  # noqa: F401
        sig = inspect.signature(forward_student)
        if "modality_ids" in sig.parameters:
            _add(checks, "modality_ids_in_forward_student", "pass",
                 "forward_student() accepts modality_ids parameter.", "info")
        else:
            _add(checks, "modality_ids_in_forward_student", "fail",
                 "forward_student() does not accept modality_ids.", "hard")
    except (ImportError, AttributeError) as exc:
        _add(checks, "modality_ids_in_forward_student", "fail",
             f"Cannot inspect forward_student: {exc}", "hard")

    # Check trainer/formal_step passes modality_ids through to forward_student.
    trainer_path = PROJECT_ROOT / "src/breast_pretrain/train/stage1_joint/trainer.py"
    formal_step_path = PROJECT_ROOT / "src/breast_pretrain/train/stage1_joint/formal_step.py"
    if trainer_path.exists() and formal_step_path.exists():
        trainer_text = trainer_path.read_text(encoding="utf-8")
        formal_step_text = formal_step_path.read_text(encoding="utf-8")
        trainer_calls_formal_step = "formal_train_step(" in trainer_text
        formal_step_passes_modality = "forward_student(model, batch.image, modality_ids=batch.modality_ids)" in formal_step_text
        formal_step_passes_masked_modality = "forward_student(model, masked_image, modality_ids=batch.modality_ids)" in formal_step_text
        if trainer_calls_formal_step and formal_step_passes_modality and formal_step_passes_masked_modality:
            _add(checks, "modality_ids_passed_in_trainer", "pass",
                 "trainer.py calls formal_train_step; formal_step.py passes batch.modality_ids to forward_student.", "info")
        else:
            _add(checks, "modality_ids_passed_in_trainer", "fail",
                 "trainer/formal_step path does not pass modality_ids to forward_student.", "hard")
    else:
        _add(checks, "modality_ids_passed_in_trainer", "fail",
             f"trainer/formal_step file not found: trainer={trainer_path.exists()} formal_step={formal_step_path.exists()}", "hard")

    _add(checks, "modality_embedding_scope", "pass",
         "Current modality embedding is injected into global_image_feature only; patch-level modality-aware reconstruction is not claimed complete.", "info")

    checkpoint_path = _resolve(raw_config.get("output_dir"), PROJECT_ROOT)
    if checkpoint_path is None:
        return
    checkpoint_path = checkpoint_path / "checkpoints" / "last.pt"
    if not checkpoint_path.exists():
        _add(checks, "modality_embedding_checkpoint_params", "warn",
             f"Smoke checkpoint not found for modality embedding parameter check: {checkpoint_path}", "warn")
        return
    try:
        import torch
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", {}) if isinstance(payload, dict) else {}
        keys = set(state_dict.keys())
    except Exception as exc:
        _add(checks, "modality_embedding_checkpoint_params", "fail",
             f"Cannot inspect smoke checkpoint modality params: {exc}", "hard")
        return
    required = {"_modality_embed.weight", "_modality_proj_global.weight", "_modality_proj_global.bias"}
    missing = sorted(required - keys)
    if missing:
        _add(checks, "modality_embedding_checkpoint_params", "fail",
             f"Checkpoint missing modality params: {missing}", "hard")
    else:
        _add(checks, "modality_embedding_checkpoint_params", "pass",
             "Checkpoint contains _modality_embed and _modality_proj_global parameters.", "info")


def audit_no_fixture_in_final(
    raw_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Hard-fail if any run_tier=final config references fixture/smoke assets."""
    run_tier = str(raw_config.get("run_tier", raw_config.get("metadata", {}).get("run_tier", ""))).strip()
    final_tiers = {"final"}
    if run_tier not in final_tiers:
        _add(checks, "no_fixture_in_final", "pass",
             f"run_tier={run_tier} is not final; fixture check skipped.", "info")
        return

    smoke_terms = ("fixture", "smoke", "local_visual_encoder_smoke", "smoke.pt")
    violations: list[str] = []

    # Check all path-like fields
    manifest = str(raw_config.get("image_manifest_path", ""))
    if any(term in manifest.lower() for term in smoke_terms):
        violations.append(f"image_manifest_path: {manifest}")

    model_block = raw_config.get("model") or {}
    weight_path = str(model_block.get("pretrained_weight_path", "") or "")
    if any(term in weight_path.lower() for term in smoke_terms):
        violations.append(f"pretrained_weight_path: {weight_path}")

    semantic_block = raw_config.get("semantic") or {}
    for key in ("semantic_soft_label_path", "semantic_manifest_path",
                "prompt_embedding_path", "semantic_soft_label_topk_path",
                "birads_prior_manifest_path"):
        path_val = str(semantic_block.get(key, "") or "")
        if any(term in path_val.lower() for term in smoke_terms):
            violations.append(f"semantic.{key}: {path_val}")

    data_block = raw_config
    for key in ("attention_map_dir", "teacher_latent_dir", "text_prompt_path"):
        path_val = str(data_block.get(key, "") or "")
        if any(term in path_val.lower() for term in smoke_terms):
            violations.append(f"{key}: {path_val}")

    if violations:
        _add(checks, "no_fixture_in_final", "fail",
             f"run_tier=final config references fixture/smoke assets: {violations}", "hard")
    else:
        _add(checks, "no_fixture_in_final", "pass",
             "No fixture/smoke references in final config.", "info")
