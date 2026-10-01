from __future__ import annotations

from pathlib import Path
from typing import Any

from breast_pretrain.train.stage1_joint.types import Stage1JointTrainerConfig


def audit_joint_pretrain_config(config: Stage1JointTrainerConfig) -> dict[str, Any]:
    warnings: list[str] = []
    reference_paths = {
        "panderm_repo_path": config.references.panderm_repo_path,
        "fgclip_repo_path": config.references.fgclip_repo_path,
        "cogaze_repo_path": config.references.cogaze_repo_path,
        "ultrasound_clip_repo_path": config.references.ultrasound_clip_repo_path,
    }
    for label, path in reference_paths.items():
        if path is None:
            continue
        if not Path(path).exists():
            warnings.append(f"external_reference_missing_warn_only:{label}:{path}")
    if config.references.import_policy != "read_only_reference_only":
        raise ValueError("External references must stay read_only_reference_only.")
    teacher_block = config.teacher or {}
    if teacher_block:
        if str(teacher_block.get("local_files_only", True)).strip().lower() in {"false", "0", "no"}:
            warnings.append("teacher_local_files_only_disabled")
    return {
        "status": "pass_with_warnings" if warnings else "pass",
        "warnings": warnings,
        "import_policy": config.references.import_policy,
        "v5_mainline_note": (
            "Teacher latent supervision is optional/legacy/ablation. "
            "Default V5 mainline uses require_teacher_latents=false "
            "with self_masked_reconstruction (no teacher)."
        ),
    }
