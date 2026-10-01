# ── Legacy / Ablation Audit ──
# Teacher latent audit is kept for regression surface and ablation studies.
# V5 mainline default configs use require_teacher_latents=false.

from __future__ import annotations

from pathlib import Path
from typing import Any

from breast_pretrain.teachers.teacher_latent_cache import audit_teacher_latent_cache


def audit_joint_pretrain_teacher_latents(
    *,
    teacher_latent_dir: str | Path | None,
    expected_patch_count: int,
    expected_latent_dim: int,
) -> dict[str, Any]:
    if teacher_latent_dir is None:
        return {
            "status": "missing_teacher_latent_dir",
            "entry_count": 0,
            "missing_count": 0,
            "shape_mismatch_count": 0,
            "duplicate_checksum_count": 0,
            "unique_checksum_count": 0,
        }
    summary = audit_teacher_latent_cache(
        teacher_latent_dir=teacher_latent_dir,
        expected_patch_count=expected_patch_count,
        expected_latent_dim=expected_latent_dim,
    )
    summary["status"] = (
        "pass"
        if summary["missing_count"] == 0 and summary["shape_mismatch_count"] == 0
        else "warn"
    )
    return summary
