from __future__ import annotations

import torch

from breast_pretrain.teachers.base import (
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
    TeacherImageEncoder,
)
from breast_pretrain.teachers.deterministic_fixture_teacher import (
    DeterministicFixtureTeacher,
)
from breast_pretrain.teachers.real_clip_image_teacher import RealClipImageTeacher
from breast_pretrain.utils.config import TeacherConfig


def create_teacher_encoder(
    teacher_config: TeacherConfig,
    patch_size: int,
    latent_dim: int,
    device: torch.device,
) -> TeacherImageEncoder:
    source_type = str(teacher_config.source_type).strip().lower()
    if source_type == TEACHER_SOURCE_DETERMINISTIC_FIXTURE:
        return DeterministicFixtureTeacher(
            patch_size=patch_size,
            latent_dim=latent_dim,
        )
    if source_type == TEACHER_SOURCE_REAL_CLIP_IMAGE:
        return RealClipImageTeacher(
            model_name=teacher_config.model_name or "",
            device=device,
            local_files_only=teacher_config.local_files_only,
        )
    raise ValueError(f"Unsupported teacher.source_type: {teacher_config.source_type}")
