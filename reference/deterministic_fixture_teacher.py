from __future__ import annotations

import torch

from breast_pretrain.teachers.base import (
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TeacherLatentBatch,
)
from breast_pretrain.teachers.token_utils import infer_patch_grid
from breast_pretrain.train.masked_latent_smoke import build_deterministic_teacher_latent


class DeterministicFixtureTeacher:
    source_type = TEACHER_SOURCE_DETERMINISTIC_FIXTURE
    teacher_model_name = "deterministic_fixture_teacher_v1"

    def __init__(self, patch_size: int, latent_dim: int) -> None:
        self.patch_size = int(patch_size)
        self.latent_dim = int(latent_dim)

    def build_latents(self, image: torch.Tensor) -> TeacherLatentBatch:
        tokens = build_deterministic_teacher_latent(
            image=image,
            patch_size=self.patch_size,
            latent_dim=self.latent_dim,
        )
        raw_patch_grid = infer_patch_grid(int(tokens.shape[1]))
        return TeacherLatentBatch(
            tokens=tokens.to(dtype=torch.float32),
            teacher_model_name=self.teacher_model_name,
            source_type=self.source_type,
            raw_patch_grid=raw_patch_grid,
        )
