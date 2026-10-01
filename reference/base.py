from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


TEACHER_SOURCE_DETERMINISTIC_FIXTURE = "deterministic_fixture_teacher"
TEACHER_SOURCE_REAL_CLIP_IMAGE = "real_clip_image_teacher"
TEACHER_SOURCE_FALLBACK = "fallback_dummy_image_latent"
TEACHER_SOURCE_MISSING = "missing_teacher_latent"

SUPPORTED_TEACHER_SOURCE_TYPES = {
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
}


@dataclass(frozen=True)
class TeacherLatentBatch:
    tokens: torch.Tensor
    teacher_model_name: str
    source_type: str
    raw_patch_grid: tuple[int, int]


class TeacherImageEncoder(Protocol):
    source_type: str
    teacher_model_name: str

    def build_latents(self, image: torch.Tensor) -> TeacherLatentBatch:
        ...
