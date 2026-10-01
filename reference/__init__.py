# ── V5 Mainline: Teacher latent supervision is OPTIONAL / LEGACY / ABLATION ──
# The teachers/ package is kept for regression surface and ablation studies only.
# Default Stage 1 production configs use no-teacher path (self_masked_reconstruction).
# deterministic_fixture_teacher and real_clip_teacher are NOT part of the V5 mainline.

from breast_pretrain.teachers.base import (
    SUPPORTED_TEACHER_SOURCE_TYPES,
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TEACHER_SOURCE_FALLBACK,
    TEACHER_SOURCE_MISSING,
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
    TeacherImageEncoder,
    TeacherLatentBatch,
)
from breast_pretrain.teachers.token_utils import (
    adapt_teacher_latent_dim,
    infer_patch_grid,
    resize_teacher_tokens_2d,
    reshape_teacher_tokens,
)


def create_teacher_encoder(*args, **kwargs):
    from breast_pretrain.teachers.factory import create_teacher_encoder as _factory

    return _factory(*args, **kwargs)


__all__ = [
    "SUPPORTED_TEACHER_SOURCE_TYPES",
    "TEACHER_SOURCE_DETERMINISTIC_FIXTURE",
    "TEACHER_SOURCE_REAL_CLIP_IMAGE",
    "TEACHER_SOURCE_FALLBACK",
    "TEACHER_SOURCE_MISSING",
    "TeacherImageEncoder",
    "TeacherLatentBatch",
    "create_teacher_encoder",
    "infer_patch_grid",
    "reshape_teacher_tokens",
    "resize_teacher_tokens_2d",
    "adapt_teacher_latent_dim",
]
