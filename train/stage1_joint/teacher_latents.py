from __future__ import annotations

import math
from collections import Counter

import torch
from torch.nn import functional as F

from breast_pretrain.train.masked_latent_smoke import load_teacher_latents
from breast_pretrain.train.stage1_joint.types import (
    Stage1JointBatch,
    Stage1JointTrainerConfig,
    TeacherLatentBatch,
)
from breast_pretrain.train.teacher_guard import assert_required_teacher_latents

RECONSTRUCTION_SOURCE_SELF = "self_masked_reconstruction"
RECONSTRUCTION_SOURCE_TEACHER_NPY = "teacher_latent_npy"


def build_image_patch_reconstruction_target(
    image: torch.Tensor,
    patch_size: int,
    latent_dim: int,
) -> torch.Tensor:
    """No-teacher visual reconstruction target from normalized image patches.

    Unfolds the raw image into patches, normalises each patch to zero-mean
    unit-variance, and projects to ``latent_dim`` via a fixed deterministic
    DCT-like cosine basis.  This is *not* a teacher latent, a CLIP teacher,
    or a deterministic fixture — it is a hand-crafted image-patch feature
    transform that provides a stable self-supervised reconstruction target
    for the V5 no-teacher mainline.
    """
    batch_size, channels, _, _ = image.shape
    patch_area = patch_size * patch_size
    unfolded = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    num_patches = unfolded.shape[-1]
    patches = unfolded.transpose(1, 2).reshape(batch_size, num_patches, channels, patch_area)

    flat_patches = patches.reshape(batch_size, num_patches, channels * patch_area)
    patch_mean = flat_patches.mean(dim=-1, keepdim=True)
    patch_std = flat_patches.std(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = (flat_patches - patch_mean) / patch_std

    in_dim = channels * patch_area
    i_idx = torch.arange(in_dim, dtype=image.dtype, device=image.device).view(-1, 1)
    j_idx = torch.arange(latent_dim, dtype=image.dtype, device=image.device).view(1, -1)
    basis = torch.cos(math.pi * (i_idx + 0.5) * (j_idx + 1) / max(in_dim, latent_dim))

    return normalized @ basis


def build_no_teacher_latent_batch(
    batch: Stage1JointBatch,
    config: Stage1JointTrainerConfig,
    device: torch.device,
) -> TeacherLatentBatch:
    """Build a reconstruction-target batch for the no-teacher mainline.

    Returns a ``TeacherLatentBatch`` whose ``teacher_latent`` field holds
    the *image-patch reconstruction target* (not a teacher latent).
    Every sample is tagged with source ``"self_masked_reconstruction"``.
    """
    target = build_image_patch_reconstruction_target(
        image=batch.image,
        patch_size=config.model.patch_size,
        latent_dim=config.model.output_patch_dim or config.model.latent_dim,
    )
    batch_size = int(batch.image.shape[0])
    return TeacherLatentBatch(
        teacher_latent=target.to(device=device, dtype=batch.image.dtype),
        teacher_sources=[RECONSTRUCTION_SOURCE_SELF] * batch_size,
        missing_teacher_flags=[False] * batch_size,
    )


def validate_dataset_teacher_latents(
    dataset: object,
    require_teacher_latents: bool,
) -> dict[str, object]:
    summary = dataset.summarize_teacher_latent_availability()
    assert_required_teacher_latents(
        require_teacher_latents=require_teacher_latents,
        teacher_latent_summary=summary,
        context="stage1_joint trainer dataset teacher_latent_availability summary",
    )
    return summary


def load_teacher_latent_batch(
    batch: Stage1JointBatch,
    config: Stage1JointTrainerConfig,
    device: torch.device,
) -> TeacherLatentBatch:
    """Load teacher latents from .npy cache (legacy / ablation only).

    Raises ``RuntimeError`` when called under ``self_masked_reconstruction``
    to prevent accidental teacher-latent usage on the V5 no-teacher mainline.
    """
    source = config.semantic.reconstruction_teacher_source
    if source == RECONSTRUCTION_SOURCE_SELF:
        raise RuntimeError(
            "load_teacher_latent_batch must NOT be called when "
            "reconstruction_teacher_source=self_masked_reconstruction. "
            "Use build_no_teacher_latent_batch for the V5 no-teacher mainline."
        )

    teacher_latent, teacher_sources, missing_teacher_flags = load_teacher_latents(
        image=batch.image,
        image_ids=batch.image_ids,
        teacher_latent_paths=batch.teacher_latent_paths,
        teacher_source_types=batch.teacher_source_types,
        patch_size=config.model.patch_size,
        latent_dim=config.model.output_patch_dim or config.model.latent_dim,
        device=device,
    )
    return TeacherLatentBatch(
        teacher_latent=teacher_latent,
        teacher_sources=teacher_sources,
        missing_teacher_flags=[bool(item) for item in missing_teacher_flags],
    )


def build_teacher_latent_source_summary(
    teacher_sources: Counter[str],
    missing_teacher_latent_count: int,
) -> dict[str, int]:
    summary = {source_name: int(count) for source_name, count in sorted(teacher_sources.items())}
    summary["missing_teacher_latent_count"] = int(missing_teacher_latent_count)
    return summary


def validate_training_teacher_latents(
    config: Stage1JointTrainerConfig,
    teacher_latent_source_summary: dict[str, int],
) -> None:
    source = config.semantic.reconstruction_teacher_source
    if source == RECONSTRUCTION_SOURCE_SELF:
        return
    assert_required_teacher_latents(
        require_teacher_latents=config.data.require_teacher_latents,
        teacher_latent_summary=teacher_latent_source_summary,
        context="stage1_joint trainer training teacher_latent_source_summary",
    )
