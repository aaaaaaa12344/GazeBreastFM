from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from breast_pretrain.teachers.base import (
    TEACHER_SOURCE_DETERMINISTIC_FIXTURE,
    TEACHER_SOURCE_FALLBACK,
    TEACHER_SOURCE_MISSING,
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
)
from breast_pretrain.teachers.token_utils import (
    adapt_teacher_latent_dim,
    resize_teacher_tokens_2d,
    reshape_teacher_tokens,
)


GAZE_LOSS_MODES = {
    "no_gaze",
    "soft_attention",
    "high_conf_mask",
    "soft_attention_plus_high_conf",
    "random_prior",
    "center_prior",
    "shuffled_gaze",
    "breast_tissue_prior",
    "shuffled_gaze_within_breast",
}

TEACHER_LATENT_SOURCE_OFFLINE_NPY = TEACHER_SOURCE_DETERMINISTIC_FIXTURE
TEACHER_LATENT_SOURCE_FALLBACK = TEACHER_SOURCE_FALLBACK
TEACHER_LATENT_SOURCE_MISSING = TEACHER_SOURCE_MISSING
TEACHER_LATENT_SOURCE_REAL_CLIP = TEACHER_SOURCE_REAL_CLIP_IMAGE

MASK_STRATEGIES = {
    "random",
    "gaze_biased",
    "high_conf_quota",
    "tissue_aware",
    "mixed_random_gaze",
    "adaptive_gaze_masking",
}

PATCH_MASK_PRIOR_MODES = {
    "observed_gaze",
    "center_prior",
    "shuffled_gaze",
}


def validate_gaze_loss_mode(gaze_loss_mode: str) -> str:
    normalized = str(gaze_loss_mode).strip().lower()
    if normalized not in GAZE_LOSS_MODES:
        supported = ", ".join(sorted(GAZE_LOSS_MODES))
        raise ValueError(
            f"Unsupported gaze_loss_mode '{gaze_loss_mode}'. Expected one of: {supported}."
        )
    return normalized


def validate_mask_strategy(mask_strategy: str) -> str:
    normalized = str(mask_strategy).strip().lower()
    if normalized not in MASK_STRATEGIES:
        supported = ", ".join(sorted(MASK_STRATEGIES))
        raise ValueError(
            f"Unsupported mask_strategy '{mask_strategy}'. Expected one of: {supported}."
        )
    return normalized


def validate_patch_mask_prior_mode(prior_mode: str) -> str:
    normalized = str(prior_mode).strip().lower()
    if normalized not in PATCH_MASK_PRIOR_MODES:
        supported = ", ".join(sorted(PATCH_MASK_PRIOR_MODES))
        raise ValueError(
            f"Unsupported patch_mask_prior_mode '{prior_mode}'. Expected one of: {supported}."
        )
    return normalized


def compute_tensor_checksum(tensor: torch.Tensor) -> str:
    normalized = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("utf-8"))
    digest.update(str(normalized.dtype).encode("utf-8"))
    digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()[:16]


def expected_patch_count(image_size: int, patch_size: int) -> int:
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("image_size and patch_size must be positive.")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")
    grid_size = image_size // patch_size
    return grid_size * grid_size


def _num_masked_patches(num_patches: int, mask_ratio: float) -> int:
    mask_ratio = float(mask_ratio)
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1.")
    return max(1, min(num_patches, int(round(num_patches * mask_ratio))))


def _build_random_patch_mask(
    batch_size: int,
    num_patches: int,
    num_masked: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    scores = torch.rand(
        (batch_size, num_patches),
        generator=generator,
        dtype=torch.float32,
    )
    sorted_indices = torch.argsort(scores, dim=1)
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    mask.scatter_(1, sorted_indices[:, :num_masked], True)
    return mask


def _validate_patch_prior(
    prior: torch.Tensor | None,
    expected_shape: tuple[int, int],
    field_name: str,
) -> torch.Tensor | None:
    if prior is None:
        return None
    if tuple(prior.shape) != expected_shape:
        raise ValueError(
            f"{field_name} must have shape {expected_shape}, got {tuple(prior.shape)}."
        )
    return prior.detach().cpu().to(dtype=torch.float32).clamp_min(0.0)


def _build_center_prior_tokens(
    batch_size: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    row_coords = torch.arange(grid_height, device=device, dtype=dtype)
    col_coords = torch.arange(grid_width, device=device, dtype=dtype)
    row_center = (float(grid_height) - 1.0) / 2.0
    col_center = (float(grid_width) - 1.0) / 2.0
    row_grid = row_coords.view(-1, 1).expand(grid_height, grid_width)
    col_grid = col_coords.view(1, -1).expand(grid_height, grid_width)
    distance = torch.sqrt((row_grid - row_center).pow(2) + (col_grid - col_center).pow(2))
    max_distance = float(distance.max().item())
    if max_distance > 0.0:
        center_template = 1.0 - (distance / max_distance)
    else:
        center_template = torch.ones_like(distance)
    return center_template.reshape(1, -1).expand(batch_size, -1)


def _shuffle_spatial_prior_batch(
    prior_map: torch.Tensor,
    random_seed: int,
) -> torch.Tensor:
    if prior_map.ndim != 4:
        raise ValueError(f"prior_map must have shape [batch, channels, height, width], got {tuple(prior_map.shape)}.")
    if prior_map.shape[0] > 1:
        return torch.roll(prior_map, shifts=1, dims=0)

    flattened = prior_map.reshape(prior_map.shape[0], prior_map.shape[1], -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    permutation = torch.randperm(int(flattened.shape[-1]), generator=generator)
    return flattened[:, :, permutation].reshape_as(prior_map)


def _build_sampling_scores(
    batch_size: int,
    num_patches: int,
    patch_gaze_scores: torch.Tensor | None,
    high_conf_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    expected_shape = (batch_size, num_patches)
    gaze_scores = _validate_patch_prior(
        patch_gaze_scores,
        expected_shape=expected_shape,
        field_name="patch_gaze_scores",
    )
    high_conf_scores = _validate_patch_prior(
        high_conf_mask,
        expected_shape=expected_shape,
        field_name="high_conf_mask",
    )

    if gaze_scores is None and high_conf_scores is None:
        return None
    if gaze_scores is None:
        return high_conf_scores
    if high_conf_scores is None:
        return gaze_scores
    return gaze_scores + high_conf_scores


def _has_positive_prior(scores: torch.Tensor, eps: float) -> bool:
    return bool((scores.sum(dim=1) > float(eps)).all().item())


def build_gaze_biased_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    patch_gaze_scores: torch.Tensor | None,
    high_conf_mask: torch.Tensor | None = None,
    gaze_mask_sampling_alpha: float = 0.7,
    min_random_mask_fraction: float = 0.3,
    mask_sampling_temperature: float = 1.0,
    gaze_mask_eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    num_masked = _num_masked_patches(num_patches=num_patches, mask_ratio=mask_ratio)
    scores = _build_sampling_scores(
        batch_size=batch_size,
        num_patches=num_patches,
        patch_gaze_scores=patch_gaze_scores,
        high_conf_mask=high_conf_mask,
    )
    if scores is None or not _has_positive_prior(scores, eps=gaze_mask_eps):
        warnings.warn(
            "gaze_biased mask sampling fell back to random because gaze priors are missing or all zero.",
            stacklevel=2,
        )
        return _build_random_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            num_masked=num_masked,
            generator=generator,
        )

    alpha = min(max(float(gaze_mask_sampling_alpha), 0.0), 1.0)
    min_random_fraction = min(max(float(min_random_mask_fraction), 0.0), 1.0)
    gaze_fraction = min(alpha, 1.0 - min_random_fraction)
    random_fraction = 1.0 - gaze_fraction
    temperature = max(float(mask_sampling_temperature), float(gaze_mask_eps))

    adjusted_scores = scores.clamp_min(0.0).pow(1.0 / temperature)
    score_sums = adjusted_scores.sum(dim=1, keepdim=True).clamp_min(float(gaze_mask_eps))
    gaze_prob = adjusted_scores / score_sums
    uniform_prob = torch.full_like(gaze_prob, 1.0 / float(num_patches))
    sampling_prob = random_fraction * uniform_prob + gaze_fraction * gaze_prob

    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    for batch_index in range(batch_size):
        selected = torch.multinomial(
            sampling_prob[batch_index],
            num_samples=num_masked,
            replacement=False,
            generator=generator,
        )
        mask[batch_index, selected] = True
    return mask


def build_high_conf_quota_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    high_conf_mask: torch.Tensor | None,
    high_conf_mask_quota: float = 0.6,
    min_random_mask_fraction: float = 0.3,
    generator: torch.Generator | None = None,
    gaze_mask_eps: float = 1e-6,
) -> torch.Tensor:
    num_masked = _num_masked_patches(num_patches=num_patches, mask_ratio=mask_ratio)
    high_conf_scores = _validate_patch_prior(
        high_conf_mask,
        expected_shape=(batch_size, num_patches),
        field_name="high_conf_mask",
    )
    if high_conf_scores is None or not _has_positive_prior(high_conf_scores, eps=gaze_mask_eps):
        warnings.warn(
            "high_conf_quota mask sampling fell back to random because high-conf priors are missing or all zero.",
            stacklevel=2,
        )
        return _build_random_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            num_masked=num_masked,
            generator=generator,
        )

    quota = min(max(float(high_conf_mask_quota), 0.0), 1.0)
    min_random_fraction = min(max(float(min_random_mask_fraction), 0.0), 1.0)
    min_random_count = min(num_masked, int(math.ceil(num_masked * min_random_fraction)))
    high_conf_target = min(
        int(round(num_masked * quota)),
        max(0, num_masked - min_random_count),
    )

    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    all_indices = torch.arange(num_patches, dtype=torch.long)
    for batch_index in range(batch_size):
        high_indices = torch.nonzero(
            high_conf_scores[batch_index] > float(gaze_mask_eps),
            as_tuple=False,
        ).flatten()
        sample_high_count = min(high_conf_target, int(high_indices.numel()))
        if sample_high_count > 0:
            order = torch.randperm(int(high_indices.numel()), generator=generator)
            selected_high = high_indices[order[:sample_high_count]]
            mask[batch_index, selected_high] = True

        remaining_count = num_masked - int(mask[batch_index].sum().item())
        if remaining_count > 0:
            remaining_indices = all_indices[~mask[batch_index]]
            order = torch.randperm(int(remaining_indices.numel()), generator=generator)
            selected_remaining = remaining_indices[order[:remaining_count]]
            mask[batch_index, selected_remaining] = True

    return mask


def generate_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    fixed_mask: torch.Tensor | None = None,
    mask_strategy: str = "random",
    patch_gaze_scores: torch.Tensor | None = None,
    high_conf_mask: torch.Tensor | None = None,
    gaze_mask_sampling_alpha: float = 0.7,
    high_conf_mask_quota: float = 0.6,
    min_random_mask_fraction: float = 0.3,
    mask_sampling_temperature: float = 1.0,
    gaze_mask_eps: float = 1e-6,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_patches <= 0:
        raise ValueError("num_patches must be positive.")

    mask_ratio = float(mask_ratio)
    num_masked = _num_masked_patches(num_patches=num_patches, mask_ratio=mask_ratio)
    normalized_strategy = validate_mask_strategy(mask_strategy)

    if fixed_mask is not None:
        expected_shape = (batch_size, num_patches)
        if tuple(fixed_mask.shape) != expected_shape:
            raise ValueError(
                f"fixed_mask must have shape {expected_shape}, got {tuple(fixed_mask.shape)}."
            )
        return fixed_mask.to(device=device, dtype=torch.bool)

    if normalized_strategy == "random":
        mask = _build_random_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            num_masked=num_masked,
            generator=generator,
        )
    elif normalized_strategy in {"gaze_biased", "mixed_random_gaze", "tissue_aware"}:
        mask = build_gaze_biased_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=mask_ratio,
            patch_gaze_scores=patch_gaze_scores,
            high_conf_mask=high_conf_mask,
            gaze_mask_sampling_alpha=(
                0.5 if normalized_strategy == "mixed_random_gaze" else gaze_mask_sampling_alpha
            ),
            min_random_mask_fraction=min_random_mask_fraction,
            mask_sampling_temperature=mask_sampling_temperature,
            gaze_mask_eps=gaze_mask_eps,
            generator=generator,
        )
    else:
        mask = build_high_conf_quota_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=mask_ratio,
            high_conf_mask=high_conf_mask,
            high_conf_mask_quota=high_conf_mask_quota,
            min_random_mask_fraction=min_random_mask_fraction,
            generator=generator,
            gaze_mask_eps=gaze_mask_eps,
        )
    return mask.to(device=device)


def build_patch_mask_sampling_priors(
    attention_map: torch.Tensor,
    high_conf_mask: torch.Tensor,
    patch_size: int,
    prior_mode: str = "observed_gaze",
    random_seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_map.ndim != 4 or high_conf_mask.ndim != 4:
        raise ValueError("attention_map and high_conf_mask must have shape [batch, 1, height, width].")
    normalized_prior_mode = validate_patch_mask_prior_mode(prior_mode)

    if normalized_prior_mode == "observed_gaze":
        pooled_attention = F.avg_pool2d(attention_map, kernel_size=patch_size, stride=patch_size)
        pooled_high_conf = F.avg_pool2d(high_conf_mask, kernel_size=patch_size, stride=patch_size)
        return pooled_attention.flatten(1), pooled_high_conf.flatten(1).clamp_(0.0, 1.0)

    pooled_template = F.avg_pool2d(attention_map, kernel_size=patch_size, stride=patch_size)
    batch_size = int(pooled_template.shape[0])
    grid_height = int(pooled_template.shape[-2])
    grid_width = int(pooled_template.shape[-1])

    if normalized_prior_mode == "center_prior":
        center_tokens = _build_center_prior_tokens(
            batch_size=batch_size,
            grid_height=grid_height,
            grid_width=grid_width,
            device=attention_map.device,
            dtype=attention_map.dtype,
        )
        return center_tokens, torch.zeros_like(center_tokens)

    shuffled_attention = _shuffle_spatial_prior_batch(
        attention_map,
        random_seed=int(random_seed),
    )
    shuffled_high_conf = _shuffle_spatial_prior_batch(
        high_conf_mask,
        random_seed=int(random_seed) + 1,
    )
    pooled_attention = F.avg_pool2d(shuffled_attention, kernel_size=patch_size, stride=patch_size)
    pooled_high_conf = F.avg_pool2d(shuffled_high_conf, kernel_size=patch_size, stride=patch_size)
    return pooled_attention.flatten(1), pooled_high_conf.flatten(1).clamp_(0.0, 1.0)


def build_patch_gaze_weights(
    attention_map: torch.Tensor,
    high_conf_mask: torch.Tensor,
    patch_size: int,
    gaze_loss_mode: str = "soft_attention_plus_high_conf",
    gaze_weight_alpha: float = 1.0,
    random_seed: int = 42,
) -> tuple[torch.Tensor, dict[str, float]]:
    if attention_map.ndim != 4 or high_conf_mask.ndim != 4:
        raise ValueError("attention_map and high_conf_mask must have shape [batch, 1, height, width].")

    normalized_mode = validate_gaze_loss_mode(gaze_loss_mode)
    alpha = float(gaze_weight_alpha)
    pooled_attention = F.avg_pool2d(attention_map, kernel_size=patch_size, stride=patch_size)
    pooled_mask = F.avg_pool2d(high_conf_mask, kernel_size=patch_size, stride=patch_size)
    attention_tokens = pooled_attention.flatten(1)
    mask_tokens = pooled_mask.flatten(1).clamp_(0.0, 1.0)
    grid_height = int(pooled_attention.shape[-2])
    grid_width = int(pooled_attention.shape[-1])

    if normalized_mode in {"breast_tissue_prior", "shuffled_gaze_within_breast"}:
        raise ValueError(
            f"gaze_loss_mode '{normalized_mode}' requires image-aware prior construction "
            "and must be handled by the Stage 1 tissue-aware helper."
        )

    if normalized_mode == "random_prior":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))
        template = torch.rand(
            (1, attention_tokens.shape[1]),
            generator=generator,
            dtype=attention_tokens.dtype,
        ).to(device=attention_tokens.device)
        prior_tokens = template.expand_as(attention_tokens)
    elif normalized_mode == "center_prior":
        prior_tokens = _build_center_prior_tokens(
            batch_size=int(attention_tokens.shape[0]),
            grid_height=grid_height,
            grid_width=grid_width,
            device=attention_tokens.device,
            dtype=attention_tokens.dtype,
        )
    elif normalized_mode == "shuffled_gaze":
        shuffled_attention = _shuffle_spatial_prior_batch(
            attention_map,
            random_seed=int(random_seed),
        )
        shuffled_mask = _shuffle_spatial_prior_batch(
            high_conf_mask,
            random_seed=int(random_seed) + 1,
        )
        attention_tokens = F.avg_pool2d(
            shuffled_attention,
            kernel_size=patch_size,
            stride=patch_size,
        ).flatten(1)
        mask_tokens = F.avg_pool2d(
            shuffled_mask,
            kernel_size=patch_size,
            stride=patch_size,
        ).flatten(1).clamp_(0.0, 1.0)
        prior_tokens = None
    else:
        prior_tokens = None

    if normalized_mode == "no_gaze":
        weights = torch.ones_like(attention_tokens)
    elif normalized_mode == "soft_attention":
        weights = 1.0 + alpha * attention_tokens
    elif normalized_mode == "high_conf_mask":
        weights = 1.0 + alpha * mask_tokens
    elif normalized_mode in {"random_prior", "center_prior"}:
        weights = 1.0 + alpha * prior_tokens
    else:
        weights = 1.0 + alpha * (attention_tokens + mask_tokens)

    summary = {
        "gaze_loss_mode": normalized_mode,
        "gaze_weight_alpha": alpha,
        "patch_weight_min": float(weights.min().item()),
        "patch_weight_max": float(weights.max().item()),
        "patch_weight_mean": float(weights.mean().item()),
        "attention_min": float(attention_map.min().item()),
        "attention_max": float(attention_map.max().item()),
        "attention_mean": float(attention_map.mean().item()),
        "high_conf_min": float(high_conf_mask.min().item()),
        "high_conf_max": float(high_conf_mask.max().item()),
        "high_conf_mean": float(high_conf_mask.mean().item()),
    }
    return weights, summary


def build_deterministic_teacher_latent(
    image: torch.Tensor,
    patch_size: int,
    latent_dim: int,
) -> torch.Tensor:
    batch_size, channels, _, _ = image.shape
    unfolded = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    num_patches = unfolded.shape[-1]
    patch_area = patch_size * patch_size
    patches = unfolded.transpose(1, 2).reshape(batch_size, num_patches, channels, patch_area)

    patch_mean = patches.mean(dim=-1)
    patch_std = patches.std(dim=-1, unbiased=False)
    patch_energy = patches.square().mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
    features = torch.cat((patch_mean, patch_std, patch_energy), dim=-1)

    feature_dim = features.shape[-1]
    feature_index = torch.arange(
        1,
        feature_dim + 1,
        device=image.device,
        dtype=image.dtype,
    ).view(feature_dim, 1)
    latent_index = torch.arange(
        1,
        latent_dim + 1,
        device=image.device,
        dtype=image.dtype,
    ).view(1, latent_dim)
    projection = torch.sin(feature_index * latent_index / float(latent_dim + 1))
    projection = projection + torch.cos(feature_index * latent_index / float(feature_dim + 1))

    position_index = torch.arange(
        1,
        num_patches + 1,
        device=image.device,
        dtype=image.dtype,
    ).view(1, num_patches, 1)
    position_phase = torch.sin(position_index * latent_index.view(1, 1, latent_dim) / float(num_patches + 1))
    return torch.matmul(features, projection) + position_phase


def load_teacher_latents(
    image: torch.Tensor,
    image_ids: list[str],
    teacher_latent_paths: list[str],
    teacher_source_types: list[str] | None,
    patch_size: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[bool]]:
    if image.ndim != 4:
        raise ValueError("image must have shape [batch, channels, height, width].")

    dummy_latents = build_deterministic_teacher_latent(
        image,
        patch_size=patch_size,
        latent_dim=latent_dim,
    )
    latents: list[torch.Tensor] = []
    sources: list[str] = []
    missing_flags: list[bool] = []

    expected_patches = dummy_latents.shape[1]
    for index, raw_path in enumerate(teacher_latent_paths):
        image_id = image_ids[index]
        teacher_path = Path(str(raw_path)).expanduser() if str(raw_path).strip() else None
        if teacher_path is None or not teacher_path.exists():
            latents.append(dummy_latents[index])
            sources.append(TEACHER_LATENT_SOURCE_FALLBACK)
            missing_flags.append(True)
            continue

        if teacher_path.suffix.lower() != ".npy":
            raise ValueError(
                f"Teacher latent for {image_id} must be a .npy file, got: {teacher_path}"
            )

        loaded = np.load(teacher_path, allow_pickle=False)
        tokens = reshape_teacher_tokens(
            loaded,
            image_id=image_id,
            teacher_path=teacher_path,
        )
        tokens = resize_teacher_tokens_2d(
            tokens,
            expected_patch_count=expected_patches,
            image_id=image_id,
            teacher_path=teacher_path,
        )
        tokens = adapt_teacher_latent_dim(
            tokens,
            latent_dim=latent_dim,
            image_id=image_id,
            teacher_path=teacher_path,
        )
        latents.append(tokens.to(device=device, dtype=image.dtype))
        source_type = (
            str(teacher_source_types[index]).strip()
            if teacher_source_types is not None and index < len(teacher_source_types)
            else ""
        )
        sources.append(source_type or TEACHER_SOURCE_DETERMINISTIC_FIXTURE)
        missing_flags.append(False)

    teacher_latent = torch.stack(latents, dim=0).to(device=device, dtype=image.dtype)
    return teacher_latent, sources, missing_flags


def compute_masked_latent_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_weights: torch.Tensor,
) -> torch.Tensor:
    if student_latent.shape != teacher_latent.shape:
        raise ValueError(
            f"student_latent and teacher_latent must have the same shape, got "
            f"{tuple(student_latent.shape)} and {tuple(teacher_latent.shape)}"
        )
    if patch_mask.shape != patch_weights.shape:
        raise ValueError(
            f"patch_mask and patch_weights must have the same shape, got "
            f"{tuple(patch_mask.shape)} and {tuple(patch_weights.shape)}"
        )

    per_patch_loss = (student_latent - teacher_latent).pow(2).mean(dim=-1)
    effective_weights = patch_weights * patch_mask.to(dtype=patch_weights.dtype)
    normalizer = effective_weights.sum().clamp_min(1.0)
    return (per_patch_loss * effective_weights).sum() / normalizer


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    image = batch["image"]
    attention_map = batch["attention_map"]
    high_conf_mask = batch["high_conf_mask"]
    return {
        "batch_size": int(image.shape[0]),
        "image_shape": tuple(image.shape),
        "attention_shape": tuple(attention_map.shape),
        "high_conf_shape": tuple(high_conf_mask.shape),
        "modalities": sorted(set(str(item) for item in batch["modality"])),
        "teacher_latent_available": int(
            torch.as_tensor(batch["teacher_latent_exists"], dtype=torch.int32).sum().item()
        ),
    }
