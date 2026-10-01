from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from breast_pretrain.train.masked_latent_smoke import (
    build_patch_mask_sampling_priors,
    generate_patch_mask,
    load_teacher_latents,
    validate_mask_strategy,
)
from breast_pretrain.train.reproducibility import (
    build_patch_mask_sequence,
    compute_mask_sequence_checksum,
)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _pool_patch_priors(
    attention_map: torch.Tensor,
    high_conf_mask: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_map.ndim != 4 or high_conf_mask.ndim != 4:
        raise ValueError("attention_map and high_conf_mask must have shape [batch, 1, height, width].")

    pooled_attention = F.avg_pool2d(attention_map, kernel_size=patch_size, stride=patch_size)
    pooled_high_conf = F.avg_pool2d(high_conf_mask, kernel_size=patch_size, stride=patch_size)
    return pooled_attention.flatten(1), pooled_high_conf.flatten(1)


def _build_attention_band_masks(
    attention_tokens: torch.Tensor,
    top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_tokens.ndim != 2:
        raise ValueError("attention_tokens must have shape [batch, num_patches].")

    num_patches = int(attention_tokens.shape[1])
    if num_patches <= 0:
        raise ValueError("attention_tokens must contain at least one patch.")
    k = max(1, int(math.ceil(num_patches * float(top_fraction))))

    high_indices = torch.topk(attention_tokens, k=k, dim=1, largest=True).indices
    low_indices = torch.topk(attention_tokens, k=k, dim=1, largest=False).indices

    high_mask = torch.zeros_like(attention_tokens, dtype=torch.bool)
    low_mask = torch.zeros_like(attention_tokens, dtype=torch.bool)
    high_mask.scatter_(1, high_indices, True)
    low_mask.scatter_(1, low_indices, True)
    return high_mask, low_mask


def _update_region_accumulator(
    per_patch_loss: torch.Tensor,
    patch_selector: torch.Tensor,
    loss_total: float,
    patch_count: int,
) -> tuple[float, int]:
    normalized_selector = patch_selector.to(dtype=torch.bool)
    selected_losses = per_patch_loss.masked_select(normalized_selector)
    return (
        loss_total + float(selected_losses.sum().item()),
        patch_count + int(selected_losses.numel()),
    )


def _safe_mean(loss_total: float, patch_count: int) -> float | None:
    if patch_count <= 0:
        return None
    return loss_total / float(patch_count)


def evaluate_masked_latent_reconstruction(
    model: torch.nn.Module,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    patch_size: int,
    latent_dim: int,
    mask_ratio: float,
    eval_seed: int,
    attention_top_fraction: float = 0.2,
    mask_strategy: str = "random",
    gaze_mask_sampling_alpha: float = 0.7,
    high_conf_mask_quota: float = 0.6,
    min_random_mask_fraction: float = 0.3,
    mask_sampling_temperature: float = 1.0,
    gaze_mask_eps: float = 1e-6,
) -> dict[str, object]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    num_batches = len(dataloader)
    if num_batches <= 0:
        raise ValueError("Evaluation dataloader is empty.")

    num_patches = int(model.num_patches)
    normalized_mask_strategy = validate_mask_strategy(mask_strategy)
    if normalized_mask_strategy == "random":
        precomputed_patch_masks, mask_checksum = build_patch_mask_sequence(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=mask_ratio,
            max_steps=num_batches,
            seed=eval_seed,
        )
        mask_generator = None
        eval_mask_policy = "deterministic_fixed_seed"
    else:
        precomputed_patch_masks = None
        mask_checksum = "computed_after_eval"
        mask_generator = torch.Generator(device="cpu")
        mask_generator.manual_seed(int(eval_seed))
        eval_mask_policy = f"deterministic_{normalized_mask_strategy}_seed"

    sample_losses: list[float] = []
    used_patch_masks: list[torch.Tensor] = []
    high_conf_loss_total = 0.0
    high_conf_patch_count = 0
    non_high_conf_loss_total = 0.0
    non_high_conf_patch_count = 0
    high_attention_loss_total = 0.0
    high_attention_patch_count = 0
    low_attention_loss_total = 0.0
    low_attention_patch_count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            batch = _to_device(dict(raw_batch), device=device)
            image = batch["image"]
            attention_map = batch["attention_map"]
            high_conf_mask = batch["high_conf_mask"]
            current_batch_size = int(image.shape[0])

            if precomputed_patch_masks is not None:
                fixed_mask = precomputed_patch_masks[batch_index]
                patch_mask = generate_patch_mask(
                    batch_size=current_batch_size,
                    num_patches=num_patches,
                    mask_ratio=mask_ratio,
                    device=device,
                    fixed_mask=fixed_mask[:current_batch_size],
                )
            else:
                patch_gaze_scores, high_conf_patch_mask_for_sampling = (
                    build_patch_mask_sampling_priors(
                        attention_map=attention_map,
                        high_conf_mask=high_conf_mask,
                        patch_size=patch_size,
                    )
                )
                patch_mask = generate_patch_mask(
                    batch_size=current_batch_size,
                    num_patches=num_patches,
                    mask_ratio=mask_ratio,
                    device=device,
                    generator=mask_generator,
                    mask_strategy=normalized_mask_strategy,
                    patch_gaze_scores=patch_gaze_scores,
                    high_conf_mask=high_conf_patch_mask_for_sampling,
                    gaze_mask_sampling_alpha=gaze_mask_sampling_alpha,
                    high_conf_mask_quota=high_conf_mask_quota,
                    min_random_mask_fraction=min_random_mask_fraction,
                    mask_sampling_temperature=mask_sampling_temperature,
                    gaze_mask_eps=gaze_mask_eps,
                )
            used_patch_masks.append(patch_mask.detach().cpu().clone())
            student_latent = model(image)
            teacher_latent, _, _ = load_teacher_latents(
                image=image,
                image_ids=[str(item) for item in batch["image_id"]],
                teacher_latent_paths=[str(item) for item in batch["teacher_latent_path"]],
                teacher_source_types=[str(item) for item in batch["teacher_latent_source_type"]],
                patch_size=patch_size,
                latent_dim=latent_dim,
                device=device,
            )

            per_patch_loss = (student_latent - teacher_latent).pow(2).mean(dim=-1)
            patch_mask_float = patch_mask.to(dtype=per_patch_loss.dtype)
            per_sample_loss = (per_patch_loss * patch_mask_float).sum(dim=1) / patch_mask_float.sum(
                dim=1
            ).clamp_min(1.0)
            sample_losses.extend(float(item) for item in per_sample_loss.detach().cpu().tolist())

            attention_tokens, high_conf_tokens = _pool_patch_priors(
                attention_map=attention_map,
                high_conf_mask=high_conf_mask,
                patch_size=patch_size,
            )
            high_conf_patch_mask = high_conf_tokens >= 0.5
            non_high_conf_patch_mask = ~high_conf_patch_mask
            high_attention_patch_mask, low_attention_patch_mask = _build_attention_band_masks(
                attention_tokens=attention_tokens,
                top_fraction=attention_top_fraction,
            )

            high_conf_loss_total, high_conf_patch_count = _update_region_accumulator(
                per_patch_loss=per_patch_loss,
                patch_selector=patch_mask & high_conf_patch_mask,
                loss_total=high_conf_loss_total,
                patch_count=high_conf_patch_count,
            )
            non_high_conf_loss_total, non_high_conf_patch_count = _update_region_accumulator(
                per_patch_loss=per_patch_loss,
                patch_selector=patch_mask & non_high_conf_patch_mask,
                loss_total=non_high_conf_loss_total,
                patch_count=non_high_conf_patch_count,
            )
            high_attention_loss_total, high_attention_patch_count = _update_region_accumulator(
                per_patch_loss=per_patch_loss,
                patch_selector=patch_mask & high_attention_patch_mask,
                loss_total=high_attention_loss_total,
                patch_count=high_attention_patch_count,
            )
            low_attention_loss_total, low_attention_patch_count = _update_region_accumulator(
                per_patch_loss=per_patch_loss,
                patch_selector=patch_mask & low_attention_patch_mask,
                loss_total=low_attention_loss_total,
                patch_count=low_attention_patch_count,
            )

    if was_training:
        model.train()

    if precomputed_patch_masks is None:
        mask_checksum = compute_mask_sequence_checksum(used_patch_masks)

    sample_loss_tensor = torch.as_tensor(sample_losses, dtype=torch.float64)
    return {
        "eval_mask_policy": eval_mask_policy,
        "eval_seed": int(eval_seed),
        "eval_mask_checksum": mask_checksum,
        "eval_num_samples": int(sample_loss_tensor.numel()),
        "eval_num_batches": int(num_batches),
        "eval_unweighted_masked_mse_mean": float(sample_loss_tensor.mean().item()),
        "eval_unweighted_masked_mse_std": float(
            sample_loss_tensor.std(unbiased=False).item()
        ),
        "eval_high_conf_patch_mse": _safe_mean(high_conf_loss_total, high_conf_patch_count),
        "eval_non_high_conf_patch_mse": _safe_mean(
            non_high_conf_loss_total,
            non_high_conf_patch_count,
        ),
        "eval_high_attention_patch_mse": _safe_mean(
            high_attention_loss_total,
            high_attention_patch_count,
        ),
        "eval_low_attention_patch_mse": _safe_mean(
            low_attention_loss_total,
            low_attention_patch_count,
        ),
    }
