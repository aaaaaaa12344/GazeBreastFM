from __future__ import annotations

import numpy as np
import torch

from breast_pretrain.audit.tissue_prior import (
    build_breast_tissue_mask,
    build_shuffled_map_within_mask,
    build_uniform_prior_from_mask,
    normalized_map,
    pool_spatial_map,
    tensor_to_numpy_2d,
)
from breast_pretrain.train.masked_latent_smoke import (
    build_patch_gaze_weights,
    validate_gaze_loss_mode,
)


TISSUE_AWARE_GAZE_MODES = {
    "breast_tissue_prior",
    "shuffled_gaze_within_breast",
}


def _to_patch_prior_tensor(
    prior_map: np.ndarray,
    patch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pooled_prior = pool_spatial_map(
        torch.from_numpy(prior_map.astype(np.float32, copy=False)).unsqueeze(0),
        patch_size=patch_size,
    )
    scaled_prior = normalized_map(pooled_prior)
    return torch.from_numpy(scaled_prior.reshape(-1)).to(device=device, dtype=dtype)


def _build_tissue_pixel_prior(
    image_tensor: torch.Tensor,
    attention_tensor: torch.Tensor,
    mode: str,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    tissue_mask = build_breast_tissue_mask(image_tensor.detach().cpu())
    if mode == "breast_tissue_prior":
        pixel_prior = build_uniform_prior_from_mask(tissue_mask)
    else:
        pixel_prior = build_shuffled_map_within_mask(
            source_map=tensor_to_numpy_2d(attention_tensor),
            mask=tissue_mask,
            random_seed=random_seed,
        )
    return tissue_mask, pixel_prior


def build_stage1_patch_gaze_weights(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    high_conf_mask: torch.Tensor,
    patch_size: int,
    gaze_loss_mode: str = "soft_attention_plus_high_conf",
    gaze_weight_alpha: float = 1.0,
    random_seed: int = 42,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalized_mode = validate_gaze_loss_mode(gaze_loss_mode)
    if normalized_mode not in TISSUE_AWARE_GAZE_MODES:
        return build_patch_gaze_weights(
            attention_map=attention_map,
            high_conf_mask=high_conf_mask,
            patch_size=patch_size,
            gaze_loss_mode=normalized_mode,
            gaze_weight_alpha=gaze_weight_alpha,
            random_seed=random_seed,
        )

    if image.ndim != 4:
        raise ValueError(f"image must have shape [batch, channels, height, width], got {tuple(image.shape)}")
    if attention_map.ndim != 4 or high_conf_mask.ndim != 4:
        raise ValueError("attention_map and high_conf_mask must have shape [batch, 1, height, width].")
    if image.shape[0] != attention_map.shape[0] or image.shape[0] != high_conf_mask.shape[0]:
        raise ValueError(
            "image, attention_map, and high_conf_mask must have the same batch dimension, got "
            f"{tuple(image.shape)}, {tuple(attention_map.shape)}, and {tuple(high_conf_mask.shape)}."
        )

    alpha = float(gaze_weight_alpha)
    batch_priors: list[torch.Tensor] = []
    tissue_mask_coverages: list[float] = []
    tissue_patch_support_counts: list[float] = []
    pixel_prior_sums: list[float] = []

    for index in range(int(image.shape[0])):
        tissue_mask, pixel_prior = _build_tissue_pixel_prior(
            image_tensor=image[index],
            attention_tensor=attention_map[index],
            mode=normalized_mode,
            random_seed=int(random_seed) + index,
        )
        patch_prior = _to_patch_prior_tensor(
            prior_map=pixel_prior,
            patch_size=patch_size,
            device=attention_map.device,
            dtype=attention_map.dtype,
        )
        batch_priors.append(patch_prior)
        tissue_mask_coverages.append(float(np.asarray(tissue_mask, dtype=np.float32).mean()))
        tissue_patch_support_counts.append(float((patch_prior > 0).to(dtype=torch.float32).sum().item()))
        pixel_prior_sums.append(float(np.asarray(pixel_prior, dtype=np.float32).sum()))

    prior_tokens = torch.stack(batch_priors, dim=0)
    weights = 1.0 + alpha * prior_tokens
    pooled_attention = torch.nn.functional.avg_pool2d(
        attention_map,
        kernel_size=patch_size,
        stride=patch_size,
    )
    pooled_mask = torch.nn.functional.avg_pool2d(
        high_conf_mask,
        kernel_size=patch_size,
        stride=patch_size,
    )

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
        "patch_prior_min": float(prior_tokens.min().item()),
        "patch_prior_max": float(prior_tokens.max().item()),
        "patch_prior_mean": float(prior_tokens.mean().item()),
        "tissue_mask_coverage_mean": float(np.mean(tissue_mask_coverages)),
        "tissue_patch_support_count_mean": float(np.mean(tissue_patch_support_counts)),
        "pixel_prior_sum_mean": float(np.mean(pixel_prior_sums)),
        "pooled_attention_mean": float(pooled_attention.mean().item()),
        "pooled_high_conf_mean": float(pooled_mask.mean().item()),
    }
    return weights, summary
