from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from breast_pretrain.train.masked_latent_smoke import build_patch_gaze_weights


def tensor_to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().numpy().astype(np.float32, copy=False)


def pool_spatial_map(tensor: torch.Tensor, patch_size: int) -> np.ndarray:
    pooled = F.avg_pool2d(
        tensor.unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    )
    return pooled.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def image_tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {array.shape}")
    array = np.transpose(array, (1, 2, 0))
    rgb = np.clip(array, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).round().astype(np.uint8), mode="RGB")


def rgb_tensor_to_grayscale(image_tensor: torch.Tensor) -> np.ndarray:
    array = image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {array.shape}")
    if array.shape[0] == 1:
        return array[0]
    if array.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got {array.shape[0]}")
    return (
        0.2989 * array[0]
        + 0.5870 * array[1]
        + 0.1140 * array[2]
    ).astype(np.float32, copy=False)


def _normalize_kernel_size(kernel_size: int) -> int:
    normalized = max(0, int(kernel_size))
    if normalized <= 1:
        return 0
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def _binary_dilate(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    normalized = _normalize_kernel_size(kernel_size)
    if normalized == 0:
        return mask.astype(bool, copy=True)
    tensor = torch.from_numpy(mask.astype(np.float32, copy=False)).view(1, 1, *mask.shape)
    dilated = F.max_pool2d(
        tensor,
        kernel_size=normalized,
        stride=1,
        padding=normalized // 2,
    )
    return dilated.squeeze(0).squeeze(0).numpy() >= 0.5


def _binary_erode(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    normalized = _normalize_kernel_size(kernel_size)
    if normalized == 0:
        return mask.astype(bool, copy=True)
    inverse = 1.0 - mask.astype(np.float32, copy=False)
    tensor = torch.from_numpy(inverse).view(1, 1, *mask.shape)
    dilated_inverse = F.max_pool2d(
        tensor,
        kernel_size=normalized,
        stride=1,
        padding=normalized // 2,
    )
    eroded = 1.0 - dilated_inverse.squeeze(0).squeeze(0).numpy()
    return eroded >= 0.5


def binary_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, kernel_size=kernel_size), kernel_size=kernel_size)


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    foreground = mask.astype(bool, copy=False)
    height, width = foreground.shape
    if height == 0 or width == 0:
        return foreground.astype(np.float32)

    background = ~foreground
    reachable = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(row: int, col: int) -> None:
        if 0 <= row < height and 0 <= col < width and background[row, col] and not reachable[row, col]:
            reachable[row, col] = True
            queue.append((row, col))

    for row in range(height):
        push(row, 0)
        push(row, width - 1)
    for col in range(width):
        push(0, col)
        push(height - 1, col)

    while queue:
        row, col = queue.popleft()
        push(row - 1, col)
        push(row + 1, col)
        push(row, col - 1)
        push(row, col + 1)

    holes = background & ~reachable
    return np.logical_or(foreground, holes)


def build_breast_tissue_mask(
    image_tensor: torch.Tensor,
    non_black_threshold: float = 8.0 / 255.0,
    morphology_close_kernel: int = 7,
    fill_holes: bool = True,
) -> np.ndarray:
    grayscale = rgb_tensor_to_grayscale(image_tensor)
    mask = grayscale > float(non_black_threshold)
    if _normalize_kernel_size(morphology_close_kernel) > 0:
        mask = binary_close(mask, kernel_size=morphology_close_kernel)
    if fill_holes:
        mask = fill_binary_holes(mask)
    return mask.astype(np.float32, copy=False)


def build_uniform_prior_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.float32, copy=False)
    total = float(binary.sum())
    if total <= 0.0:
        return np.zeros_like(binary, dtype=np.float32)
    return binary / total


def normalize_spatial_prior(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.zeros_like(clipped, dtype=np.float32)
    return clipped / total


def build_shuffled_map_within_mask(
    source_map: np.ndarray,
    mask: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    values = np.asarray(source_map, dtype=np.float32)
    support = np.asarray(mask, dtype=np.float32) > 0.5
    shuffled = np.zeros_like(values, dtype=np.float32)
    masked_values = values[support]
    if masked_values.size == 0:
        return shuffled
    generator = np.random.default_rng(int(random_seed))
    shuffled_values = masked_values[generator.permutation(masked_values.size)]
    shuffled[support] = shuffled_values
    return shuffled


def build_center_prior(
    image_size: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dummy_attention = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    dummy_mask = torch.zeros_like(dummy_attention)
    center_weights, _ = build_patch_gaze_weights(
        attention_map=dummy_attention,
        high_conf_mask=dummy_mask,
        patch_size=patch_size,
        gaze_loss_mode="center_prior",
        gaze_weight_alpha=1.0,
    )
    grid_size = image_size // patch_size
    center_tokens = (center_weights - 1.0).reshape(1, 1, grid_size, grid_size)
    center_map = F.interpolate(
        center_tokens,
        size=(image_size, image_size),
        mode="nearest",
    )
    return (
        normalize_spatial_prior(
            center_tokens.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        ),
        normalize_spatial_prior(
            center_map.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        ),
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64, copy=False)
    right_flat = right.reshape(-1).astype(np.float64, copy=False)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left_flat, right_flat) / (left_norm * right_norm))


def topk_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if flattened.size == 0:
        return np.zeros_like(values, dtype=bool)
    max_value = float(flattened.max())
    if max_value <= 0.0:
        return np.zeros_like(values, dtype=bool)
    k = max(1, min(flattened.size, int(round(flattened.size * ratio))))
    selected = np.argpartition(-flattened, kth=k - 1)[:k]
    mask = np.zeros(flattened.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def intersection_over_union(left_mask: np.ndarray, right_mask: np.ndarray) -> float:
    intersection = float(np.logical_and(left_mask, right_mask).sum())
    union = float(np.logical_or(left_mask, right_mask).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def binary_mask_overlap(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    left = np.asarray(numerator_mask, dtype=np.float32) > 0.5
    right = np.asarray(denominator_mask, dtype=np.float32) > 0.5
    denominator = float(left.sum())
    if denominator <= 0.0:
        return 0.0
    return float(np.logical_and(left, right).sum()) / denominator


def prior_mass_overlap(prior: np.ndarray, binary_mask: np.ndarray) -> float:
    normalized_prior = normalize_spatial_prior(prior)
    if normalized_prior.size == 0:
        return 0.0
    support = (np.asarray(binary_mask, dtype=np.float32) > 0.5).astype(np.float32, copy=False)
    total = float(normalized_prior.sum())
    if total <= 0.0:
        return 0.0
    return float((normalized_prior * support).sum())


def normalized_map(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, None)
    max_value = float(clipped.max()) if clipped.size > 0 else 0.0
    if max_value <= 0.0:
        return np.zeros_like(clipped, dtype=np.float32)
    return clipped / max_value


def soft_overlay(
    base_rgb: np.ndarray,
    soft_values: np.ndarray,
    color: tuple[int, int, int],
    alpha_scale: float = 0.65,
) -> np.ndarray:
    heat = normalized_map(soft_values)
    color_layer = np.zeros_like(base_rgb, dtype=np.float32)
    color_layer[..., 0] = float(color[0]) / 255.0
    color_layer[..., 1] = float(color[1]) / 255.0
    color_layer[..., 2] = float(color[2]) / 255.0
    alpha = np.clip(heat[..., None] * alpha_scale, 0.0, 1.0)
    blended = base_rgb * (1.0 - alpha) + color_layer * alpha
    return np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)


def mask_overlay(
    base_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int],
    alpha_value: float = 0.65,
) -> np.ndarray:
    mask = (binary_mask > 0.5).astype(np.float32, copy=False)[..., None]
    color_layer = np.zeros_like(base_rgb, dtype=np.float32)
    color_layer[..., 0] = float(color[0]) / 255.0
    color_layer[..., 1] = float(color[1]) / 255.0
    color_layer[..., 2] = float(color[2]) / 255.0
    alpha = mask * alpha_value
    blended = base_rgb * (1.0 - alpha) + color_layer * alpha
    return np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)


def save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB").save(path)
