from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from breast_pretrain.models import MinimalPatchStudentEncoder
from breast_pretrain.train.masked_latent_smoke import generate_patch_mask


def set_global_seed(seed: int, deterministic_ablation: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic_ablation, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_ablation
        torch.backends.cudnn.benchmark = not deterministic_ablation


def clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def compute_state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()[:16]


def compute_mask_sequence_checksum(patch_masks: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for index, patch_mask in enumerate(patch_masks):
        normalized = patch_mask.detach().cpu().to(dtype=torch.uint8).contiguous()
        digest.update(str(index).encode("utf-8"))
        digest.update(str(tuple(normalized.shape)).encode("utf-8"))
        digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()[:16]


def build_initial_model_state(
    image_size: int,
    patch_size: int,
    latent_dim: int,
    seed: int,
    deterministic_ablation: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    set_global_seed(seed=seed, deterministic_ablation=deterministic_ablation)
    model = MinimalPatchStudentEncoder(
        image_size=image_size,
        patch_size=patch_size,
        latent_dim=latent_dim,
    ).cpu()
    initial_state = clone_state_dict_to_cpu(model.state_dict())
    return initial_state, compute_state_dict_checksum(initial_state)


def build_patch_mask_sequence(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    max_steps: int,
    seed: int,
) -> tuple[list[torch.Tensor], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    patch_masks = [
        generate_patch_mask(
            batch_size=batch_size,
            num_patches=num_patches,
            mask_ratio=mask_ratio,
            device=torch.device("cpu"),
            generator=generator,
        ).cpu()
        for _ in range(max_steps)
    ]
    return patch_masks, compute_mask_sequence_checksum(patch_masks)
