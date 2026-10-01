from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class HighresFeatureScale:
    """One hierarchical feature scale emitted by a high-res visual encoder."""

    name: str
    patch_tokens: torch.Tensor
    patch_grid: tuple[int, int]
    feature_stride: int
    embedding_dim: int
    is_derived: bool = False
    derived_from_primary: bool = False

    @property
    def num_patches(self) -> int:
        return int(self.patch_grid[0] * self.patch_grid[1])


@dataclass(frozen=True)
class HighresEncoderOutput:
    """Unified high-resolution Stage 1 visual encoder contract (frozen)."""

    # -- primary patch tokens --
    patch_tokens: torch.Tensor
    patch_grid: tuple[int, int]
    num_patches: int
    encoder_output_dim: int

    # -- global feature --
    global_image_feature: torch.Tensor

    # -- multi-scale (dict keyed by scale name) --
    multi_scale_features: dict[str, HighresFeatureScale] = field(default_factory=dict)

    # -- scale semantics --
    primary_scale_name: str = "stride16"
    local_scale_name: str = "stride8_derived"
    global_scale_name: str = "global_pooled"

    # -- content mask --
    valid_content_patch_mask: torch.Tensor | None = None
    valid_content_patch_mask_source: str = "assumed_all_valid"
    is_padding_aware: bool = False

    # -- stride --
    feature_stride: int = 16

    # -- audit --
    image_size: tuple[int, int] | None = None
    input_size: tuple[int, int] | None = None
    normalization_profile: dict[str, object] = field(default_factory=dict)
    projection_dim: int = 0
    encoder_backend_name: str = ""

    # -- status --
    supports_modality_specific_image_size: bool = False
    is_formal_production_backbone: bool = False
    backend_name: str = ""
    memory_estimate: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.projection_dim == 0:
            object.__setattr__(self, "projection_dim", self.encoder_output_dim)
        if not self.encoder_backend_name:
            object.__setattr__(self, "encoder_backend_name", self.backend_name)

    def validate_contract(self) -> None:
        if self.patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must have shape [B, N, D], got {tuple(self.patch_tokens.shape)}."
            )
        if self.global_image_feature.ndim != 2:
            raise ValueError(
                "global_image_feature must have shape [B, D], "
                f"got {tuple(self.global_image_feature.shape)}."
            )
        batch, token_count, dim = (int(v) for v in self.patch_tokens.shape)
        global_batch, global_dim = (int(v) for v in self.global_image_feature.shape)
        if global_batch != batch:
            raise ValueError(f"global batch {global_batch} does not match patch batch {batch}.")
        if token_count != int(self.num_patches):
            raise ValueError(
                f"patch token count {token_count} does not match num_patches {self.num_patches}."
            )
        if int(self.patch_grid[0] * self.patch_grid[1]) != int(self.num_patches):
            raise ValueError(
                f"patch_grid {self.patch_grid} does not match num_patches {self.num_patches}."
            )
        if dim != int(self.encoder_output_dim) or global_dim != int(self.encoder_output_dim):
            raise ValueError(
                "encoder_output_dim mismatch: "
                f"patch dim={dim}, global dim={global_dim}, expected={self.encoder_output_dim}."
            )
        if self.primary_scale_name and self.primary_scale_name not in self.multi_scale_features:
            raise ValueError(
                f"primary_scale_name={self.primary_scale_name!r} not in "
                f"multi_scale_features keys {sorted(self.multi_scale_features)}."
            )
        if self.local_scale_name and self.local_scale_name not in self.multi_scale_features:
            raise ValueError(
                f"local_scale_name={self.local_scale_name!r} not in "
                f"multi_scale_features keys {sorted(self.multi_scale_features)}."
            )
        if self.global_scale_name and self.global_scale_name not in self.multi_scale_features:
            raise ValueError(
                f"global_scale_name={self.global_scale_name!r} not in "
                f"multi_scale_features keys {sorted(self.multi_scale_features)}."
            )
        if self.valid_content_patch_mask is not None:
            v_shape = tuple(int(v) for v in self.valid_content_patch_mask.shape)
            if v_shape != (batch, token_count):
                raise ValueError(
                    f"valid_content_patch_mask shape {v_shape} != "
                    f"expected [{batch}, {token_count}]."
                )
        if not self.local_scale_name:
            raise ValueError("local_scale_name must not be empty.")
        if self.is_formal_production_backbone and not self.encoder_backend_name:
            raise ValueError(
                "is_formal_production_backbone=True requires encoder_backend_name."
            )

    def to_dict(self) -> dict[str, Any]:
        scale_list: list[dict[str, object]] = []
        for scale in self.multi_scale_features.values():
            scale_list.append({
                "name": scale.name,
                "patch_tokens_shape": [int(v) for v in scale.patch_tokens.shape],
                "patch_grid": [int(scale.patch_grid[0]), int(scale.patch_grid[1])],
                "num_patches": int(scale.num_patches),
                "feature_stride": int(scale.feature_stride),
                "embedding_dim": int(scale.embedding_dim),
                "is_derived": bool(scale.is_derived),
                "derived_from_primary": bool(scale.derived_from_primary),
            })
        valid_mask = None
        vcm_shape = None
        if self.valid_content_patch_mask is not None:
            valid_mask = self.valid_content_patch_mask.detach().cpu().clone()
            vcm_shape = [int(v) for v in self.valid_content_patch_mask.shape]
        return {
            "patch_tokens": self.patch_tokens,
            "global_image_feature": self.global_image_feature,
            "global_feature": self.global_image_feature,  # R2: alias for backward compat
            "patch_grid": self.patch_grid,
            "num_patches": self.num_patches,
            "encoder_output_dim": self.encoder_output_dim,
            "multi_scale_features": self.multi_scale_features,
            "multi_scale_features_list": scale_list,
            "primary_scale_name": self.primary_scale_name,
            "local_scale_name": self.local_scale_name,
            "global_scale_name": self.global_scale_name,
            "valid_content_patch_mask": valid_mask,
            "valid_content_patch_mask_source": self.valid_content_patch_mask_source,
            "is_padding_aware": self.is_padding_aware,
            "vcm_shape": vcm_shape,
            "feature_stride": self.feature_stride,
            "image_size": list(self.image_size) if self.image_size else None,
            "input_size": list(self.input_size) if self.input_size else None,
            "normalization_profile": dict(self.normalization_profile),
            "projection_dim": self.projection_dim,
            "embedding_dim": self.encoder_output_dim,
            "is_formal_production_backbone": self.is_formal_production_backbone,
            "backend_name": self.backend_name,
            "encoder_backend_name": self.encoder_backend_name,
            "memory_estimate": dict(self.memory_estimate),
            "supports_modality_specific_image_size": (
                self.supports_modality_specific_image_size
            ),
            "contract_status": "formal_frozen",
            "multiscale_status": (
                "native_full_multiscale"
                if not any(s.is_derived for s in self.multi_scale_features.values())
                else "derived_local_multiscale"
            ),
        }


# ---------------------------------------------------------------------------
# Helper: derived local scale from primary patch tokens (R1)
# ---------------------------------------------------------------------------

def build_derived_local_scale(
    primary_scale: HighresFeatureScale,
    *,
    local_scale_name: str = "stride8_derived",
    target_stride: int = 8,
) -> HighresFeatureScale:
    """Construct a derived stride-8 scale from a primary stride-16 scale via 2x nearest.

    This is used when the backbone cannot natively produce stride-8 features.
    The derived scale is honest about its provenance: is_derived=True,
    derived_from_primary=True.
    """
    if primary_scale.feature_stride < target_stride:
        raise ValueError(
            f"primary stride {primary_scale.feature_stride} is finer than "
            f"target local stride {target_stride}; no derivation needed."
        )
    scale_factor = primary_scale.feature_stride // target_stride
    if scale_factor <= 1:
        raise ValueError(
            f"scale_factor {scale_factor} not > 1; cannot derive finer scale "
            f"from stride {primary_scale.feature_stride}."
        )
    tokens = primary_scale.patch_tokens
    batch, num_patches, dim = (int(v) for v in tokens.shape)
    grid_h, grid_w = int(primary_scale.patch_grid[0]), int(primary_scale.patch_grid[1])
    if num_patches != grid_h * grid_w:
        raise ValueError(
            f"primary scale token count {num_patches} != grid {grid_h}x{grid_w}."
        )
    spatial = tokens.transpose(1, 2).reshape(batch, dim, grid_h, grid_w)
    upsampled = F.interpolate(
        spatial,
        scale_factor=float(scale_factor),
        mode="nearest",
    )
    new_grid_h = grid_h * scale_factor
    new_grid_w = grid_w * scale_factor
    up_tokens = upsampled.reshape(batch, dim, new_grid_h * new_grid_w).transpose(1, 2)
    return HighresFeatureScale(
        name=local_scale_name,
        patch_tokens=up_tokens,
        patch_grid=(new_grid_h, new_grid_w),
        feature_stride=target_stride,
        embedding_dim=dim,
        is_derived=True,
        derived_from_primary=True,
    )


# ---------------------------------------------------------------------------
# Helper: valid content patch mask derivation (R3)
# ---------------------------------------------------------------------------

def derive_valid_content_patch_mask(
    *,
    batch_size: int,
    patch_grid: tuple[int, int],
    transform_policy: str = "",
    geometry: dict[str, object] | None = None,
) -> tuple[torch.Tensor, str, bool]:
    """Derive a [B, N] boolean mask indicating patches with valid content.

    When padding metadata is available (aspect_ratio_preserving_resize_pad),
    projects the pad regions onto the patch grid. Otherwise returns all-True
    with an explicit source marker to prevent silent assumptions (R3).
    """
    grid_h, grid_w = int(patch_grid[0]), int(patch_grid[1])
    num_patches = grid_h * grid_w

    if geometry is not None and transform_policy == "aspect_ratio_preserving_resize_pad":
        pad = geometry.get("pad", {}) if isinstance(geometry, dict) else {}
        pad_top = int(pad.get("top", 0))
        pad_bottom = int(pad.get("bottom", 0))
        pad_left = int(pad.get("left", 0))
        pad_right = int(pad.get("right", 0))
        resized = geometry.get("resized_size", [])
        resized_h = int(resized[0]) if resized and len(resized) >= 1 else grid_h
        resized_w = int(resized[1]) if resized and len(resized) >= 2 else grid_w
        # project padding to patch grid
        pad_top_patches = pad_top // 16 if pad_top > 0 else 0
        pad_bottom_patches = pad_bottom // 16 if pad_bottom > 0 else 0
        pad_left_patches = pad_left // 16 if pad_left > 0 else 0
        pad_right_patches = pad_right // 16 if pad_right > 0 else 0
        total_pad_h = pad_top_patches + pad_bottom_patches
        total_pad_w = pad_left_patches + pad_right_patches
        if total_pad_h >= grid_h or total_pad_w >= grid_w:
            mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
        elif total_pad_h > 0 or total_pad_w > 0:
            mask_2d = torch.ones(grid_h, grid_w, dtype=torch.bool)
            if pad_top_patches > 0:
                mask_2d[:pad_top_patches, :] = False
            if pad_bottom_patches > 0:
                mask_2d[grid_h - pad_bottom_patches:, :] = False
            if pad_left_patches > 0:
                mask_2d[:, :pad_left_patches] = False
            if pad_right_patches > 0:
                mask_2d[:, grid_w - pad_right_patches:] = False
            mask = mask_2d.reshape(1, num_patches).expand(batch_size, -1).clone()
        else:
            mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
        source = "projected_from_padding"
        is_padding_aware = True
    else:
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
        if transform_policy == "aspect_ratio_preserving_resize_pad":
            source = "assumed_all_valid_missing_geometry_padding_expected"
        else:
            source = "assumed_all_valid"
        is_padding_aware = False

    return mask, source, is_padding_aware


# ---------------------------------------------------------------------------
# Helper: global scale builder
# ---------------------------------------------------------------------------

def build_global_scale(
    global_image_feature: torch.Tensor,
    *,
    scale_name: str = "global_pooled",
    encoder_output_dim: int | None = None,
) -> HighresFeatureScale:
    """Wrap global vector as a singleton-patch scale for multi_scale_features."""
    if global_image_feature.ndim != 2:
        raise ValueError(
            f"global_image_feature must have shape [B, D], "
            f"got {tuple(global_image_feature.shape)}."
        )
    dim = int(global_image_feature.shape[1])
    if encoder_output_dim is not None and dim != int(encoder_output_dim):
        raise ValueError(
            f"global dim {dim} != encoder_output_dim {encoder_output_dim}."
        )
    return HighresFeatureScale(
        name=scale_name,
        patch_tokens=global_image_feature.unsqueeze(1),
        patch_grid=(1, 1),
        feature_stride=0,
        embedding_dim=dim,
        is_derived=False,
        derived_from_primary=False,
    )


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def tensor_memory_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def build_memory_estimate(
    *,
    input_image: torch.Tensor,
    patch_tokens: torch.Tensor,
    global_image_feature: torch.Tensor,
    multi_scale_features: dict[str, HighresFeatureScale] | None = None,
) -> dict[str, int]:
    input_bytes = tensor_memory_bytes(input_image)
    patch_token_bytes = tensor_memory_bytes(patch_tokens)
    global_feature_bytes = tensor_memory_bytes(global_image_feature)
    multi_scale_bytes = 0
    if multi_scale_features:
        multi_scale_bytes = sum(
            tensor_memory_bytes(scale.patch_tokens)
            for scale in multi_scale_features.values()
        )
    total_bytes = input_bytes + patch_token_bytes + global_feature_bytes + multi_scale_bytes
    return {
        "input_image_bytes": input_bytes,
        "patch_tokens_bytes": patch_token_bytes,
        "global_image_feature_bytes": global_feature_bytes,
        "multi_scale_feature_bytes": int(multi_scale_bytes),
        "estimated_activation_bytes": int(total_bytes),
        "estimated_activation_mib": int(round(total_bytes / (1024 * 1024))),
    }


__all__ = [
    "HighresEncoderOutput",
    "HighresFeatureScale",
    "build_derived_local_scale",
    "build_global_scale",
    "build_memory_estimate",
    "derive_valid_content_patch_mask",
    "tensor_memory_bytes",
]
