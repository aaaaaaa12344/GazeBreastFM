from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


ImageSize = int | tuple[int, int] | list[int]

POLICY_DIRECT_RESIZE = "direct_resize"
POLICY_ASPECT_RATIO_PAD = "aspect_ratio_preserving_resize_pad"


@dataclass(frozen=True)
class Stage1TransformSpec:
    modality: str
    image_size: tuple[int, int]
    policy: str = POLICY_DIRECT_RESIZE
    patch_size: int = 16
    interpolation: str = "bilinear"
    mask_interpolation: str = "nearest"
    pad_value: float = 0.0

    @property
    def patch_grid(self) -> tuple[int, int]:
        return patch_grid_from_image_size(self.image_size, self.patch_size)

    def to_metadata(self) -> dict[str, object]:
        grid_h, grid_w = self.patch_grid
        return {
            "modality": self.modality,
            "stage1_image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "patch_size": int(self.patch_size),
            "patch_grid_h": int(grid_h),
            "patch_grid_w": int(grid_w),
            "transform_policy": self.policy,
            "interpolation": self.interpolation,
            "mask_interpolation": self.mask_interpolation,
            "pad_value": float(self.pad_value),
        }


def stable_transform_checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def stage1_transform_spec_checksum(spec: Stage1TransformSpec) -> str:
    return stable_transform_checksum(spec.to_metadata())


def stage1_transform_geometry_checksum(
    spec: Stage1TransformSpec,
    geometry: dict[str, object],
) -> str:
    return stable_transform_checksum(
        {
            "transform_spec": spec.to_metadata(),
            "geometry": geometry,
        }
    )


def normalize_image_size(image_size: ImageSize) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = int(image_size)
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height = int(image_size[0])
        width = int(image_size[1])
    else:
        raise ValueError(f"image_size must be an int or [height, width], got {image_size!r}.")
    if height <= 0 or width <= 0:
        raise ValueError("image_size height and width must be positive.")
    return height, width


def patch_grid_from_image_size(image_size: ImageSize, patch_size: int) -> tuple[int, int]:
    height, width = normalize_image_size(image_size)
    patch = int(patch_size)
    if patch <= 0:
        raise ValueError("patch_size must be positive.")
    if height % patch != 0 or width % patch != 0:
        raise ValueError(
            f"image_size {height}x{width} must be divisible by patch_size={patch}."
        )
    return height // patch, width // patch


def patch_count_from_image_size(image_size: ImageSize, patch_size: int) -> int:
    grid_h, grid_w = patch_grid_from_image_size(image_size, patch_size)
    return int(grid_h * grid_w)


def build_stage1_transform_spec(
    *,
    modality: str,
    image_size: ImageSize | None = None,
    patch_size: int = 16,
    policy: str | None = None,
) -> Stage1TransformSpec:
    normalized_modality = str(modality).strip().lower()
    if normalized_modality in {"mammo", "mammography"}:
        resolved_size = normalize_image_size(image_size or (1536, 1024))
        resolved_policy = policy or POLICY_ASPECT_RATIO_PAD
        normalized_modality = "mammography"
    elif normalized_modality == "mri":
        resolved_size = normalize_image_size(image_size or 512)
        resolved_policy = policy or POLICY_DIRECT_RESIZE
    elif normalized_modality in {"us", "ultrasound"}:
        resolved_size = normalize_image_size(image_size or 512)
        resolved_policy = policy or POLICY_DIRECT_RESIZE
        normalized_modality = "ultrasound"
    else:
        resolved_size = normalize_image_size(image_size or 224)
        resolved_policy = policy or POLICY_DIRECT_RESIZE
    return Stage1TransformSpec(
        modality=normalized_modality,
        image_size=resolved_size,
        policy=resolved_policy,
        patch_size=int(patch_size),
    )


def _resize_tensor(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, size=size, **kwargs)


TARGET_ALIGNMENT_KEYS = (
    "modality",
    "transform_policy",
    "target_image_size",
    "resized_size",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
    "patch_size",
    "patch_grid",
)


def build_target_alignment_payload(
    spec: Stage1TransformSpec,
    geometry: dict[str, object],
) -> dict[str, object]:
    grid_h, grid_w = spec.patch_grid
    pad = geometry["pad"]
    return {
        "modality": spec.modality,
        "transform_policy": spec.policy,
        "target_image_size": [int(spec.image_size[0]), int(spec.image_size[1])],
        "resized_size": [int(geometry["resized_size"][0]), int(geometry["resized_size"][1])],
        "pad_top": int(pad["top"]),
        "pad_bottom": int(pad["bottom"]),
        "pad_left": int(pad["left"]),
        "pad_right": int(pad["right"]),
        "patch_size": int(spec.patch_size),
        "patch_grid": [int(grid_h), int(grid_w)],
    }


def target_alignment_checksum(
    spec: Stage1TransformSpec,
    geometry: dict[str, object],
) -> str:
    return stable_transform_checksum(build_target_alignment_payload(spec, geometry))


def compute_stage1_transform_geometry(
    spec: Stage1TransformSpec,
    source_size: ImageSize,
) -> dict[str, object]:
    """Return the canonical Stage 1 image-content geometry for ``source_size``.

    This intentionally takes the physical image size separately from any
    sidecar raster size.  Gaze sidecars may subsequently be resampled with the
    returned geometry, but must never determine it.
    """
    source_h, source_w = normalize_image_size(source_size)
    target_h, target_w = spec.image_size
    if spec.policy == POLICY_ASPECT_RATIO_PAD:
        scale = min(target_h / float(source_h), target_w / float(source_w))
        resized_h = max(1, min(target_h, int(round(source_h * scale))))
        resized_w = max(1, min(target_w, int(round(source_w * scale))))
        pad_top = (target_h - resized_h) // 2
        pad_bottom = target_h - resized_h - pad_top
        pad_left = (target_w - resized_w) // 2
        pad_right = target_w - resized_w - pad_left
    elif spec.policy == POLICY_DIRECT_RESIZE:
        scale = None
        resized_h, resized_w = target_h, target_w
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        raise ValueError(f"Unsupported Stage 1 transform policy: {spec.policy!r}.")
    return {
        "source_size": [source_h, source_w],
        "resized_size": [resized_h, resized_w],
        "pad": {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right,
        },
        "scale": scale,
    }


def apply_stage1_spatial_transform_with_geometry(
    tensor: torch.Tensor,
    spec: Stage1TransformSpec,
    geometry: dict[str, object],
    *,
    is_mask: bool = False,
) -> torch.Tensor:
    """Transform a tensor using already-resolved image-content geometry.

    ``tensor`` may be a gaze raster whose source resolution differs from the
    physical image.  It is resized into the image content rectangle before the
    canonical padding is applied.
    """
    if tensor.ndim != 4:
        raise ValueError(f"tensor must have shape [B, C, H, W], got {tuple(tensor.shape)}.")
    resized_size = geometry.get("resized_size")
    pad = geometry.get("pad")
    if not isinstance(resized_size, (tuple, list)) or len(resized_size) != 2:
        raise ValueError("geometry.resized_size must be [height, width].")
    if not isinstance(pad, dict):
        raise ValueError("geometry.pad must be a mapping.")
    resized_h, resized_w = int(resized_size[0]), int(resized_size[1])
    pad_top = int(pad.get("top", 0))
    pad_bottom = int(pad.get("bottom", 0))
    pad_left = int(pad.get("left", 0))
    pad_right = int(pad.get("right", 0))
    target_h, target_w = spec.image_size
    if (resized_h + pad_top + pad_bottom, resized_w + pad_left + pad_right) != (
        target_h,
        target_w,
    ):
        raise ValueError(
            "geometry does not resolve to the Stage 1 target size: "
            f"geometry={(resized_h + pad_top + pad_bottom, resized_w + pad_left + pad_right)} "
            f"target={spec.image_size}."
        )
    mode = spec.mask_interpolation if is_mask else spec.interpolation
    resized = _resize_tensor(tensor, size=(resized_h, resized_w), mode=mode)
    return F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=float(spec.pad_value),
    )


def build_valid_content_patch_mask_from_geometry(
    *,
    transform_policy: str,
    geometry: dict[str, object] | None,
    patch_grid: tuple[int, int],
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build the canonical positive-overlap valid-content patch mask.

    A patch is valid when it has strictly positive area overlap with image
    content.  This preserves partially overlapping border patches while
    excluding patches that lie fully inside padding.
    """
    grid_h, grid_w = (int(patch_grid[0]), int(patch_grid[1]))
    patch = int(patch_size)
    if grid_h <= 0 or grid_w <= 0 or patch <= 0:
        raise ValueError("patch_grid and patch_size must be positive.")
    count = grid_h * grid_w
    if str(transform_policy).strip().lower() != POLICY_ASPECT_RATIO_PAD:
        return torch.ones(count, dtype=torch.bool), None
    if geometry is None:
        raise ValueError("aspect_ratio_preserving_resize_pad requires transform geometry.")
    pad = geometry.get("pad")
    resized = geometry.get("resized_size")
    if not isinstance(pad, dict) or not isinstance(resized, (tuple, list)) or len(resized) != 2:
        raise ValueError(
            "aspect_ratio_preserving_resize_pad requires geometry.pad and geometry.resized_size."
        )
    top = int(pad.get("top", 0))
    left = int(pad.get("left", 0))
    content_bottom = top + int(resized[0])
    content_right = left + int(resized[1])
    patch_area = patch * patch
    mask = torch.zeros(count, dtype=torch.bool)
    overlap = torch.zeros(count, dtype=torch.float32)
    for grid_i in range(grid_h):
        for grid_j in range(grid_w):
            patch_top, patch_left = grid_i * patch, grid_j * patch
            patch_bottom, patch_right = patch_top + patch, patch_left + patch
            height = max(0, min(patch_bottom, content_bottom) - max(patch_top, top))
            width = max(0, min(patch_right, content_right) - max(patch_left, left))
            area = height * width
            index = grid_i * grid_w + grid_j
            if area > 0:
                mask[index] = True
                overlap[index] = float(area) / float(patch_area)
    return mask, overlap


def apply_stage1_spatial_transform(
    tensor: torch.Tensor,
    spec: Stage1TransformSpec,
    *,
    is_mask: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    if tensor.ndim != 4:
        raise ValueError(f"tensor must have shape [B, C, H, W], got {tuple(tensor.shape)}.")
    source_h, source_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    geometry = compute_stage1_transform_geometry(spec, (source_h, source_w))
    transformed = apply_stage1_spatial_transform_with_geometry(
        tensor,
        spec,
        geometry,
        is_mask=is_mask,
    )
    return transformed, geometry


__all__ = [
    "ImageSize",
    "POLICY_ASPECT_RATIO_PAD",
    "POLICY_DIRECT_RESIZE",
    "Stage1TransformSpec",
    "TARGET_ALIGNMENT_KEYS",
    "apply_stage1_spatial_transform",
    "apply_stage1_spatial_transform_with_geometry",
    "build_stage1_transform_spec",
    "build_target_alignment_payload",
    "build_valid_content_patch_mask_from_geometry",
    "compute_stage1_transform_geometry",
    "normalize_image_size",
    "patch_count_from_image_size",
    "patch_grid_from_image_size",
    "stable_transform_checksum",
    "stage1_transform_geometry_checksum",
    "stage1_transform_spec_checksum",
    "target_alignment_checksum",
]
