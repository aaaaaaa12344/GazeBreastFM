from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from breast_pretrain.data.transforms.stage1_transform_spec import (
    build_valid_content_patch_mask_from_geometry,
    patch_grid_from_image_size,
)


@dataclass(frozen=True)
class ResolvedPatchGrid:
    num_patches: int
    patch_grid: tuple[int, int]
    per_modality_grids: dict[str, tuple[int, int]]
    modality_list: list[str]
    valid_content_patch_mask: torch.Tensor
    content_overlap_ratio: torch.Tensor | None


def assert_frozen_dataset_entry_v2_geometry(
    *,
    patch_count: int,
    patch_grid: tuple[int, int],
    patch_token_order_version: str,
    valid_content_mask: torch.Tensor,
) -> None:
    """Hard-check frozen E1 geometry without touching Patch embedding logic."""
    expected = {(57, 95): 5415, (32, 32): 1024}
    if expected.get(tuple(patch_grid)) != int(patch_count):
        raise ValueError("Frozen Dataset Entry V2 Patch grid/count is invalid.")
    if patch_token_order_version != "row_major_h_w_v1":
        raise ValueError("Frozen Dataset Entry V2 requires row_major_h_w_v1 token order.")
    if int(valid_content_mask.numel()) != int(patch_count):
        raise ValueError("Frozen Dataset Entry V2 valid-content mask length is invalid.")


def _intersection_area(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> int:
    """Axis-aligned rectangle intersection area."""
    x_overlap = max(0, min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0]))
    y_overlap = max(0, min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1]))
    return x_overlap * y_overlap


def _patch_rect(grid_i: int, grid_j: int, patch_size: int) -> tuple[int, int, int, int]:
    """Pixel-space rectangle for patch at (grid_i, grid_j)."""
    return (
        grid_j * patch_size,
        grid_i * patch_size,
        (grid_j + 1) * patch_size,
        (grid_i + 1) * patch_size,
    )


def build_valid_content_patch_mask(
    transform_metadata: dict[str, Any] | None,
    patch_grid: tuple[int, int],
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build a boolean mask marking patches that contain real image content.

    For aspect_ratio_preserving_resize_pad: excludes patches entirely in the
    padding region. Border patches that partially overlap content are VALID.

    For direct_resize: all patches are valid.

    Returns (valid_mask [grid_h * grid_w], overlap_ratio [grid_h * grid_w] | None).
    """
    if transform_metadata is None:
        return torch.ones(patch_grid[0] * patch_grid[1], dtype=torch.bool), None

    policy = str(transform_metadata.get("transform_policy", "")).strip().lower()
    geometry = (
        transform_metadata.get("image_transform_geometry")
        or transform_metadata.get("geometry")
        or transform_metadata.get("transform_geometry")
    )
    return build_valid_content_patch_mask_from_geometry(
        transform_policy=policy,
        geometry=geometry,
        patch_grid=patch_grid,
        patch_size=patch_size,
    )


def _normalise_projection_records(
    projection_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    raw_records = projection_metadata.get("records")
    if isinstance(raw_records, list):
        for item in raw_records:
            if isinstance(item, dict) and str(item.get("image_id", "")).strip():
                records[str(item["image_id"]).strip()] = item
    for key, value in projection_metadata.items():
        if isinstance(value, dict) and key not in {
            "records",
            "image_size_by_modality",
            "patch_grid_by_modality",
            "transform_policy_by_modality",
        }:
            records.setdefault(str(key), value)
    return records


def _projection_record_for_image(
    projection_metadata: dict[str, Any] | None,
    image_id: str,
) -> dict[str, Any] | None:
    if projection_metadata is None:
        return None
    return _normalise_projection_records(projection_metadata).get(str(image_id).strip())


def _metadata_for_content_mask(
    *,
    projection_metadata: dict[str, Any] | None,
    image_id: str,
    modality: str,
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    modality = modality.strip().lower()
    if projection_metadata is None:
        return None
    if modality != "mammography":
        return {"transform_policy": "direct_resize"}
    record = _projection_record_for_image(projection_metadata, image_id)
    if record is None:
        raise ValueError(
            f"Missing mammography projection metadata for image_id={image_id!r}; "
            "formal Stage 1 launch is blocked."
        )
    policy = str(
        record.get("transform_policy")
        or record.get("policy")
        or (projection_metadata or {}).get("transform_policy_by_modality", {}).get("mammography", "")
    ).strip()
    if not policy:
        policy = "aspect_ratio_preserving_resize_pad"
    geometry = (
        record.get("image_transform_geometry")
        or record.get("geometry")
        or record.get("transform_geometry")
    )
    if geometry is None:
        # Metadata generated by older materializers may store the direct fields.
        geometry = {
            "pad": record.get("pad"),
            "resized_size": record.get("resized_size"),
        }
    if policy == "aspect_ratio_preserving_resize_pad" and (
        geometry is None or geometry.get("pad") is None or geometry.get("resized_size") is None
    ):
        raise ValueError(
            f"Missing mammography geometry for image_id={image_id!r}; "
            "formal Stage 1 launch is blocked."
        )
    return {
        "transform_policy": policy,
        "geometry": geometry,
        "target_size": image_size,
    }


def expand_patch_mask_to_spatial(
    patch_mask: torch.Tensor,
    patch_grid: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """Expand a [B, N] patch-level boolean mask to [B, 1, H, W] spatial mask.

    Each patch is expanded to its full patch_size × patch_size region.
    """
    batch_size, num_patches = patch_mask.shape[:2]
    grid_h, grid_w = patch_grid
    assert num_patches == grid_h * grid_w, (
        f"patch_mask has {num_patches} patches but grid is {grid_h}×{grid_w}"
    )

    patch_mask_2d = patch_mask.reshape(batch_size, grid_h, grid_w).float()
    spatial = patch_mask_2d.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    return spatial.unsqueeze(1)


def resolve_patch_grid_from_batch(
    image: torch.Tensor,
    modalities: list[str],
    patch_size: int,
    image_size_by_modality: dict[str, tuple[int, int]] | None = None,
    transform_metadata: dict[str, Any] | None = None,
    *,
    image_ids: list[str] | None = None,
    projection_metadata: dict[str, Any] | None = None,
    frozen_valid_content_patch_mask: torch.Tensor | None = None,
) -> ResolvedPatchGrid:
    """Resolve the patch grid for a batch of potentially mixed-modality images.

    Validates that the resolved grid is consistent within each modality bucket
    and that all image dimensions are divisible by patch_size.

    If image_size_by_modality is provided, validates image shapes match.

    When a frozen Dataset Entry mask is provided, it is the primary authority
    for valid-content selection. Projection metadata, when present, is used
    only for an exact geometry consistency cross-check.
    """
    batch_size = int(image.shape[0])
    per_modality_grids: dict[str, tuple[int, int]] = {}
    modality_list: list[str] = []

    primary_h, primary_w = int(image.shape[-2]), int(image.shape[-1])
    if primary_h % patch_size != 0 or primary_w % patch_size != 0:
        raise ValueError(
            f"image size ({primary_h},{primary_w}) not divisible by patch_size={patch_size}"
        )
    primary_grid = (primary_h // patch_size, primary_w // patch_size)

    for i in range(batch_size):
        mod = modalities[i].strip().lower()
        modality_list.append(mod)
        h, w = int(image[i].shape[-2]), int(image[i].shape[-1])

        if image_size_by_modality and mod in image_size_by_modality:
            expected = image_size_by_modality[mod]
            if (h, w) != tuple(expected):
                raise ValueError(
                    f"modality {mod} image size ({h},{w}) != configured {expected}"
                )

        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"modality {mod} size ({h},{w}) not divisible by patch_size={patch_size}"
            )

        grid = (h // patch_size, w // patch_size)
        if mod not in per_modality_grids:
            per_modality_grids[mod] = grid
        elif per_modality_grids[mod] != grid:
            raise ValueError(
                f"inconsistent grid for modality {mod}: "
                f"{per_modality_grids[mod]} vs {grid}"
            )

    num_patches = primary_grid[0] * primary_grid[1]

    frozen_mask: torch.Tensor | None = None
    if frozen_valid_content_patch_mask is not None:
        if not isinstance(frozen_valid_content_patch_mask, torch.Tensor):
            raise TypeError("frozen_valid_content_patch_mask must be a torch.Tensor.")
        if frozen_valid_content_patch_mask.ndim != 2:
            raise ValueError(
                "frozen_valid_content_patch_mask must have shape [B, N], got "
                f"{tuple(frozen_valid_content_patch_mask.shape)}"
            )
        expected_shape = (batch_size, num_patches)
        if tuple(frozen_valid_content_patch_mask.shape) != expected_shape:
            raise ValueError(
                "frozen valid-content mask shape does not match runtime patch grid: "
                f"mask={tuple(frozen_valid_content_patch_mask.shape)} expected={expected_shape}"
            )
        if frozen_valid_content_patch_mask.dtype != torch.bool:
            raise TypeError("Frozen Dataset Entry valid-content mask must have bool dtype.")
        if not frozen_valid_content_patch_mask.any(dim=1).all().item():
            raise ValueError("Frozen Dataset Entry valid-content mask cannot be empty for a sample.")
        frozen_mask = frozen_valid_content_patch_mask.to(device=image.device, dtype=torch.bool)

    valid_masks: list[torch.Tensor] = []
    overlap_ratios: list[torch.Tensor] = []
    has_overlap = False
    for index in range(batch_size):
        if frozen_mask is not None:
            valid_mask = frozen_mask[index]
            overlap_ratio = valid_mask.to(dtype=torch.float32)
            # Metadata is optional at runtime, but when present it must agree
            # exactly with the immutable Dataset Entry mask.
            sample_metadata = transform_metadata
            if sample_metadata is None:
                image_id = image_ids[index] if image_ids is not None else ""
                if projection_metadata is not None and modality_list[index] == "mammography":
                    # Metadata is optional; an absent legacy per-image record
                    # simply leaves the frozen Dataset Entry as sole authority.
                    if _projection_record_for_image(projection_metadata, image_id) is not None:
                        sample_metadata = _metadata_for_content_mask(
                            projection_metadata=projection_metadata,
                            image_id=image_id,
                            modality=modality_list[index],
                            image_size=(int(image[index].shape[-2]), int(image[index].shape[-1])),
                        )
                else:
                    sample_metadata = _metadata_for_content_mask(
                        projection_metadata=projection_metadata,
                        image_id=image_id,
                        modality=modality_list[index],
                        image_size=(int(image[index].shape[-2]), int(image[index].shape[-1])),
                    )
            if sample_metadata is not None:
                metadata_mask, _ = build_valid_content_patch_mask(
                    transform_metadata=sample_metadata,
                    patch_grid=primary_grid,
                    patch_size=patch_size,
                )
                if not torch.equal(valid_mask.cpu(), metadata_mask.cpu()):
                    image_id = image_ids[index] if image_ids is not None else index
                    raise ValueError(
                        "Frozen Dataset Entry valid-content mask does not match "
                        f"projection metadata geometry for image_id={image_id!r}."
                    )
        else:
            sample_metadata = transform_metadata
            if sample_metadata is None:
                sample_metadata = _metadata_for_content_mask(
                    projection_metadata=projection_metadata,
                    image_id=(image_ids[index] if image_ids is not None else ""),
                    modality=modality_list[index],
                    image_size=(int(image[index].shape[-2]), int(image[index].shape[-1])),
                )
            valid_mask, overlap_ratio = build_valid_content_patch_mask(
                transform_metadata=sample_metadata,
                patch_grid=primary_grid,
                patch_size=patch_size,
            )
        valid_masks.append(valid_mask)
        if overlap_ratio is None:
            overlap_ratios.append(valid_mask.to(dtype=torch.float32))
        else:
            has_overlap = True
            overlap_ratios.append(overlap_ratio)

    return ResolvedPatchGrid(
        num_patches=num_patches,
        patch_grid=primary_grid,
        per_modality_grids=per_modality_grids,
        modality_list=modality_list,
        valid_content_patch_mask=torch.stack(valid_masks, dim=0),
        content_overlap_ratio=torch.stack(overlap_ratios, dim=0) if has_overlap else None,
    )


def validate_patch_token_grid_consistency(
    patch_tokens: torch.Tensor,
    resolved: ResolvedPatchGrid,
) -> None:
    """Validate that encoder output matches the expected patch grid."""
    actual_n = int(patch_tokens.shape[1])
    expected_n = resolved.num_patches
    if actual_n != expected_n:
        raise ValueError(
            f"encoder emitted {actual_n} patch tokens, expected {expected_n} "
            f"(grid {resolved.patch_grid})"
        )


def build_dynamic_patch_sampling_priors(
    batch_patch_gaze_weight: torch.Tensor | None,
    batch_high_conf_patch_prior: torch.Tensor | None,
    resolved: ResolvedPatchGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build gaze and high-confidence sampling priors at the correct patch grid.

    Primary source for gaze: batch_patch_gaze_weight (already at final grid).
    High-confidence: pre-materialized batch_high_conf_patch_prior.

    Never uses 224×224 masks with patch_size=16 to generate 196-patch priors.
    """
    batch_size = len(resolved.modality_list)
    n = resolved.num_patches

    reference_device = (
        batch_patch_gaze_weight.device
        if batch_patch_gaze_weight is not None
        else batch_high_conf_patch_prior.device
        if batch_high_conf_patch_prior is not None
        else resolved.valid_content_patch_mask.device
    )
    if batch_patch_gaze_weight is not None:
        gaze_scores = batch_patch_gaze_weight.reshape(batch_size, n).to(device=reference_device)
    else:
        gaze_scores = torch.zeros(batch_size, n, device=reference_device)

    if batch_high_conf_patch_prior is not None:
        high_conf = batch_high_conf_patch_prior.reshape(batch_size, n).to(device=reference_device)
    else:
        high_conf = torch.zeros(batch_size, n, device=reference_device)

    content_mask = resolved.valid_content_patch_mask.to(dtype=gaze_scores.dtype, device=gaze_scores.device)
    gaze_scores = gaze_scores * content_mask
    high_conf = high_conf * content_mask

    return gaze_scores, high_conf


__all__ = [
    "ResolvedPatchGrid",
    "build_dynamic_patch_sampling_priors",
    "build_valid_content_patch_mask",
    "expand_patch_mask_to_spatial",
    "resolve_patch_grid_from_batch",
    "validate_patch_token_grid_consistency",
]
