from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalRoiOutput:
    roi_box: torch.Tensor
    roi_valid: torch.Tensor
    fallback_reason: tuple[str, ...]
    local_image: torch.Tensor


def _single_bbox_from_mask(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    positive = torch.nonzero(mask > 0.5, as_tuple=False)
    if positive.numel() == 0:
        return None
    y0 = int(positive[:, 0].min().item())
    y1 = int(positive[:, 0].max().item()) + 1
    x0 = int(positive[:, 1].min().item())
    x1 = int(positive[:, 1].max().item()) + 1
    return y0, x0, y1, x1


def _single_bbox_from_heatmap(heatmap: torch.Tensor) -> tuple[int, int, int, int] | None:
    heatmap = heatmap.clamp_min(0.0)
    max_value = float(heatmap.max().item())
    if max_value <= 0.0:
        return None
    threshold = max_value * 0.5
    return _single_bbox_from_mask((heatmap >= threshold).to(dtype=torch.float32))


def _expand_and_clamp(
    box: tuple[int, int, int, int],
    *,
    margin: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = box
    y0 = max(0, y0 - int(margin))
    x0 = max(0, x0 - int(margin))
    y1 = min(int(height), y1 + int(margin))
    x1 = min(int(width), x1 + int(margin))
    if y1 <= y0:
        y1 = min(int(height), y0 + 1)
    if x1 <= x0:
        x1 = min(int(width), x0 + 1)
    return y0, x0, y1, x1


def build_local_high_conf_roi_batch(
    *,
    image: torch.Tensor,
    high_conf_mask: torch.Tensor,
    heatmap: torch.Tensor,
    margin: int = 32,
    local_input_size: tuple[int, int] = (512, 512),
) -> LocalRoiOutput:
    if image.ndim != 4:
        raise ValueError(f"image must have shape [B, C, H, W], got {tuple(image.shape)}.")
    if high_conf_mask.ndim != 4 or int(high_conf_mask.shape[1]) != 1:
        raise ValueError("high_conf_mask must have shape [B, 1, H, W].")
    if heatmap.ndim != 4 or int(heatmap.shape[1]) != 1:
        raise ValueError("heatmap must have shape [B, 1, H, W].")
    if image.shape[0] != high_conf_mask.shape[0] or image.shape[0] != heatmap.shape[0]:
        raise ValueError("image, high_conf_mask, and heatmap batch sizes must match.")
    if tuple(image.shape[-2:]) != tuple(high_conf_mask.shape[-2:]) or tuple(image.shape[-2:]) != tuple(heatmap.shape[-2:]):
        raise ValueError("image, high_conf_mask, and heatmap spatial sizes must match after Stage 1 transform.")

    batch_size = int(image.shape[0])
    height, width = int(image.shape[-2]), int(image.shape[-1])
    crops: list[torch.Tensor] = []
    boxes: list[list[int]] = []
    valid: list[bool] = []
    reasons: list[str] = []
    for index in range(batch_size):
        mask_box = _single_bbox_from_mask(high_conf_mask[index, 0])
        if mask_box is not None:
            reason = "high_conf_mask_bbox"
            box = mask_box
            is_valid = True
        else:
            heatmap_box = _single_bbox_from_heatmap(heatmap[index, 0])
            if heatmap_box is not None:
                reason = "heatmap_top_mass"
                box = heatmap_box
                is_valid = True
            else:
                reason = "empty_mask_and_heatmap_full_image"
                box = (0, 0, height, width)
                is_valid = False
        y0, x0, y1, x1 = _expand_and_clamp(box, margin=margin, height=height, width=width)
        crop = image[index : index + 1, :, y0:y1, x0:x1]
        crop = F.interpolate(crop, size=local_input_size, mode="bilinear", align_corners=False)
        crops.append(crop[0])
        boxes.append([y0, x0, y1, x1])
        valid.append(is_valid)
        reasons.append(reason)

    return LocalRoiOutput(
        roi_box=torch.tensor(boxes, dtype=torch.long, device=image.device),
        roi_valid=torch.tensor(valid, dtype=torch.bool, device=image.device),
        fallback_reason=tuple(reasons),
        local_image=torch.stack(crops, dim=0),
    )


__all__ = [
    "LocalRoiOutput",
    "build_local_high_conf_roi_batch",
]
