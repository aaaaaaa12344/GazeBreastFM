from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from breast_pretrain.teachers.base import (
    TEACHER_SOURCE_REAL_CLIP_IMAGE,
    TeacherLatentBatch,
)
from breast_pretrain.teachers.token_utils import infer_patch_grid


def _normalize_crop_size(raw_value: object) -> int:
    getter = getattr(raw_value, "get", None)
    if callable(getter):
        height = getter("height")
        if height is not None:
            return int(height)
        shortest_edge = getter("shortest_edge")
        if shortest_edge is not None:
            return int(shortest_edge)
    if isinstance(raw_value, (tuple, list)) and raw_value:
        return int(raw_value[0])
    return int(raw_value)


class RealClipImageTeacher:
    source_type = TEACHER_SOURCE_REAL_CLIP_IMAGE

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModel
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ImportError(
                "transformers is required for real_clip_image_teacher."
            ) from exc

        self.teacher_model_name = str(model_name).strip()
        if not self.teacher_model_name:
            raise ValueError("model_name is required for real_clip_image_teacher.")

        self.device = device
        self.model = CLIPVisionModel.from_pretrained(
            self.teacher_model_name,
            local_files_only=bool(local_files_only),
        ).to(device)
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.teacher_model_name,
            local_files_only=bool(local_files_only),
        )
        self.input_resolution = _normalize_crop_size(
            self.image_processor.crop_size or self.image_processor.size
        )
        self.image_mean = torch.tensor(
            self.image_processor.image_mean,
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            self.image_processor.image_std,
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)

    def _prepare_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [batch, channels, height, width], got {tuple(image.shape)}"
            )
        pixel_values = image.to(device=self.device, dtype=torch.float32)
        if pixel_values.shape[-2:] != (self.input_resolution, self.input_resolution):
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            )
        return (pixel_values - self.image_mean) / self.image_std

    def build_latents(self, image: torch.Tensor) -> TeacherLatentBatch:
        pixel_values = self._prepare_pixel_values(image)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state
        if hidden_state.shape[1] <= 1:
            raise ValueError(
                f"CLIP hidden_state does not contain patch tokens: {tuple(hidden_state.shape)}"
            )

        patch_tokens = hidden_state[:, 1:, :].detach().to(dtype=torch.float32)
        raw_patch_grid = infer_patch_grid(int(patch_tokens.shape[1]))
        return TeacherLatentBatch(
            tokens=patch_tokens,
            teacher_model_name=self.teacher_model_name,
            source_type=self.source_type,
            raw_patch_grid=raw_patch_grid,
        )
