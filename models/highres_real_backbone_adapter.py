from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn

from breast_pretrain.data.transforms.stage1_transform_spec import (
    ImageSize,
    normalize_image_size,
    patch_grid_from_image_size,
)
from breast_pretrain.models.highres_backbone_registry import (
    MAMMO_FM_LIKE_HIGHRES_BACKEND,
    MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND,
    TIMM_HIGHRES_HIERARCHICAL_BACKEND,
    get_highres_backbone_spec,
)
from breast_pretrain.models.highres_encoder_outputs import (
    HighresEncoderOutput,
    HighresFeatureScale,
    build_derived_local_scale,
    build_global_scale,
    build_memory_estimate,
    derive_valid_content_patch_mask,
)
from breast_pretrain.models.visual_encoder_factory import _extract_visual_state_dict
from breast_pretrain.models.mammo_fm_timm_keymap import (
    build_mammo_fm_projection,
    load_translated_mammo_fm_into_timm_efficientnet,
)


@dataclass(frozen=True)
class HighresRealBackboneConfig:
    backend_name: str
    image_size: ImageSize
    patch_size: int = 16
    encoder_output_dim: int = 768
    pretrained_weight_path: str | Path | None = None
    backbone_expected_sha256: str | None = None
    timm_model_name: str | None = None
    freeze_backbone: bool = False


class HighresBackboneIncompatibilityError(ValueError):
    """Raised when a real backbone cannot satisfy the high-res patch-token contract."""


def _verify_expected_sha256(weight_path: Path, expected_sha256: str | None) -> str | None:
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("model.backbone_expected_sha256 must be a lowercase SHA256.")
    actual = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "Formal backbone checkpoint SHA256 mismatch: "
            f"declared={expected}, actual={actual}, path={weight_path}"
        )
    return actual


class MammoFmTimmEfficientNetB5Adapter(nn.Module):
    """Mammo-FM stripped EfficientNet-B5 as a global-feature backbone.

    This adapter intentionally exposes only global/projection features.
    It is NOT a formal production backbone (is_formal_production_backbone=False).
    Use MammoFmSpatialBackbone for the full hierarchical contract.
    """

    def __init__(self, config: HighresRealBackboneConfig) -> None:
        super().__init__()
        spec = get_highres_backbone_spec(config.backend_name)
        if config.pretrained_weight_path is None:
            raise FileNotFoundError(
                f"{spec.backend_name} requires a stripped Mammo-FM image_encoder "
                f"checkpoint; refusing random init."
            )
        weight_path = Path(config.pretrained_weight_path).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"{spec.backend_name} pretrained weight not found: {weight_path}."
            )
        verified_sha256 = _verify_expected_sha256(
            weight_path, config.backbone_expected_sha256
        )
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"{spec.backend_name} requires timm installed in the pre_train environment."
            ) from exc

        self.config = config
        self.image_size = normalize_image_size(config.image_size)
        self.backend_name = str(spec.backend_name)
        self.visual_encoder_backend = self.backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.pretrained_weight_path = str(weight_path)
        self.encoder_output_dim = 2048
        self.projection_output_dim = 0
        self.supports_modality_specific_image_size = True
        self.is_formal_production_backbone = False
        self.is_real_pretrained_backbone = True
        self.is_minimal_or_wrapper_backend = False
        self.spatial_patch_tokens_ready = False
        self.local_branch_training_ready = False

        model_name = str(config.timm_model_name or spec.default_timm_model_name)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        payload = torch.load(weight_path, map_location="cpu")
        result = load_translated_mammo_fm_into_timm_efficientnet(
            mammo_fm_state_dict=payload,
            timm_model=self.backbone,
            strict=True,
            min_loaded_ratio=0.99,
        )
        self.image_projection, projection_report = build_mammo_fm_projection(payload)
        if self.image_projection is not None and bool(projection_report.get("loaded")):
            self.projection_output_dim = int(projection_report["output_dim"])
        self.pretrained_load_report = {
            "weight_path": str(weight_path),
            "model_name": model_name,
            "mapping_report": result.mapping_report,
            "projection_report": projection_report,
            "spatial_patch_tokens_ready": False,
            "local_branch_training_ready": False,
            "is_formal_production_backbone": False,
            "weight_sha256": verified_sha256,
        }
        if bool(config.freeze_backbone):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if self.image_projection is not None:
                for parameter in self.image_projection.parameters():
                    parameter.requires_grad = False

    def forward(
        self, image: torch.Tensor, return_dict: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B, C, H, W], got {tuple(image.shape)}."
            )
        global_image_feature = self.backbone(image)
        if global_image_feature.ndim != 2 or int(global_image_feature.shape[1]) != 2048:
            raise HighresBackboneIncompatibilityError(
                f"{self.backend_name} emitted global feature shape "
                f"{tuple(global_image_feature.shape)}, expected [B, 2048]."
            )
        output: dict[str, torch.Tensor] = {"global_image_feature": global_image_feature}
        if self.image_projection is not None:
            output["projection_image_feature"] = self.image_projection(global_image_feature)
        if return_dict:
            return output
        return output.get("projection_image_feature", global_image_feature)


class TimmHighresHierarchicalAdapter(nn.Module):
    """Real timm ViT/Swin backbone emitting the full HighresEncoderOutput contract.

    This is a real pretrained backbone with local scale derived from primary.
    """

    def __init__(self, config: HighresRealBackboneConfig) -> None:
        super().__init__()
        spec = get_highres_backbone_spec(config.backend_name)
        if spec.is_stub:
            raise ValueError(
                "TimmHighresHierarchicalAdapter cannot build the preflight stub backend."
            )
        if config.pretrained_weight_path is None:
            raise FileNotFoundError(
                f"{spec.backend_name} requires a real local pretrained weight path; "
                f"refusing random init."
            )
        weight_path = Path(config.pretrained_weight_path).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"{spec.backend_name} pretrained weight not found: {weight_path}. "
                "Set production_blocked_by_missing_real_weights instead of falling back to stub."
            )
        verified_sha256 = _verify_expected_sha256(
            weight_path, config.backbone_expected_sha256
        )
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                f"{spec.backend_name} requires timm installed in the pre_train environment; "
                "external reference repositories must remain read-only and are not "
                "runtime dependencies."
            ) from exc

        self.config = config
        self.image_size = normalize_image_size(config.image_size)
        self.patch_size = int(config.patch_size)
        self.expected_patch_grid = patch_grid_from_image_size(self.image_size, self.patch_size)
        self.expected_num_patches = int(
            self.expected_patch_grid[0] * self.expected_patch_grid[1]
        )
        self.encoder_output_dim = int(config.encoder_output_dim)
        self.backend_name = str(spec.backend_name)
        self.visual_encoder_backend = self.backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.pretrained_weight_path = str(weight_path)
        self.supports_modality_specific_image_size = True
        self.is_formal_production_backbone = True
        self.is_real_pretrained_backbone = True
        self.is_minimal_or_wrapper_backend = False

        model_name = str(config.timm_model_name or spec.default_timm_model_name)
        create_kwargs: dict[str, Any] = {
            "pretrained": False, "num_classes": 0, "img_size": self.image_size,
        }
        if self.backend_name == MAMMO_FM_LIKE_HIGHRES_BACKEND:
            create_kwargs["features_only"] = False
        self.backbone = timm.create_model(model_name, **create_kwargs)
        backbone_dim = int(
            getattr(
                self.backbone,
                "embed_dim",
                getattr(self.backbone, "num_features", self.encoder_output_dim),
            )
        )
        self.proj = (
            nn.Linear(backbone_dim, self.encoder_output_dim)
            if backbone_dim != self.encoder_output_dim
            else nn.Identity()
        )
        self.pretrained_load_report = self._load_weights(weight_path)
        if bool(config.freeze_backbone):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _load_weights(self, weight_path: Path) -> dict[str, object]:
        payload = torch.load(weight_path, map_location="cpu")
        state = _extract_visual_state_dict(payload)
        load_result = self.load_state_dict(state, strict=False)
        if not state:
            raise ValueError(
                f"{self.backend_name} checkpoint contains no tensor weights: {weight_path}"
            )
        return {
            "weight_path": str(weight_path),
            "loaded_tensor_count": int(len(state)),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "semantic_compatibility_validation": (
                "load_state_dict_strict_false_plus_forward_grid_check"
            ),
            "weight_sha256": verified_sha256,
        }

    def _features_to_patch_tokens(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int], int]:
        if features.ndim == 3:
            token_count = int(features.shape[1])
            if token_count == self.expected_num_patches + 1:
                return features[:, 1:, :], self.expected_patch_grid, self.patch_size
            if token_count == self.expected_num_patches:
                return features, self.expected_patch_grid, self.patch_size
            raise HighresBackboneIncompatibilityError(
                f"{self.backend_name} emitted {token_count} tokens, expected "
                f"{self.expected_num_patches} patch tokens or "
                f"{self.expected_num_patches + 1} with cls token."
            )
        if features.ndim == 4:
            batch, channels, grid_h, grid_w = (int(v) for v in features.shape)
            patch_grid = (grid_h, grid_w)
            feature_stride_h = self.image_size[0] // grid_h
            feature_stride_w = self.image_size[1] // grid_w
            if feature_stride_h != feature_stride_w:
                raise HighresBackboneIncompatibilityError(
                    f"{self.backend_name} emitted non-square effective stride "
                    f"{feature_stride_h}x{feature_stride_w} for image {self.image_size}."
                )
            if patch_grid != self.expected_patch_grid:
                raise HighresBackboneIncompatibilityError(
                    f"{self.backend_name} emitted feature grid {patch_grid} with "
                    f"stride {feature_stride_h}; expected patch16 grid "
                    f"{self.expected_patch_grid}. Do not force-reshape incompatible features."
                )
            tokens = features.reshape(batch, channels, grid_h * grid_w).transpose(1, 2)
            return tokens, patch_grid, int(feature_stride_h)
        raise HighresBackboneIncompatibilityError(
            f"{self.backend_name} emitted unsupported feature shape "
            f"{tuple(features.shape)}."
        )

    def forward(
        self, image: torch.Tensor, return_dict: bool = False
    ) -> HighresEncoderOutput | dict[str, object]:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B, C, H, W], got {tuple(image.shape)}."
            )
        if tuple(int(v) for v in image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"image size {tuple(image.shape[-2:])} does not match "
                f"configured {self.image_size}."
            )
        batch_size = int(image.shape[0])
        features = self.backbone.forward_features(image)
        patch_tokens, patch_grid, feature_stride = self._features_to_patch_tokens(features)
        patch_tokens = self.proj(patch_tokens)
        if int(patch_tokens.shape[1]) != self.expected_num_patches:
            raise HighresBackboneIncompatibilityError(
                f"{self.backend_name} projected token count "
                f"{int(patch_tokens.shape[1])} does not match expected "
                f"{self.expected_num_patches}."
            )
        global_image_feature = patch_tokens.mean(dim=1)

        primary_scale = HighresFeatureScale(
            name=f"stride{feature_stride}",
            patch_tokens=patch_tokens,
            patch_grid=patch_grid,
            feature_stride=int(feature_stride),
            embedding_dim=self.encoder_output_dim,
            is_derived=False,
            derived_from_primary=False,
        )

        # Derived local scale (R1)
        local_scale = build_derived_local_scale(
            primary_scale,
            local_scale_name="stride8_derived",
            target_stride=8,
        )

        global_scale = build_global_scale(
            global_image_feature,
            scale_name="global_pooled",
            encoder_output_dim=self.encoder_output_dim,
        )

        multi_scale_features = {
            primary_scale.name: primary_scale,
            "stride8_derived": local_scale,
            "global_pooled": global_scale,
        }

        vcm, vcm_source, is_padding_aware = derive_valid_content_patch_mask(
            batch_size=batch_size,
            patch_grid=patch_grid,
        )
        image_size_actual = (int(image.shape[-2]), int(image.shape[-1]))

        output = HighresEncoderOutput(
            patch_tokens=patch_tokens,
            global_image_feature=global_image_feature,
            patch_grid=patch_grid,
            num_patches=self.expected_num_patches,
            encoder_output_dim=self.encoder_output_dim,
            multi_scale_features=multi_scale_features,
            primary_scale_name=primary_scale.name,
            local_scale_name="stride8_derived",
            global_scale_name="global_pooled",
            valid_content_patch_mask=vcm,
            valid_content_patch_mask_source=vcm_source,
            is_padding_aware=is_padding_aware,
            feature_stride=int(feature_stride),
            image_size=image_size_actual,
            input_size=None,
            normalization_profile={"timm_backend": "no_explicit_normalization"},
            projection_dim=self.encoder_output_dim,
            encoder_backend_name=self.backend_name,
            supports_modality_specific_image_size=self.supports_modality_specific_image_size,
            is_formal_production_backbone=self.is_formal_production_backbone,
            backend_name=self.backend_name,
            memory_estimate=build_memory_estimate(
                input_image=image,
                patch_tokens=patch_tokens,
                global_image_feature=global_image_feature,
                multi_scale_features=multi_scale_features,
            ),
        )
        output.validate_contract()

        if return_dict:
            return output.to_dict()
        return output


def build_highres_real_backbone_adapter(config: HighresRealBackboneConfig) -> nn.Module:
    backend = get_highres_backbone_spec(config.backend_name)
    if backend.backend_name == MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND:
        from breast_pretrain.models.mammo_fm_spatial import MammoFmSpatialBackbone

        return MammoFmSpatialBackbone(
            image_size=config.image_size,
            patch_size=config.patch_size,
            encoder_output_dim=config.encoder_output_dim,
            pretrained_weight_path=config.pretrained_weight_path,
            backbone_expected_sha256=config.backbone_expected_sha256,
            freeze_backbone=config.freeze_backbone,
        )
    if backend.backend_name not in {
        TIMM_HIGHRES_HIERARCHICAL_BACKEND,
        MAMMO_FM_LIKE_HIGHRES_BACKEND,
    }:
        raise ValueError(f"Unsupported real high-res backend: {backend.backend_name}")
    return TimmHighresHierarchicalAdapter(config)


__all__ = [
    "HighresBackboneIncompatibilityError",
    "HighresRealBackboneConfig",
    "MammoFmTimmEfficientNetB5Adapter",
    "TimmHighresHierarchicalAdapter",
    "build_highres_real_backbone_adapter",
]
