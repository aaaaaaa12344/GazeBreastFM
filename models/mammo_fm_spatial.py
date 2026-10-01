from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from torch import nn

from breast_pretrain.data.transforms.stage1_transform_spec import (
    normalize_image_size,
    patch_grid_from_image_size,
)
from breast_pretrain.models.highres_encoder_outputs import (
    HighresEncoderOutput,
    HighresFeatureScale,
    build_derived_local_scale,
    build_global_scale,
    build_memory_estimate,
    derive_valid_content_patch_mask,
)
from breast_pretrain.models.mammo_fm_timm_keymap import (
    load_translated_mammo_fm_into_timm_efficientnet,
)


class MammoFmSpatialBackbone(nn.Module):
    """Mammo-FM EfficientNet-B5 with hierarchical multi-scale patch-token output.

    Path A (global): final global feature [B,2048] → global_proj → global_image_feature [B,D]
    Path B (spatial): stride-16 intermediate feature [B,C,H/16,W/16] → spatial_proj → patch_tokens [B,N,D]
    Path C (derived local): 2x nearest upsampling of stride-16 → stride8_derived [B,4N,D]

    Contract status: formal_frozen (is_formal_production_backbone=True after grid validation).
    multi_scale_status: derived_local_multiscale (local scale is derived, not native).
    """

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int = 16,
        encoder_output_dim: int = 768,
        pretrained_weight_path: str | Path | None = None,
        backbone_expected_sha256: str | None = None,
        timm_model_name: str = "tf_efficientnet_b5",
        freeze_backbone: bool = False,
        batch_norm_policy: str = "freeze_running_stats",
        train_batch_norm_affine: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "MammoFmSpatialBackbone requires timm. Install with: pip install timm"
            ) from exc

        if pretrained_weight_path is None:
            raise FileNotFoundError(
                "MammoFmSpatialBackbone requires a stripped Mammo-FM checkpoint; "
                "refusing random init."
            )

        weight_path = Path(pretrained_weight_path).expanduser().resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Mammo-FM pretrained weight not found: {weight_path}"
            )
        expected_sha256 = str(backbone_expected_sha256 or "").strip().lower()
        actual_sha256 = None
        if expected_sha256:
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError(
                    "model.backbone_expected_sha256 must be a lowercase SHA256."
                )
            actual_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "Formal backbone checkpoint SHA256 mismatch: "
                    f"declared={expected_sha256}, actual={actual_sha256}, path={weight_path}"
                )

        self.image_size = normalize_image_size(image_size)
        self.patch_size = int(patch_size)
        self.patch_grid = patch_grid_from_image_size(self.image_size, self.patch_size)
        self._num_patches = int(self.patch_grid[0] * self.patch_grid[1])
        self.encoder_output_dim = int(encoder_output_dim)
        self.backend_name = "mammo_fm_timm_efficientnet_b5"
        self.visual_encoder_backend = self.backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.pretrained_weight_path = str(weight_path)
        self.supports_modality_specific_image_size = True
        self.is_formal_production_backbone = True
        self.is_real_pretrained_backbone = True
        self.is_minimal_or_wrapper_backend = False
        self.spatial_patch_tokens_ready = True
        self.multi_scale_features_ready = True
        self.local_branch_training_ready = False
        self._stride16_module_name: str | None = None
        self.mammo_fm_reference_validation = {
            "reference_repo": "Mammo-FM upstream reference",
            "source_files": [
                "src/codebase/train_detector.py",
                "src/codebase/train_classifier.py",
                "src/codebase/breastclip/data/data_utils.py",
                "src/codebase/Detectors/retinanet/detector_model.py",
            ],
            "normalization_mean": 0.3089279,
            "normalization_std": 0.25053555408335154,
            "authoritative_image_size_hw": [1520, 912],
            "detector_source_layer_indexes": [26, 37],
            "required_stage": "effective_stride_16",
        }
        self.register_buffer(
            "_mammo_fm_pixel_mean",
            torch.tensor(0.3089279, dtype=torch.float32).view(1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_mammo_fm_pixel_std",
            torch.tensor(0.25053555408335154, dtype=torch.float32).view(1, 1, 1, 1),
            persistent=False,
        )
        self.last_input_normalization_audit: dict[str, object] = {}

        self.backbone = timm.create_model(
            timm_model_name, pretrained=False, num_classes=0
        )

        self._stride16_dim = self._discover_stride16_dim()
        if self._stride16_dim is None:
            raise RuntimeError(
                "MammoFmSpatialBackbone: cannot discover stride-16 feature stage. "
                "Verify the timm EfficientNet-B5 model provides feature_info or "
                "intermediate features at effective stride 16."
            )

        payload = torch.load(weight_path, map_location="cpu")
        result = load_translated_mammo_fm_into_timm_efficientnet(
            mammo_fm_state_dict=payload,
            timm_model=self.backbone,
            strict=True,
            min_loaded_ratio=0.99,
        )
        self.pretrained_load_report = {
            "weight_path": str(weight_path),
            "weight_sha256": actual_sha256,
            "model_name": timm_model_name,
            "mapping_report": result.mapping_report,
            "stride16_channels": int(self._stride16_dim),
        }

        self.global_proj = nn.Linear(2048, self.encoder_output_dim)
        self.spatial_proj = nn.Linear(self._stride16_dim, self.encoder_output_dim)

        self._bn_policy = batch_norm_policy
        self._train_bn_affine = train_batch_norm_affine

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self._apply_bn_policy()
        self._validate_grid()

    def _discover_stride16_dim(self) -> int | None:
        if hasattr(self.backbone, "feature_info") and self.backbone.feature_info:
            for fi in self.backbone.feature_info:
                try:
                    info = dict(fi) if not isinstance(fi, dict) else fi
                except Exception:
                    continue
                reduction = info.get("reduction", 0)
                if reduction == 16:
                    chs = info.get("num_chs", 0)
                    module_name = info.get("module") or info.get("module_name")
                    if chs > 0 and module_name:
                        if module_name not in dict(self.backbone.named_modules()):
                            raise RuntimeError(
                                f"MammoFmSpatialBackbone: feature_info stride-16 module "
                                f"{module_name!r} not found in model modules."
                            )
                        self._stride16_module_name = str(module_name)
                        return int(chs)

        diagnostics: list[str] = []
        try:
            dummy = torch.randn(1, 3, *self.image_size)
            self.backbone.eval()

            try:
                with torch.no_grad():
                    feats = self.backbone.forward_features(dummy)
                if isinstance(feats, (list, tuple)):
                    for idx, feat in enumerate(feats):
                        if feat.ndim == 4:
                            _, c, gh, gw = feat.shape
                            effective_stride = self.image_size[0] // gh
                            if effective_stride == 16:
                                return int(c)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                diagnostics.append(f"forward_features_list_probe:{type(exc).__name__}:{exc}")

            for stage_idx, stage in enumerate(self.backbone.blocks):
                captured: dict[str, torch.Tensor] = {}

                def _hook(module, inp, out, store=captured):
                    store["out"] = out

                handle = stage.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        self.backbone.forward_features(dummy)
                    if "out" in captured:
                        feat = captured["out"]
                        if feat.ndim == 4:
                            _, c, gh, _ = feat.shape
                            effective_stride = self.image_size[0] // gh
                            if effective_stride == 16:
                                self._stride16_module_name = f"blocks.{stage_idx}"
                                return int(c)
                finally:
                    handle.remove()

            try:
                with torch.no_grad():
                    feats = self.backbone.forward_intermediates(
                        dummy, indices=[i for i in range(-6, 0)],
                        return_prefix=True, stop_early=True,
                    )
                if isinstance(feats, (list, tuple)):
                    for _, feat in feats:
                        if feat.ndim == 4:
                            _, c, gh, _ = feat.shape
                            effective_stride = self.image_size[0] // gh
                            if effective_stride == 16:
                                return int(c)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                diagnostics.append(f"forward_intermediates_probe:{type(exc).__name__}:{exc}")

        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            diagnostics.append(f"block_hook_probe:{type(exc).__name__}:{exc}")

        if diagnostics:
            self._stride16_discovery_diagnostics = diagnostics
        return None

    def _apply_bn_policy(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if self._bn_policy == "freeze_running_stats":
                    m.eval()
                if m.weight is not None:
                    m.weight.requires_grad = bool(self._train_bn_affine)
                if m.bias is not None:
                    m.bias.requires_grad = bool(self._train_bn_affine)

    def _validate_grid(self) -> None:
        test_sizes = [self.image_size]
        if self.image_size[:2] == (912, 1520):
            test_sizes.append((512, 512))

        self.backbone.eval()
        with torch.no_grad():
            for size in test_sizes:
                dummy = torch.randn(1, 3, *size)
                try:
                    _, features = self._forward_global_and_stride16(dummy)
                    _, c, gh, gw = features.shape
                    expected_gh = size[0] // self.patch_size
                    expected_gw = size[1] // self.patch_size
                    if (gh, gw) != (expected_gh, expected_gw):
                        raise RuntimeError(
                            f"MammoFmSpatialBackbone: grid mismatch for input {size}. "
                            f"Got ({gh},{gw}), expected ({expected_gh},{expected_gw})"
                        )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"MammoFmSpatialBackbone: production grid validation failed "
                        f"for input {size}: {exc}"
                    ) from exc

        self.backbone.train()
        self._apply_bn_policy()
        self.spatial_patch_tokens_ready = True
        self.is_formal_production_backbone = True

    def _forward_global_and_stride16(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._stride16_module_name:
            raise RuntimeError(
                "MammoFmSpatialBackbone: stride-16 module was not cached at construction."
            )
        features: dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            features["intermediate"] = output

        modules = dict(self.backbone.named_modules())
        module = modules.get(self._stride16_module_name)
        if module is None:
            raise RuntimeError(
                f"MammoFmSpatialBackbone: cached stride-16 module "
                f"{self._stride16_module_name!r} not found."
            )
        handle = module.register_forward_hook(_hook)
        try:
            global_2048 = self.backbone(image)
        finally:
            handle.remove()

        if "intermediate" not in features:
            raise RuntimeError(
                "MammoFmSpatialBackbone: hook did not capture intermediate features."
            )

        feat = features["intermediate"]
        if feat.ndim != 4:
            raise RuntimeError(
                f"MammoFmSpatialBackbone: expected 4D spatial features, got {feat.ndim}D."
            )

        _, _, gh, gw = feat.shape
        expected_gh = int(image.shape[-2]) // self.patch_size
        expected_gw = int(image.shape[-1]) // self.patch_size
        if (gh, gw) != (expected_gh, expected_gw):
            raise RuntimeError(
                f"MammoFmSpatialBackbone: stride-16 grid mismatch. "
                f"Got ({gh},{gw}), expected ({expected_gh},{expected_gw}) "
                f"for input size {tuple(image.shape[-2:])}. "
                f"Verify the correct EfficientNet stage is hooked."
            )

        return global_2048, feat

    def forward(
        self, image: torch.Tensor, return_dict: bool = False
    ) -> HighresEncoderOutput | dict[str, object]:
        if image.ndim != 4:
            raise ValueError(
                f"image must have shape [B, C, H, W], got {tuple(image.shape)}."
            )

        batch_size = int(image.shape[0])
        raw_image = image
        normalized_image = (
            raw_image - self._mammo_fm_pixel_mean.to(dtype=raw_image.dtype, device=raw_image.device)
        ) / self._mammo_fm_pixel_std.to(dtype=raw_image.dtype, device=raw_image.device)
        self.last_input_normalization_audit = {
            "normalization_name": "mammo_fm_author_repo",
            "normalization_applied": True,
            "normalization_mean": float(self._mammo_fm_pixel_mean.item()),
            "normalization_std": float(self._mammo_fm_pixel_std.item()),
            "input_min": float(raw_image.detach().amin().cpu().item()),
            "input_max": float(raw_image.detach().amax().cpu().item()),
            "input_mean": float(raw_image.detach().mean().cpu().item()),
            "input_std": float(raw_image.detach().std(unbiased=False).cpu().item()),
            "normalized_min": float(normalized_image.detach().amin().cpu().item()),
            "normalized_max": float(normalized_image.detach().amax().cpu().item()),
            "normalized_mean": float(normalized_image.detach().mean().cpu().item()),
            "normalized_std": float(normalized_image.detach().std(unbiased=False).cpu().item()),
            "source": "Mammo-FM repo train_detector.py / data_utils.py",
            "reference_repo": "Mammo-FM upstream reference",
        }

        global_2048, spatial_feat = self._forward_global_and_stride16(normalized_image)
        if global_2048.ndim != 2 or int(global_2048.shape[1]) != 2048:
            raise RuntimeError(
                f"EfficientNet-B5 emitted global shape {tuple(global_2048.shape)}, "
                f"expected [B, 2048]."
            )
        global_image_feature = self.global_proj(global_2048)

        _, _, grid_h, grid_w = spatial_feat.shape
        patch_tokens = spatial_feat.flatten(2).transpose(1, 2)
        patch_tokens = self.spatial_proj(patch_tokens)
        patch_grid = (int(grid_h), int(grid_w))

        stride16_scale = HighresFeatureScale(
            name="stride16",
            patch_tokens=patch_tokens,
            patch_grid=patch_grid,
            feature_stride=16,
            embedding_dim=self.encoder_output_dim,
            is_derived=False,
            derived_from_primary=False,
        )

        # Derived stride-8 local scale (R1: local_scale_name cannot be None)
        stride8_scale = build_derived_local_scale(
            stride16_scale,
            local_scale_name="stride8_derived",
            target_stride=8,
        )

        # Pooled stride-32 (metadata / coarser scale)
        pooled_feat = torch.nn.functional.avg_pool2d(spatial_feat, kernel_size=2, stride=2)
        pooled_grid_h, pooled_grid_w = int(pooled_feat.shape[-2]), int(pooled_feat.shape[-1])
        pooled_tokens = pooled_feat.flatten(2).transpose(1, 2)
        pooled_tokens = self.spatial_proj(pooled_tokens)
        stride32_scale = HighresFeatureScale(
            name="stride32_pooled",
            patch_tokens=pooled_tokens,
            patch_grid=(pooled_grid_h, pooled_grid_w),
            feature_stride=32,
            embedding_dim=self.encoder_output_dim,
            is_derived=True,
            derived_from_primary=True,
        )

        # Global scale (singleton-patch)
        global_scale = build_global_scale(
            global_image_feature,
            scale_name="global_pooled",
            encoder_output_dim=self.encoder_output_dim,
        )

        multi_scale_features = {
            "stride16": stride16_scale,
            "stride8_derived": stride8_scale,
            "stride32_pooled": stride32_scale,
            "global_pooled": global_scale,
        }

        # Valid content mask (R3: never silently assume all True)
        vcm, vcm_source, is_padding_aware = derive_valid_content_patch_mask(
            batch_size=batch_size,
            patch_grid=patch_grid,
            transform_policy="aspect_ratio_preserving_resize_pad",
            geometry=None,  # geometry not available at encoder level; caller should supply
        )

        image_size_actual = (int(image.shape[-2]), int(image.shape[-1]))

        output = HighresEncoderOutput(
            patch_tokens=patch_tokens,
            global_image_feature=global_image_feature,
            patch_grid=patch_grid,
            num_patches=int(patch_grid[0] * patch_grid[1]),
            encoder_output_dim=self.encoder_output_dim,
            multi_scale_features=multi_scale_features,
            primary_scale_name="stride16",
            local_scale_name="stride8_derived",
            global_scale_name="global_pooled",
            valid_content_patch_mask=vcm,
            valid_content_patch_mask_source=vcm_source,
            is_padding_aware=is_padding_aware,
            feature_stride=16,
            image_size=image_size_actual,
            input_size=None,
            normalization_profile=dict(self.last_input_normalization_audit),
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

    @property
    def num_patches(self) -> int:
        return self._num_patches


__all__ = ["MammoFmSpatialBackbone"]
