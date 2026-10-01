from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from breast_pretrain.data.transforms.stage1_transform_spec import (
    ImageSize,
    normalize_image_size,
    patch_grid_from_image_size,
)
from breast_pretrain.models.minimal_student_encoder import MinimalPatchStudentEncoder
from breast_pretrain.models.highres_backbone_registry import (
    MAMMO_FM_LIKE_HIGHRES_BACKEND,
    MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND,
    TIMM_HIGHRES_HIERARCHICAL_BACKEND,
    real_highres_backends,
)


MINIMAL_PATCH_ENCODER = "minimal_patch_encoder"
LOCAL_TORCH_CHECKPOINT = "local_torch_checkpoint"
PRETRAINED_VISUAL_ENCODER = "pretrained_visual_encoder"
GENERIC_TORCH_CHECKPOINT = "generic_torch_checkpoint"

# -- wrapper / smoke-only backends (all route through LocalTorchCheckpointVisualEncoder) --
WRAPPER_BACKEND_NAMES = {
    PRETRAINED_VISUAL_ENCODER,
    LOCAL_TORCH_CHECKPOINT,
    GENERIC_TORCH_CHECKPOINT,
}

# -- real pretrained backends (each gets its own builder, fail-fast on missing deps) --
REAL_BACKEND_NAMES = {
    "generic_timm_vit",
    "hf_clip_vit",
    "clip_vit_b16",
    "biomedclip",
    "medsiglip",
    "mammo_clip",
    "mammo_fm",
    "rad_dino",
}
HIGHRES_REAL_BACKEND_NAMES = set(real_highres_backends())

PRETRAINED_BACKEND_NAMES = WRAPPER_BACKEND_NAMES | REAL_BACKEND_NAMES | HIGHRES_REAL_BACKEND_NAMES

_UNIFIED_OUTPUT_DOC = """
Unified Stage 1 encoder output contract:
  patch_tokens          [B, N, D]
  global_image_feature  [B, D]
  num_patches           int (property)
  patch_grid            [H, W] (property)
  encoder_output_dim    int
"""


class VisualEncoderConfig(Protocol):
    patch_size: int
    latent_dim: int
    vision_encoder_name: str
    pretrained_model_path: str | None
    pretrained_weight_path: str | None
    backbone_expected_sha256: str | None
    allow_missing_pretrained_fallback: bool
    freeze_backbone: bool
    output_patch_dim: int | None


def _normalise_encoder_name(raw_name: str | None) -> str:
    return str(raw_name or MINIMAL_PATCH_ENCODER).strip().lower() or MINIMAL_PATCH_ENCODER


def _resolve_output_patch_dim(model_config: VisualEncoderConfig) -> int:
    output_patch_dim = model_config.output_patch_dim
    if output_patch_dim is None:
        output_patch_dim = model_config.latent_dim
    output_patch_dim = int(output_patch_dim)
    if output_patch_dim <= 0:
        raise ValueError("model.output_patch_dim must be positive.")
    return output_patch_dim


def _candidate_weight_path(model_config: VisualEncoderConfig) -> str | None:
    return model_config.pretrained_weight_path or model_config.pretrained_model_path


def _patch_grid(image_size: ImageSize, patch_size: int) -> list[int]:
    grid_h, grid_w = patch_grid_from_image_size(image_size, patch_size)
    return [int(grid_h), int(grid_w)]


def _resolve_hf_clip_local_dir(model_dir: Path) -> tuple[str, dict[str, object]]:
    required_files = ("config.json", "preprocessor_config.json")
    missing_files = [filename for filename in required_files if not (model_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "hf_clip_vit local model directory is incomplete: "
            f"{model_dir}. Missing required file(s): {', '.join(missing_files)}."
        )

    safetensors_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.is_file():
        return str(model_dir), {"local_files_only": True, "use_safetensors": True}
    if pytorch_bin_path.is_file():
        raise ValueError(
            "hf_clip_vit local model directory contains pytorch_model.bin but no model.safetensors: "
            f"{model_dir}. Convert the weights to safetensors before running this preflight."
        )
    raise FileNotFoundError(
        "hf_clip_vit local model directory is missing model.safetensors: "
        f"{model_dir}. Do not rely on HuggingFace downloads during preflight."
    )


# ---------------------------------------------------------------------------
#  Minimal smoke encoder (dev only)
# ---------------------------------------------------------------------------

def _build_minimal_patch_encoder(
    model_config: VisualEncoderConfig,
    image_size: ImageSize,
) -> MinimalPatchStudentEncoder:
    output_patch_dim = _resolve_output_patch_dim(model_config)
    model = MinimalPatchStudentEncoder(
        image_size=image_size,
        patch_size=int(model_config.patch_size),
        latent_dim=output_patch_dim,
    )
    model.visual_encoder_backend = MINIMAL_PATCH_ENCODER
    model.visual_encoder_impl_class = type(model).__name__
    model.pretrained_weight_path = None
    model.encoder_output_dim = output_patch_dim
    model.encoder_trainable = not bool(model_config.freeze_backbone)
    model.patch_grid = _patch_grid(image_size, model_config.patch_size)
    model.pretrained_load_report = {}
    model.is_real_pretrained_backbone = False
    model.is_minimal_or_wrapper_backend = True
    return model


# ---------------------------------------------------------------------------
#  LocalTorchCheckpointVisualEncoder – compliance smoke only
# ---------------------------------------------------------------------------

class _CheckpointLoadingMixin:
    """Shared checkpoint-loading logic independent of encoder architecture."""

    def _load_pretrained_checkpoint(self, weight_path: Path) -> dict[str, object]:
        payload = torch.load(weight_path, map_location="cpu")
        state = _extract_visual_state_dict(payload)
        load_result = self.load_state_dict(state, strict=False)
        return {
            "weight_path": str(weight_path),
            "loaded_tensor_count": len(state),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }


class LocalTorchCheckpointVisualEncoder(MinimalPatchStudentEncoder, _CheckpointLoadingMixin):
    """Auditable local checkpoint backend for compliance smoke only.

    This backend deliberately avoids importing external repository code. It
    provides the same Stage 1 encoder contract as the smoke encoder while
    requiring an explicit local checkpoint path.

    IMPORTANT: This wraps MinimalPatchStudentEncoder, so it is NOT a real
    pretrained backbone. Final configs must use a real backend (e.g.
    generic_timm_vit) and NOT this wrapper.
    """

    def __init__(
        self,
        *,
        image_size: ImageSize,
        patch_size: int,
        latent_dim: int,
        backend_name: str,
        weight_path: Path,
        freeze_backbone: bool,
    ) -> None:
        super().__init__(image_size=image_size, patch_size=patch_size, latent_dim=latent_dim)
        self.visual_encoder_backend = backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.pretrained_weight_path = str(weight_path)
        self.encoder_output_dim = int(latent_dim)
        self.encoder_trainable = not bool(freeze_backbone)
        self.patch_grid = _patch_grid(image_size, patch_size)
        self.pretrained_load_report = self._load_pretrained_checkpoint(weight_path)
        self.is_real_pretrained_backbone = False
        self.is_minimal_or_wrapper_backend = True
        if freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False


def _extract_visual_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        state = (
            payload.get("vision_encoder_state_dict")
            or payload.get("model_state_dict")
            or payload.get("state_dict")
            or payload.get("model")
            or payload
        )
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError("Pretrained visual checkpoint must contain a state_dict mapping.")

    stripped: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("vision_encoder."):
            key = key[len("vision_encoder."):]
        if isinstance(value, torch.Tensor):
            stripped[key] = value
    if not stripped:
        raise ValueError("Pretrained visual checkpoint did not contain tensor weights.")
    return stripped


# ---------------------------------------------------------------------------
#  GenericTimmViT – real timm ViT backbone (not a MinimalPatchStudentEncoder)
# ---------------------------------------------------------------------------

class GenericTimmViT(nn.Module):
    """Real timm ViT backend for final Stage 1 configs.

    This class does NOT inherit from MinimalPatchStudentEncoder. It uses the
    ``timm`` library to create a real Vision Transformer and outputs the
    unified Stage 1 encoder contract.

    When timm is not installed or weights are missing, this fails fast.
    """

    def __init__(
        self,
        *,
        image_size: ImageSize,
        patch_size: int,
        latent_dim: int,
        backend_name: str,
        weight_path: Path | None,
        freeze_backbone: bool,
        model_name: str = "vit_small_patch16_224",
    ) -> None:
        super().__init__()
        try:
            import timm  # noqa: F401
        except ImportError:
            raise ImportError(
                "generic_timm_vit backend requires timm. "
                "Install with: pip install timm"
            )

        self.visual_encoder_backend = backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.encoder_output_dim = int(latent_dim)
        self.encoder_trainable = not bool(freeze_backbone)
        self.image_size = normalize_image_size(image_size)
        self.patch_size = int(patch_size)
        self.patch_grid = _patch_grid(image_size, patch_size)
        self._num_patches = self.patch_grid[0] * self.patch_grid[1]
        self.pretrained_weight_path = str(weight_path) if weight_path else None
        self.pretrained_load_report: dict[str, object] = {}
        self.is_real_pretrained_backbone = True
        self.is_minimal_or_wrapper_backend = False

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=self.image_size,
        )
        # Project backbone features to latent_dim
        _backbone_dim = getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", latent_dim))
        self.proj = nn.Linear(_backbone_dim, latent_dim) if _backbone_dim != latent_dim else nn.Identity()

        if weight_path is not None and weight_path.exists():
            self._load_weights(weight_path)
        elif weight_path is not None:
            raise FileNotFoundError(
                f"generic_timm_vit pretrained weight not found: {weight_path}"
            )
        # else: no weight path → random init (smoke/ablation only)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def _load_weights(self, weight_path: Path) -> None:
        payload = torch.load(weight_path, map_location="cpu")
        state = _extract_visual_state_dict(payload)
        load_result = self.load_state_dict(state, strict=False)
        self.pretrained_load_report = {
            "weight_path": str(weight_path),
            "loaded_tensor_count": len(state),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    def forward(self, image: torch.Tensor, return_dict: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.backbone.forward_features(image)
        # timm ViT forward_features returns [B, N+1, D] (with cls token)
        # or [B, N, D] depending on model config
        if features.ndim == 3 and features.shape[1] == self._num_patches + 1:
            patch_tokens = features[:, 1:, :]  # strip cls token
            global_image_feature = features[:, 0, :]
        elif features.ndim == 3:
            patch_tokens = features
            global_image_feature = features.mean(dim=1)
        else:
            patch_tokens = features.unsqueeze(1)
            global_image_feature = features

        patch_tokens = self.proj(patch_tokens)
        global_image_feature = self.proj(global_image_feature)

        if not return_dict:
            return patch_tokens
        return {
            "patch_tokens": patch_tokens,
            "global_image_feature": global_image_feature,
        }


# ---------------------------------------------------------------------------
#  HF CLIP ViT backend
# ---------------------------------------------------------------------------

class HfClipViT(nn.Module):
    """HuggingFace CLIP ViT backend for Stage 1."""

    def __init__(
        self,
        *,
        image_size: ImageSize,
        patch_size: int,
        latent_dim: int,
        backend_name: str,
        weight_path: Path | None,
        freeze_backbone: bool,
        model_name: str = "openai/clip-vit-base-patch16",
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPVisionModel  # noqa: F401
        except ImportError:
            raise ImportError(
                "hf_clip_vit backend requires transformers. "
                "Install with: pip install transformers"
            )

        self.visual_encoder_backend = backend_name
        self.visual_encoder_impl_class = type(self).__name__
        self.encoder_output_dim = int(latent_dim)
        self.encoder_trainable = not bool(freeze_backbone)
        self.image_size = normalize_image_size(image_size)
        self.patch_size = int(patch_size)
        self.patch_grid = _patch_grid(image_size, patch_size)
        self._num_patches = self.patch_grid[0] * self.patch_grid[1]
        self.pretrained_weight_path = str(weight_path) if weight_path else None
        self.pretrained_load_report: dict[str, object] = {}
        self.is_real_pretrained_backbone = True
        self.is_minimal_or_wrapper_backend = False

        if weight_path is None:
            raise ValueError(
                "hf_clip_vit / clip_vit_b16 requires a HuggingFace local directory via "
                "model.pretrained_weight_path or model.pretrained_model_path. "
                f"Refusing to load remote model name: {model_name}."
            )
        if weight_path.is_dir():
            model_source, from_pretrained_kwargs = _resolve_hf_clip_local_dir(weight_path)
        elif weight_path.is_file():
            raise ValueError(
                "hf_clip_vit / clip_vit_b16 currently only accepts a HuggingFace local directory, "
                f"not a checkpoint file: {weight_path}."
            )
        else:
            raise FileNotFoundError(f"hf_clip_vit local model directory not found: {weight_path}")

        self.backbone = CLIPVisionModel.from_pretrained(
            model_source,
            **from_pretrained_kwargs,
        )
        _backbone_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(_backbone_dim, latent_dim) if _backbone_dim != latent_dim else nn.Identity()
        self.register_buffer(
            "_clip_pixel_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_pixel_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.pretrained_load_report = {
            "model_path": str(weight_path),
            "loaded_from": "huggingface_local_dir",
            "local_files_only": True,
            "use_safetensors": bool(from_pretrained_kwargs.get("use_safetensors", False)),
            "hidden_size": int(_backbone_dim),
        }

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def _load_weights(self, weight_path: Path) -> None:
        payload = torch.load(weight_path, map_location="cpu")
        state = _extract_visual_state_dict(payload)
        load_result = self.load_state_dict(state, strict=False)
        self.pretrained_load_report = {
            "weight_path": str(weight_path),
            "loaded_tensor_count": len(state),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    def forward(self, image: torch.Tensor, return_dict: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        normalized_image = (image - self._clip_pixel_mean.to(dtype=image.dtype)) / self._clip_pixel_std.to(dtype=image.dtype)
        outputs = self.backbone(normalized_image, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        if int(patch_tokens.shape[1]) != self._num_patches:
            raise ValueError(
                "HF CLIP patch token count does not match Stage 1 config: "
                f"got {int(patch_tokens.shape[1])}, expected {self._num_patches}."
            )
        global_image_feature = outputs.pooler_output
        patch_tokens = self.proj(patch_tokens)
        global_image_feature = self.proj(global_image_feature)
        if not return_dict:
            return patch_tokens
        return {
            "patch_tokens": patch_tokens,
            "global_image_feature": global_image_feature,
        }


# ---------------------------------------------------------------------------
#  Adapter registry stubs (mammo_clip, mammo_fm, biomedclip, medsiglip, rad_dino)
# ---------------------------------------------------------------------------

def _build_adapter_stub(
    *,
    backend_name: str,
    image_size: ImageSize,
    patch_size: int,
    latent_dim: int,
    weight_path: Path | None,
    freeze_backbone: bool,
) -> nn.Module:
    """Placeholder for external pretrained backends.

    These backends reference external model architectures (Mammo-FM,
    Mammo-CLIP, BiomedCLIP, MedSigLIP, Rad-DINO) whose implementations live
    in their respective author repositories (read-only reference only).

    When a real weight path is provided and the external package is
    importable, this creates a real adapter. Otherwise it fails fast.
    """
    import importlib

    adapter_registry = {
        "biomedclip": ("transformers", "AutoModel", "microsoft/BiomedCLIP-PubMedBERT"),
        "medsiglip": ("transformers", "AutoModel", "google/medsiglip"),
        "mammo_clip": ("breast_pretrain.reference.mammo_clip_adapter", "MammoClipAdapter", None),
        "mammo_fm": ("breast_pretrain.reference.mammo_fm_adapter", "MammoFMAdapter", None),
        "rad_dino": ("breast_pretrain.reference.rad_dino_adapter", "RadDinoAdapter", None),
        "clip_vit_b16": ("transformers", "CLIPVisionModel", "openai/clip-vit-base-patch16"),
    }

    entry = adapter_registry.get(backend_name)
    if entry is None:
        available = sorted(adapter_registry)
        raise ValueError(
            f"Unsupported real backend {backend_name!r}. Available adapters: {available}"
        )

    pkg, class_name, _default_model = entry

    try:
        mod = importlib.import_module(pkg)
        adapter_cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Real backend {backend_name!r} requires {pkg}.{class_name} but it could not be loaded. "
            f"Either install the dependency or use a different backend. "
            f"Original error: {exc}"
        ) from exc

    model = adapter_cls(
        image_size=image_size,
        patch_size=patch_size,
        latent_dim=latent_dim,
        weight_path=weight_path,
        freeze_backbone=freeze_backbone,
    )
    model.visual_encoder_backend = backend_name
    model.visual_encoder_impl_class = type(model).__name__
    model.is_real_pretrained_backbone = True
    model.is_minimal_or_wrapper_backend = False
    return model


# ---------------------------------------------------------------------------
#  Builder dispatch
# ---------------------------------------------------------------------------

def _build_real_backend(
    model_config: VisualEncoderConfig,
    image_size: ImageSize,
    encoder_name: str,
) -> nn.Module:
    output_patch_dim = _resolve_output_patch_dim(model_config)
    raw_weight_path = _candidate_weight_path(model_config)
    weight_path = Path(raw_weight_path).expanduser().resolve() if raw_weight_path else None

    freeze_backbone = bool(model_config.freeze_backbone)
    common_kwargs = dict(
        image_size=image_size,
        patch_size=int(model_config.patch_size),
        latent_dim=output_patch_dim,
        backend_name=encoder_name,
        weight_path=weight_path,
        freeze_backbone=freeze_backbone,
    )

    if encoder_name in {"generic_timm_vit"}:
        return GenericTimmViT(**common_kwargs)

    if encoder_name in {"hf_clip_vit", "clip_vit_b16"}:
        return HfClipViT(**common_kwargs)

    if encoder_name in HIGHRES_REAL_BACKEND_NAMES:
        from breast_pretrain.models.highres_real_backbone_adapter import (
            HighresRealBackboneConfig,
            build_highres_real_backbone_adapter,
        )

        return build_highres_real_backbone_adapter(
            HighresRealBackboneConfig(
                backend_name=encoder_name,
                image_size=image_size,
                patch_size=int(model_config.patch_size),
                encoder_output_dim=output_patch_dim,
                pretrained_weight_path=weight_path,
                backbone_expected_sha256=model_config.backbone_expected_sha256,
                freeze_backbone=freeze_backbone,
            )
        )

    # All other real backends: biomedclip, medsiglip, mammo_clip, mammo_fm, rad_dino
    return _build_adapter_stub(**common_kwargs)


def _build_wrapper_encoder(
    model_config: VisualEncoderConfig,
    image_size: ImageSize,
    encoder_name: str,
) -> nn.Module:
    """Build a local-checkpoint wrapper (compliance smoke / legacy only)."""
    output_patch_dim = _resolve_output_patch_dim(model_config)
    raw_weight_path = _candidate_weight_path(model_config)

    if not raw_weight_path:
        if bool(model_config.allow_missing_pretrained_fallback):
            warnings.warn(
                "Pretrained visual encoder path is missing; falling back to minimal_patch_encoder "
                "because model.allow_missing_pretrained_fallback=true. "
                "Summary will be marked not_final_backbone.",
                stacklevel=2,
            )
            model = _build_minimal_patch_encoder(model_config, image_size=image_size)
            model.visual_encoder_backend = encoder_name
            return model
        raise FileNotFoundError(
            "Final/pretrained visual encoder configs must set model.pretrained_weight_path "
            "or model.pretrained_model_path."
        )

    weight_path = Path(raw_weight_path).expanduser()
    if not weight_path.is_absolute():
        weight_path = weight_path.resolve()
    if not weight_path.exists():
        if bool(model_config.allow_missing_pretrained_fallback):
            warnings.warn(
                f"Pretrained visual encoder checkpoint is missing: {weight_path}; "
                "falling back to minimal_patch_encoder for smoke/dev only. "
                "Summary will be marked not_final_backbone.",
                stacklevel=2,
            )
            model = _build_minimal_patch_encoder(model_config, image_size=image_size)
            model.visual_encoder_backend = encoder_name
            return model
        raise FileNotFoundError(f"Pretrained visual encoder checkpoint does not exist: {weight_path}")

    return LocalTorchCheckpointVisualEncoder(
        image_size=image_size,
        patch_size=int(model_config.patch_size),
        latent_dim=output_patch_dim,
        backend_name=encoder_name,
        weight_path=weight_path,
        freeze_backbone=bool(model_config.freeze_backbone),
    )


# ---------------------------------------------------------------------------
#  Public factory
# ---------------------------------------------------------------------------

def build_visual_encoder(
    model_config: VisualEncoderConfig,
    image_size: ImageSize,
) -> nn.Module:
    """Build a Stage 1 visual encoder with explicit smoke/final separation.

    Routing:
      minimal_patch_encoder      → MinimalPatchStudentEncoder    (smoke / dev only)
      local_torch_checkpoint     → LocalTorchCheckpointVisualEncoder (compliance smoke)
      generic_timm_vit           → GenericTimmViT                (real timm ViT)
      timm_highres_hierarchical  → high-res timm adapter         (real weights required)
      mammo_fm_like_highres_adapter → timm-backed reference-shape adapter
      mammo_fm_timm_efficientnet_b5 → Mammo-FM stripped EfficientNet-B5 global adapter
      hf_clip_vit / clip_vit_b16 → HfClipViT                     (HuggingFace CLIP)
      biomedclip / medsiglip /   → adapter stub                  (external reference)
      mammo_clip / mammo_fm /
      rad_dino
    """

    encoder_name = _normalise_encoder_name(model_config.vision_encoder_name)

    if encoder_name == MINIMAL_PATCH_ENCODER:
        return _build_minimal_patch_encoder(model_config, image_size=image_size)

    if encoder_name in WRAPPER_BACKEND_NAMES:
        return _build_wrapper_encoder(model_config, image_size=image_size, encoder_name=encoder_name)

    if encoder_name in REAL_BACKEND_NAMES or encoder_name in HIGHRES_REAL_BACKEND_NAMES:
        return _build_real_backend(model_config, image_size=image_size, encoder_name=encoder_name)

    supported = ", ".join(sorted({MINIMAL_PATCH_ENCODER, *PRETRAINED_BACKEND_NAMES}))
    raise ValueError(
        "Unsupported model.vision_encoder_name: "
        f"{model_config.vision_encoder_name!r}. Supported values are: {supported}."
    )


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def visual_encoder_summary(
    model: nn.Module,
    *,
    patch_size: int,
    image_size: ImageSize,
    vision_encoder_name: str | None = None,
) -> dict[str, object]:
    model_grid = getattr(model, "patch_grid", None)
    grid = list(model_grid) if model_grid is not None else _patch_grid(image_size, patch_size)
    backend = str(getattr(model, "visual_encoder_backend", type(model).__name__))
    configured_name = _normalise_encoder_name(vision_encoder_name or backend)
    num_patches = getattr(model, "num_patches", None)
    if num_patches is None:
        num_patches = int(grid[0]) * int(grid[1])

    # Contract status fields from frozen HighresEncoderOutput contract
    is_formal = bool(getattr(model, "is_formal_production_backbone", False))
    contract_status = "formal_frozen" if is_formal else "candidate"
    spatial_ready = bool(getattr(model, "spatial_patch_tokens_ready", is_formal))

    return {
        "vision_encoder_name": configured_name,
        "backend_name": backend,
        "visual_encoder_backend": backend,
        "visual_encoder_impl_class": str(getattr(model, "visual_encoder_impl_class", type(model).__name__)),
        "pretrained_weight_path": getattr(model, "pretrained_weight_path", None),
        "encoder_trainable": bool(getattr(model, "encoder_trainable", any(p.requires_grad for p in model.parameters()))),
        "encoder_output_dim": int(getattr(model, "encoder_output_dim", getattr(model, "latent_dim", 0))),
        "patch_grid": [int(grid[0]), int(grid[1])],
        "num_patches": int(num_patches),
        "patch_size": int(patch_size),
        "supports_modality_specific_image_size": bool(
            getattr(model, "supports_modality_specific_image_size", False)
        ),
        "input_normalization_audit": getattr(model, "last_input_normalization_audit", {}),
        "is_formal_production_backbone": is_formal,
        "is_real_pretrained_backbone": bool(getattr(model, "is_real_pretrained_backbone", False)),
        "is_minimal_or_wrapper_backend": bool(getattr(model, "is_minimal_or_wrapper_backend", True)),
        "pretrained_load_report": getattr(model, "pretrained_load_report", {}),
        "contract_status": contract_status,
        "spatial_patch_tokens_ready": spatial_ready,
        "multi_scale_features_ready": bool(getattr(model, "multi_scale_features_ready", False)),
        "local_branch_training_ready": bool(getattr(model, "local_branch_training_ready", False)),
    }


__all__ = [
    "GENERIC_TORCH_CHECKPOINT",
    "GenericTimmViT",
    "HIGHRES_REAL_BACKEND_NAMES",
    "HfClipViT",
    "LOCAL_TORCH_CHECKPOINT",
    "LocalTorchCheckpointVisualEncoder",
    "MAMMO_FM_LIKE_HIGHRES_BACKEND",
    "MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND",
    "MINIMAL_PATCH_ENCODER",
    "PRETRAINED_BACKEND_NAMES",
    "PRETRAINED_VISUAL_ENCODER",
    "REAL_BACKEND_NAMES",
    "TIMM_HIGHRES_HIERARCHICAL_BACKEND",
    "WRAPPER_BACKEND_NAMES",
    "build_visual_encoder",
    "visual_encoder_summary",
]
