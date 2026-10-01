"""Formal V2 image geometry contract.

Only ``image_size_by_modality`` is user-configurable.  The legacy scalar
``image_size`` is derived for legacy consumers and never read from V2 YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from breast_pretrain.data_entry.release_common import canonical_json, sha256_text


FORMAL_MODALITIES = ("mammography", "mri", "ultrasound")
ROW_MAJOR_TOKEN_ORDER = "row_major_h_w_v1"
FORMAL_PATCH_SIZE = 16
FORMAL_IMAGE_SIZES = {
    "mammography": (912, 1520),
    "mri": (512, 512),
    "ultrasound": (512, 512),
}


def normalize_modality(value: object) -> str:
    modality = str(value or "").strip().lower()
    return {"mammo": "mammography", "us": "ultrasound"}.get(modality, modality)


@dataclass(frozen=True)
class ModalityGeometry:
    modality: str
    image_size_hw: tuple[int, int]
    patch_size_hw: tuple[int, int]
    stride_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    patch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "image_size_hw": list(self.image_size_hw),
            "patch_size_hw": list(self.patch_size_hw),
            "stride_hw": list(self.stride_hw),
            "grid_hw": list(self.grid_hw),
            "patch_count": self.patch_count,
            "patch_token_order_version": ROW_MAJOR_TOKEN_ORDER,
            "token_index_formula": "token_index = row * grid_w + column",
        }


@dataclass(frozen=True)
class ResolvedGeometryContract:
    patch_size: int
    by_modality: dict[str, ModalityGeometry]

    @property
    def legacy_image_size(self) -> tuple[int, int]:
        return self.by_modality["mammography"].image_size_hw

    @property
    def sha256(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "resolved_geometry_contract_v1",
            "patch_size": self.patch_size,
            "patch_token_order_version": ROW_MAJOR_TOKEN_ORDER,
            "token_index_formula": "token_index = row * grid_w + column",
            "image_size_by_modality": {
                modality: value.to_dict() for modality, value in sorted(self.by_modality.items())
            },
            "legacy_image_size_derived": list(self.legacy_image_size),
        }

    def geometry_for(self, modality: object) -> ModalityGeometry:
        normalized = normalize_modality(modality)
        try:
            return self.by_modality[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported formal modality: {modality!r}") from exc


def resolve_geometry_contract(raw_config: dict[str, Any]) -> ResolvedGeometryContract:
    if "image_size" in raw_config:
        raise ValueError(
            "Formal Dataset Entry V2 forbids user-configured image_size; "
            "the legacy value is derived from image_size_by_modality.mammography."
        )
    raw_sizes = raw_config.get("image_size_by_modality")
    if not isinstance(raw_sizes, dict):
        raise ValueError("Formal Dataset Entry V2 requires image_size_by_modality mapping.")
    patch_size = int(raw_config.get("patch_size", FORMAL_PATCH_SIZE))
    if patch_size != FORMAL_PATCH_SIZE:
        raise ValueError(f"Formal Dataset Entry V2 requires patch_size={FORMAL_PATCH_SIZE}, got {patch_size}.")
    normalized_sizes = {normalize_modality(key): value for key, value in raw_sizes.items()}
    if set(normalized_sizes) != set(FORMAL_MODALITIES):
        raise ValueError(
            "image_size_by_modality must contain exactly mammography, mri, ultrasound; "
            f"got {sorted(normalized_sizes)}."
        )
    resolved: dict[str, ModalityGeometry] = {}
    for modality in FORMAL_MODALITIES:
        size = normalized_sizes[modality]
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"image_size_by_modality.{modality} must be [height, width].")
        height, width = int(size[0]), int(size[1])
        if (height, width) != FORMAL_IMAGE_SIZES[modality]:
            raise ValueError(
                f"Formal {modality} image size must be {FORMAL_IMAGE_SIZES[modality]}, got {(height, width)}."
            )
        if height % patch_size or width % patch_size:
            raise ValueError(f"{modality} image size {(height, width)} is not divisible by patch_size={patch_size}.")
        grid = (height // patch_size, width // patch_size)
        resolved[modality] = ModalityGeometry(
            modality=modality,
            image_size_hw=(height, width),
            patch_size_hw=(patch_size, patch_size),
            stride_hw=(patch_size, patch_size),
            grid_hw=grid,
            patch_count=grid[0] * grid[1],
        )
    return ResolvedGeometryContract(patch_size=patch_size, by_modality=resolved)


__all__ = [
    "FORMAL_IMAGE_SIZES",
    "FORMAL_MODALITIES",
    "FORMAL_PATCH_SIZE",
    "ModalityGeometry",
    "ROW_MAJOR_TOKEN_ORDER",
    "ResolvedGeometryContract",
    "normalize_modality",
    "resolve_geometry_contract",
]
