from __future__ import annotations

from dataclasses import dataclass


PREFLIGHT_HIGHRES_STUB_BACKEND = "preflight_highres_hierarchical_stub"
TIMM_HIGHRES_HIERARCHICAL_BACKEND = "timm_highres_hierarchical"
MAMMO_FM_LIKE_HIGHRES_BACKEND = "mammo_fm_like_highres_adapter"
MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND = "mammo_fm_timm_efficientnet_b5"


@dataclass(frozen=True)
class HighresBackboneSpec:
    backend_name: str
    is_stub: bool
    requires_timm: bool
    requires_real_weights: bool
    default_timm_model_name: str | None = None


_REGISTRY: dict[str, HighresBackboneSpec] = {
    PREFLIGHT_HIGHRES_STUB_BACKEND: HighresBackboneSpec(
        backend_name=PREFLIGHT_HIGHRES_STUB_BACKEND,
        is_stub=True,
        requires_timm=False,
        requires_real_weights=False,
        default_timm_model_name=None,
    ),
    TIMM_HIGHRES_HIERARCHICAL_BACKEND: HighresBackboneSpec(
        backend_name=TIMM_HIGHRES_HIERARCHICAL_BACKEND,
        is_stub=False,
        requires_timm=True,
        requires_real_weights=True,
        default_timm_model_name="vit_base_patch16_224",
    ),
    MAMMO_FM_LIKE_HIGHRES_BACKEND: HighresBackboneSpec(
        backend_name=MAMMO_FM_LIKE_HIGHRES_BACKEND,
        is_stub=False,
        requires_timm=True,
        requires_real_weights=True,
        default_timm_model_name="swin_base_patch4_window7_224",
    ),
    MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND: HighresBackboneSpec(
        backend_name=MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND,
        is_stub=False,
        requires_timm=True,
        requires_real_weights=True,
        default_timm_model_name="tf_efficientnet_b5",
    ),
}


def normalize_highres_backend_name(raw_name: str | None) -> str:
    return str(raw_name or PREFLIGHT_HIGHRES_STUB_BACKEND).strip().lower() or PREFLIGHT_HIGHRES_STUB_BACKEND


def get_highres_backbone_spec(backend_name: str | None) -> HighresBackboneSpec:
    normalized = normalize_highres_backend_name(backend_name)
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unsupported high-res backbone backend {normalized!r}. Supported values are: {supported}.") from exc


def registered_highres_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def real_highres_backends() -> tuple[str, ...]:
    return tuple(sorted(name for name, spec in _REGISTRY.items() if not spec.is_stub))


__all__ = [
    "HighresBackboneSpec",
    "MAMMO_FM_LIKE_HIGHRES_BACKEND",
    "MAMMO_FM_TIMM_EFFICIENTNET_B5_BACKEND",
    "PREFLIGHT_HIGHRES_STUB_BACKEND",
    "TIMM_HIGHRES_HIERARCHICAL_BACKEND",
    "get_highres_backbone_spec",
    "normalize_highres_backend_name",
    "real_highres_backends",
    "registered_highres_backends",
]
