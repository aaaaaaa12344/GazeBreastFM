from __future__ import annotations

from typing import Any

import torch

from breast_pretrain.train.stage1_joint.types import (
    MaskingStepOutput,
    Stage1JointTrainerConfig,
)


LOSS_TERM_KEYS = (
    "reconstruction",
    "global_align",
    "visible_align",
    "semantic_soft",
    "concept_loss",
    "concept_consistency",
    "graph_consistency",
)


def _coverage_mean(attention_tokens: torch.Tensor, patch_mask: torch.Tensor, masked: bool) -> float:
    mass = attention_tokens.detach().clamp_min(0.0)
    total = mass.sum(dim=1).clamp_min(1e-6)
    mask = patch_mask if masked else ~patch_mask
    coverage = (mass * mask.to(dtype=mass.dtype)).sum(dim=1) / total
    return float(coverage.mean().item()) if coverage.numel() else 0.0


def _clamp_weight(value: float, min_weight: float, max_weight: float) -> float:
    return min(max(float(value), float(min_weight)), float(max_weight))


def base_dynamic_loss_weights() -> dict[str, float]:
    return {key: 1.0 for key in LOSS_TERM_KEYS}


def compute_conflict_aware_weights(
    config: Stage1JointTrainerConfig,
    masking_output: MaskingStepOutput,
) -> tuple[dict[str, float], list[str]]:
    weights = base_dynamic_loss_weights()
    if not config.losses.conflict_aware_enabled:
        return weights, []

    if str(config.masking.gaze_loss_mode).strip().lower() in {"", "no_gaze", "none"}:
        return weights, [
            "conflict_aware_fallback_base_weights:no_gaze_config"
        ]

    gaze_mass = masking_output.attention_tokens.detach().clamp_min(0.0).sum().item()
    if gaze_mass < float(config.masking.gaze_mask_eps):
        return weights, [
            "conflict_aware_fallback_base_weights:gaze_attention_mass_zero_or_below_eps"
        ]

    masked_coverage = _coverage_mean(masking_output.attention_tokens, masking_output.patch_mask, masked=True)
    visible_coverage = _coverage_mean(masking_output.attention_tokens, masking_output.patch_mask, masked=False)
    min_weight = config.losses.conflict_aware_min_weight
    max_weight = config.losses.conflict_aware_max_weight
    semantic_target = max(config.losses.conflict_aware_semantic_visible_coverage_target, 1e-6)
    recon_target = max(config.losses.conflict_aware_reconstruction_masked_gaze_target, 1e-6)

    semantic_multiplier = _clamp_weight(visible_coverage / semantic_target, min_weight, max_weight)
    if masked_coverage > recon_target:
        reconstruction_multiplier = _clamp_weight(masked_coverage / recon_target, min_weight, max_weight)
    else:
        reconstruction_multiplier = 1.0

    warnings_list: list[str] = []
    if visible_coverage < semantic_target:
        warnings_list.append(
            "conflict_aware_visible_semantic_downweighted:"
            f"visible_gaze_coverage={visible_coverage:.6f}<target={semantic_target:.6f}"
        )
    if masked_coverage <= recon_target:
        warnings_list.append(
            "conflict_aware_reconstruction_base_weight:"
            f"masked_gaze_coverage={masked_coverage:.6f}<=target={recon_target:.6f}"
        )
    else:
        warnings_list.append(
            "conflict_aware_reconstruction_upweighted:"
            f"masked_gaze_coverage={masked_coverage:.6f}>target={recon_target:.6f}"
        )

    weights.update(
        {
            "reconstruction": reconstruction_multiplier,
            "visible_align": semantic_multiplier,
            "semantic_soft": semantic_multiplier,
            "concept_loss": semantic_multiplier,
            "concept_consistency": semantic_multiplier,
        }
    )
    if config.losses.allow_dynamic_graph_consistency_weighting:
        weights["graph_consistency"] = semantic_multiplier
    return weights, warnings_list


def build_loss_weight_audit(
    *,
    losses: dict[str, torch.Tensor],
    static_weights: dict[str, float],
    dynamic_weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    loss_key_map = {
        "reconstruction": "reconstruction_total",
        "global_align": "global_align",
        "visible_align": "visible_align",
        "semantic_soft": "semantic_soft",
        "concept_loss": "concept_cls",
        "concept_consistency": "concept_consistency",
        "graph_consistency": "graph_consistency",
    }
    audit: dict[str, dict[str, float]] = {}
    for term_name, loss_key in loss_key_map.items():
        raw_loss = float(losses[loss_key].detach().item())
        static_weight = float(static_weights.get(term_name, 0.0))
        dynamic_weight = float(dynamic_weights.get(term_name, 1.0))
        effective_weight = static_weight * dynamic_weight
        audit[term_name] = {
            "raw_loss": raw_loss,
            "static_weight": static_weight,
            "dynamic_weight": dynamic_weight,
            "effective_weight": effective_weight,
            "weighted_loss": raw_loss * effective_weight,
        }
    if "reconstruction_unweighted" in losses:
        audit["masked_reconstruction_unweighted_reference"] = {
            "raw_loss": float(losses["reconstruction_unweighted"].detach().item()),
            "static_weight": float(static_weights.get("reconstruction", 0.0)),
            "dynamic_weight": float(dynamic_weights.get("reconstruction", 1.0)),
            "effective_weight": float(static_weights.get("reconstruction", 0.0))
            * float(dynamic_weights.get("reconstruction", 1.0)),
            "weighted_loss": float(losses["reconstruction_unweighted"].detach().item())
            * float(static_weights.get("reconstruction", 0.0))
            * float(dynamic_weights.get("reconstruction", 1.0)),
        }
    return audit


def loss_weight_static_config(config: Stage1JointTrainerConfig) -> dict[str, float]:
    return {
        "reconstruction": float(config.losses.reconstruction_weight),
        "global_align": float(config.losses.global_align_weight),
        "visible_align": float(config.losses.visible_align_weight),
        "semantic_soft": float(config.losses.semantic_soft_weight),
        "concept_loss": float(config.losses.concept_loss_weight),
        "concept_consistency": float(config.losses.concept_consistency_weight),
        "graph_consistency": float(config.losses.graph_consistency_weight),
    }


def validate_loss_weight_audit_payload(payload: dict[str, Any]) -> None:
    required_fields = {"raw_loss", "static_weight", "effective_weight", "weighted_loss"}
    for term_name, term_payload in payload.items():
        missing = required_fields - set(term_payload)
        if missing:
            raise ValueError(f"loss weight audit term {term_name!r} missing fields: {sorted(missing)}")


__all__ = [
    "LOSS_TERM_KEYS",
    "base_dynamic_loss_weights",
    "build_loss_weight_audit",
    "compute_conflict_aware_weights",
    "loss_weight_static_config",
    "validate_loss_weight_audit_payload",
]
