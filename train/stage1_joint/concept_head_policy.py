from __future__ import annotations

from typing import Any

STRICT_CONFIRMED_POLICY = "confirmed_labels_only_with_observed_mask_strict"
SUPPORTED_CONCEPT_HEAD_POLICIES = frozenset({STRICT_CONFIRMED_POLICY})
_FINAL_TIERS = {"formal_production", "production_ready_candidate", "final", "final_config_template"}


def parse_concept_head_policy(raw_value: Any) -> str:
    policy = str(raw_value or STRICT_CONFIRMED_POLICY).strip().lower()
    if policy not in SUPPORTED_CONCEPT_HEAD_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_CONCEPT_HEAD_POLICIES))
        raise ValueError(f"semantic.concept_head_policy must be one of {supported}; got {policy!r}.")
    return policy


def validate_concept_head_policy(
    *,
    policy: str,
    active_heads: tuple[str, ...],
    pending_heads: tuple[str, ...],
    concept_head_weights: dict[str, float],
    concept_consistency_head_weights: dict[str, float],
    run_tier: str,
) -> None:
    if policy != STRICT_CONFIRMED_POLICY:
        return

    active_set = set(active_heads)
    pending_set = set(pending_heads)
    overlap = sorted(active_set & pending_set)
    if overlap:
        raise ValueError(f"semantic active/pending concept heads overlap: {overlap}.")

    final_tier = str(run_tier).strip() in _FINAL_TIERS
    if not final_tier:
        return

    for head_name in sorted(pending_set):
        if float(concept_head_weights.get(head_name, 0.0)) > 0.0:
            raise ValueError(
                "confirmed_labels_only_with_observed_mask_strict forbids positive "
                f"losses.concept_head_weights.{head_name} while the head is pending."
            )
        if float(concept_consistency_head_weights.get(head_name, 0.0)) > 0.0:
            raise ValueError(
                "confirmed_labels_only_with_observed_mask_strict forbids positive "
                f"losses.concept_consistency_head_weights.{head_name} while the head is pending."
            )


__all__ = [
    "STRICT_CONFIRMED_POLICY",
    "SUPPORTED_CONCEPT_HEAD_POLICIES",
    "parse_concept_head_policy",
    "validate_concept_head_policy",
]
