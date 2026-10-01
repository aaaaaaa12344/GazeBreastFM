from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CONFIRMED = "confirmed"
RAW_UNCONFIRMED = "raw_unconfirmed"
MISSING = "missing"
NOT_APPLICABLE = "not_applicable"

_CONFIRMED_ALIASES = {"confirmed", "present", "explicit_absent", "observed", "structured_label"}
_RAW_UNCONFIRMED_ALIASES = {"raw_unconfirmed", "unconfirmed", "raw", "weak", "heuristic"}
_NOT_APPLICABLE_ALIASES = {"not_applicable", "not applicable", "n/a", "na"}
_MISSING_ALIASES = {"", "missing", "unknown", "not_mentioned", "none", "null"}


@dataclass(frozen=True)
class CanonicalConceptLabel:
    value: str
    status: str
    observed_mask: bool
    source: str
    confidence: float


def _clean(value: Any) -> str:
    return str(value or "").strip()


def canonicalize_concept_status(raw_status: Any, raw_value: Any = "") -> str:
    status = _clean(raw_status).lower()
    value = _clean(raw_value).lower()
    if status == "explicit_absent":
        return CONFIRMED
    if status in _CONFIRMED_ALIASES:
        return CONFIRMED if value not in _MISSING_ALIASES and value not in _NOT_APPLICABLE_ALIASES else MISSING
    if status in _RAW_UNCONFIRMED_ALIASES:
        return RAW_UNCONFIRMED
    if status in _NOT_APPLICABLE_ALIASES or value in _NOT_APPLICABLE_ALIASES:
        return NOT_APPLICABLE
    if status in _MISSING_ALIASES:
        return MISSING
    return RAW_UNCONFIRMED


def parse_observed_mask(raw_mask: Any) -> bool | None:
    value = _clean(raw_mask).lower()
    if value in {"1", "true", "yes", "y", "observed", "confirmed"}:
        return True
    if value in {"0", "false", "no", "n", "missing", "unknown", "not_mentioned", "not_applicable", ""}:
        return False
    return None


def canonicalize_concept_label(
    *,
    value: Any,
    status: Any,
    observed_mask: Any,
    source: Any = "",
    confidence: Any = "",
) -> CanonicalConceptLabel:
    clean_value = _clean(value)
    canonical_status = canonicalize_concept_status(status, clean_value)
    parsed_observed = parse_observed_mask(observed_mask)
    explicit_absent = _clean(status).lower() == "explicit_absent"
    observed = bool(parsed_observed) and canonical_status == CONFIRMED and (
        bool(clean_value) or explicit_absent
    )
    try:
        confidence_value = float(_clean(confidence) or (1.0 if observed else 0.0))
    except ValueError:
        confidence_value = 0.0
    if not observed:
        confidence_value = 0.0
    return CanonicalConceptLabel(
        value=clean_value,
        status=canonical_status,
        observed_mask=observed,
        source=_clean(source) or canonical_status,
        confidence=confidence_value,
    )


__all__ = [
    "CONFIRMED",
    "MISSING",
    "NOT_APPLICABLE",
    "RAW_UNCONFIRMED",
    "CanonicalConceptLabel",
    "canonicalize_concept_label",
    "canonicalize_concept_status",
    "parse_observed_mask",
]
