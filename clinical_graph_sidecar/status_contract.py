from __future__ import annotations

from typing import Any

ALLOWED_STATUSES = frozenset({
    "present", "explicit_absent", "not_mentioned", "unknown", "not_applicable",
    "inferred_low_confidence", "inferred_high_confidence",
})
OBSERVED_STATUSES = frozenset({"present", "explicit_absent"})


def observed_mask_for(status: str, *, permit_inferred: bool = False) -> int:
    """Apply the V2 missing-is-not-negative contract."""
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid clinical graph status: {status!r}")
    return int(status in OBSERVED_STATUSES or (permit_inferred and status == "inferred_high_confidence"))


def normalize_evidence(item: dict[str, Any]) -> dict[str, Any]:
    status = str(item.get("status", "not_mentioned"))
    normalized = dict(item)
    normalized["status"] = status
    normalized["observed_mask"] = observed_mask_for(status)
    if normalized["observed_mask"] == 0:
        normalized["value"] = None
    return normalized
