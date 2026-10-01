"""Shared checkpoint transition contract checks.

The transition reason is intentionally opaque.  A private authority binds its
SHA256 without forcing a server, username, or migration label into public code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PATH_ONLY_TRANSITION_TYPE = "PATH_ONLY_CONFIG_REBIND_V1"
LOCATOR_ONLY_TRANSITION_TYPE = "SOURCE_RUNTIME_LOCATOR_REBIND_V1"
TRANSITION_SCHEMA_VERSION = "formal_stage1_checkpoint_authority_transition_v1"
_GENERIC_REASONS = frozenset(
    {
        "VINDR_SOURCE_RUNTIME_EXACT_MIRROR_REBIND",
        "BOUNDED_REPAIR_VALIDATION_CANDIDATE",
        "MAMMO_FP32_NUMERICAL_SAFETY_SUCCESSOR",
        "AMP_AND_SAMPLER_RUNTIME_SUCCESSOR",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transition_reason_sha256(reason: Any) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("checkpoint transition reason must be non-empty opaque metadata")
    return hashlib.sha256(reason.encode("utf-8")).hexdigest()


def validate_transition_contract(
    transition: Mapping[str, Any],
    authorities: Mapping[str, Any],
    *,
    validation_only: bool,
) -> tuple[bool, bool]:
    """Validate shared transition metadata and return path/locator mode flags."""

    transition_type = transition.get("transition_type")
    path_only = transition_type == PATH_ONLY_TRANSITION_TYPE
    locator_only = transition_type == LOCATOR_ONLY_TRANSITION_TYPE
    expected_type = (
        PATH_ONLY_TRANSITION_TYPE
        if path_only
        else LOCATOR_ONLY_TRANSITION_TYPE
        if locator_only
        else "BOUNDED_REPAIR_VALIDATION_CANDIDATE"
        if validation_only
        else "APPROVED_NUMERICAL_SAFETY_SUCCESSOR"
    )
    expected = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "status": "PASS",
        "transition_type": expected_type,
        "formal_method_drift": "NOT_YET_AUTHORIZED_CANDIDATE" if validation_only else False,
        "scientific_method_changed": False,
        "training_config_changed": False,
        "checkpoint_content_changed": False,
        "checkpoint_state_changed": False,
    }
    if any(transition.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint authority transition contract fields are invalid")

    allowed_keys = transition.get("allowed_mismatch_keys")
    allowed_path = ["authorization_sha256", "runtime_code_sha256", "resolved_training_config_sha256"]
    if transition.get("formal_init_state_rebind") or (
        isinstance(allowed_keys, list)
        and "init_state_checksum" in allowed_keys
    ):
        allowed_path.append("init_state_checksum")
    expected_keys = allowed_path if path_only else ["authorization_sha256", "runtime_code_sha256"]
    if allowed_keys != expected_keys:
        raise ValueError("checkpoint authority transition mismatch allowlist is invalid")

    reason_sha = transition_reason_sha256(transition.get("transition_reason"))
    if path_only:
        expected_reason_sha = (
            transition.get("transition_reason_sha256")
            or authorities.get("checkpoint_transition_reason_sha256")
            or authorities.get("transition_reason_sha256")
        )
        if not isinstance(expected_reason_sha, str) or reason_sha != expected_reason_sha:
            raise ValueError("path-only transition reason is not bound to the private authority SHA")
    elif transition.get("transition_reason") not in _GENERIC_REASONS:
        raise ValueError("checkpoint authority transition reason is invalid")
    return path_only, locator_only


def structured_leaf_diff(parent: Any, successor: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(parent, dict) and isinstance(successor, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(parent) | set(successor)):
            child = f"{path}.{key}" if path else str(key)
            if key not in parent or key not in successor:
                changes.append({"field_path": child, "old_value": parent.get(key), "new_value": successor.get(key)})
            else:
                changes.extend(structured_leaf_diff(parent[key], successor[key], child))
        return changes
    if isinstance(parent, list) and isinstance(successor, list):
        changes: list[dict[str, Any]] = []
        for index in range(max(len(parent), len(successor))):
            child = f"{path}[{index}]"
            if index >= len(parent) or index >= len(successor):
                changes.append({"field_path": child, "old_value": parent[index] if index < len(parent) else None, "new_value": successor[index] if index < len(successor) else None})
            else:
                changes.extend(structured_leaf_diff(parent[index], successor[index], child))
        return changes
    return [] if parent == successor else [{"field_path": path, "old_value": parent, "new_value": successor}]


def validate_path_only_diff_receipt(
    transition: Mapping[str, Any], authorities: Mapping[str, Any],
) -> None:
    required = {
        "scientific_method_changed": False,
        "training_semantics_changed": False,
        "node_local_paths_changed": True,
        "formal_method_drift": False,
    }
    if any(transition.get(key) != value for key, value in required.items()):
        raise ValueError("path-only config transition metadata is invalid")
    for field in ("source_node", "target_node", "source_home", "target_home"):
        if field in transition and (not isinstance(transition.get(field), str) or not transition[field].strip()):
            raise ValueError(f"path-only config transition field {field} is invalid")
    receipt_value = transition.get("config_diff_receipt_path")
    receipt_sha = transition.get("config_diff_receipt_sha256")
    receipt_path = Path(str(receipt_value)).expanduser().resolve()
    if not receipt_path.is_file() or _sha256_file(receipt_path) != receipt_sha:
        raise ValueError("path-only config diff receipt SHA/path mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    parent_path = Path(str(receipt.get("parent_config_path"))).expanduser().resolve()
    successor_path = Path(str(receipt.get("successor_config_path"))).expanduser().resolve()
    parent_sha = transition.get("parent_resolved_training_config_sha256")
    successor_sha = transition.get("successor_resolved_training_config_sha256")
    if (
        receipt.get("schema_version") != "formal_stage1_path_only_config_rebind_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("parent_config_sha256") != parent_sha
        or receipt.get("successor_config_sha256") != successor_sha
        or successor_sha != authorities.get("resolved_config_sha256")
        or not parent_path.is_file() or _sha256_file(parent_path) != parent_sha
        or not successor_path.is_file() or _sha256_file(successor_path) != successor_sha
    ):
        raise ValueError("path-only config diff receipt bindings are invalid")
    for field in ("non_path_changed_fields", "scientific_fields_changed", "runtime_semantic_fields_changed"):
        if receipt.get(field) != []:
            raise ValueError(f"path-only config diff receipt has nonempty {field}")
    import yaml
    actual = structured_leaf_diff(
        yaml.safe_load(parent_path.read_text(encoding="utf-8")),
        yaml.safe_load(successor_path.read_text(encoding="utf-8")),
    )
    approved: list[dict[str, Any]] = []
    inferred_mapping: tuple[str, str] | None = None
    for change in actual:
        old, new = change["old_value"], change["new_value"]
        if not (isinstance(old, str) and isinstance(new, str)):
            raise ValueError(f"non-path-only config change: {change['field_path']}")
        old_parts = PurePosixPath(old.replace("\\", "/")).parts
        new_parts = PurePosixPath(new.replace("\\", "/")).parts
        if not old.startswith("/") or not new.startswith("/"):
            raise ValueError(f"non-absolute path-only config change: {change['field_path']}")
        suffix = 0
        while suffix < min(len(old_parts), len(new_parts)) and old_parts[-1 - suffix] == new_parts[-1 - suffix]:
            suffix += 1
        if suffix == 0 or suffix == len(old_parts) or suffix == len(new_parts):
            raise ValueError(f"path-only config change has no stable suffix: {change['field_path']}")
        old_prefix = "/" + "/".join(part for part in old_parts[: len(old_parts) - suffix] if part != "/") + "/"
        new_prefix = "/" + "/".join(part for part in new_parts[: len(new_parts) - suffix] if part != "/") + "/"
        pair = (old_prefix, new_prefix)
        if inferred_mapping is None:
            inferred_mapping = pair
        if pair != inferred_mapping or new[len(new_prefix):] != old[len(old_prefix):]:
            raise ValueError(f"path-only config mapping is inconsistent: {change['field_path']}")
        approved.append({**change, "classification": "NODE_LOCAL_ABSOLUTE_PATH"})
    if not approved or receipt.get("changed_fields") != approved:
        raise ValueError("path-only config diff receipt does not exactly match structured diff")
    receipt_mapping = receipt.get("path_mapping")
    if receipt_mapping is not None and receipt_mapping != {"source_prefix": inferred_mapping[0], "target_prefix": inferred_mapping[1]}:
        raise ValueError("path-only config diff receipt mapping does not match exact structured diff")


__all__ = [
    "LOCATOR_ONLY_TRANSITION_TYPE",
    "PATH_ONLY_TRANSITION_TYPE",
    "TRANSITION_SCHEMA_VERSION",
    "transition_reason_sha256",
    "structured_leaf_diff",
    "validate_path_only_diff_receipt",
    "validate_transition_contract",
]
