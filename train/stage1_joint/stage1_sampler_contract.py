"""Versioned bucket-sampler contracts for formal and historical runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any



FORMAL_BUCKETED_SAMPLER_CONTRACT_V2 = "formal_bucketed_sampler_v2"
LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1 = "legacy_historical_sampler_v1"
SUPPORTED_SAMPLER_CONTRACTS = frozenset(
    {FORMAL_BUCKETED_SAMPLER_CONTRACT_V2, LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1}
)


def resume_required_missing_keys(
    saved: Mapping[str, Any], current: Mapping[str, Any], required: Sequence[str],
    *, allow_legacy_missing_version: bool = False,
) -> list[str]:
    return [
        key
        for key in required
        if not (
            key == "sampler_contract_version"
            and key not in saved
            and allow_legacy_missing_version
            and current.get(key) == LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1
        )
        and (saved.get(key) is None or current.get(key) is None)
    ]


def resume_mismatch_keys(
    saved: Mapping[str, Any], current: Mapping[str, Any],
    *, allow_legacy_missing_version: bool = False,
) -> list[str]:
    return [
        key
        for key, value in current.items()
        if not (
            key == "sampler_contract_version"
            and key not in saved
            and allow_legacy_missing_version
            and value == LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1
        )
        and saved.get(key) != value
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_legacy_historical_sampler_authority(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    world_size: int,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require an exact existing formal authority for legacy reproduction only."""

    import yaml

    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("formal_authorization") if isinstance(raw, Mapping) else None
    auth_value = block.get("authorization_path") if isinstance(block, Mapping) else None
    expected_auth_sha = block.get("authorization_sha256") if isinstance(block, Mapping) else None
    expected_auth_sha = expected_auth_sha or os.environ.get("HSM_FORMAL_AUTHORIZATION_SHA256", "").strip()
    if not isinstance(auth_value, str) or not auth_value.strip() or not isinstance(expected_auth_sha, str):
        raise ValueError("Formal legacy sampler reproduction requires SHA-bound formal_authorization.")
    auth_path = Path(auth_value).expanduser().resolve()
    if not auth_path.is_file() or _sha256_file(auth_path) != expected_auth_sha:
        raise ValueError("Formal legacy sampler authorization path/SHA mismatch.")
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    authorities = payload.get("bound_authorities") if isinstance(payload, Mapping) else None
    record = authorities.get("historical_sampler_authority") if isinstance(authorities, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("Formal legacy sampler requires historical_sampler_authority in the existing authorization.")
    required_scope = {"HISTORICAL_REPRODUCTION_ONLY", "HISTORICAL_FORMAL_REPRODUCTION"}
    if record.get("authorization_scope") not in required_scope:
        raise ValueError("Legacy sampler authority is not explicitly historical-only.")
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError("Formal legacy sampler historical manifest is missing.")
    resolved_sha = _sha256_file(path)
    manifest_sha = _sha256_file(manifest)
    expected = {
        "sampler_contract_version": LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
        "manifest_sha256": manifest_sha,
        "resolved_config_sha256": resolved_sha,
        "world_size": int(world_size),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Formal legacy sampler historical manifest/config/world-size authority mismatch.")
    if record.get("manifest_path") and Path(str(record["manifest_path"])).expanduser().resolve() != manifest:
        raise ValueError("Formal legacy sampler historical manifest path mismatch.")
    receipt_path = record.get("checkpoint_receipt_path")
    receipt_sha = record.get("checkpoint_receipt_sha256")
    if not isinstance(receipt_path, str) or not isinstance(receipt_sha, str):
        raise ValueError("Formal legacy sampler authority must bind a checkpoint receipt SHA.")
    receipt = Path(receipt_path).expanduser().resolve()
    if not receipt.is_file() or _sha256_file(receipt) != receipt_sha:
        raise ValueError("Formal legacy sampler checkpoint receipt path/SHA mismatch.")
    if resume_checkpoint_path is not None and record.get("checkpoint_path"):
        if Path(str(record["checkpoint_path"])).expanduser().resolve() != Path(resume_checkpoint_path).expanduser().resolve():
            raise ValueError("Formal legacy sampler historical checkpoint path mismatch.")
    return dict(record)


def resolve_sampler_contract_version(raw_value: Any, *, formal: bool) -> str:
    if raw_value is None or not str(raw_value).strip():
        return FORMAL_BUCKETED_SAMPLER_CONTRACT_V2 if formal else LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1
    value = str(raw_value).strip()
    if value not in SUPPORTED_SAMPLER_CONTRACTS:
        raise ValueError(
            "sampler_contract_version must be one of "
            f"{sorted(SUPPORTED_SAMPLER_CONTRACTS)}; value omitted from error output"
        )
    return value


def validate_formal_bucket_batches(
    global_batches: Sequence[Sequence[int]],
    *,
    batch_size_by_modality: Mapping[str, int],
    world_size: int,
    describe_batch: Callable[[Sequence[int]], Mapping[str, Any]] | None = None,
) -> None:
    """Validate a global batch plan without adding, dropping, or reshaping data."""

    if world_size < 1:
        raise ValueError("Formal bucket sampler requires world_size >= 1.")
    rank_lengths = [len(global_batches[rank::world_size]) for rank in range(world_size)]
    modulo = len(global_batches) % world_size
    affected: list[dict[str, Any]] = []
    for index, batch in enumerate(global_batches):
        if not batch:
            affected.append({"global_batch_index": index, "reason": "empty_batch"})
            continue
        description = dict(describe_batch(batch) if describe_batch is not None else {})
        modality = str(description.get("modality", "unknown"))
        configured = int(batch_size_by_modality.get(modality, 0))
        size = len(batch)
        if size == 1:
            affected.append({**description, "global_batch_index": index, "actual_batch_size": size, "reason": "singleton_tail"})
        elif configured > 0 and size > configured:
            affected.append({**description, "global_batch_index": index, "configured_batch_size": configured, "actual_batch_size": size, "reason": "oversized_tail"})
        elif configured <= 0:
            affected.append({**description, "global_batch_index": index, "actual_batch_size": size, "reason": "missing_modality_batch_size"})

    if modulo != 0 or len(set(rank_lengths)) != 1 or affected:
        raise ValueError(
            "Formal bucket sampler contract failed closed before optimizer step: "
            f"global_batch_count={len(global_batches)} "
            f"world_size={world_size} modulo={modulo} "
            f"per_rank_lengths={rank_lengths} affected_bucket_tail_summary={affected[:32]}"
        )


__all__ = [
    "FORMAL_BUCKETED_SAMPLER_CONTRACT_V2",
    "LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1",
    "SUPPORTED_SAMPLER_CONTRACTS",
    "resume_mismatch_keys",
    "resume_required_missing_keys",
    "resolve_sampler_contract_version",
    "validate_legacy_historical_sampler_authority",
    "validate_formal_bucket_batches",
]
