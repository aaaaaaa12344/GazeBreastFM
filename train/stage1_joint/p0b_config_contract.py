"""Formal P0-B config validation kept outside the already-large config parser."""

from __future__ import annotations

from pathlib import Path
import hashlib
import yaml
import json
from typing import Any

from breast_pretrain.data.stage1_sparse_concept_contract import load_frozen_concept_schema


RESOLVER_VERSION = "formal_p0b_authority_resolver_v1"


def resolve_formal_p0b_authority(
    p0b: dict[str, Any],
    *,
    receipt_path: Path | None = None,
) -> tuple[tuple[str, ...], dict[str, int], dict[str, Any]]:
    """Resolve heads only from the frozen P0-B schema; never use legacy defaults."""
    heads, dims = derive_formal_p0b_heads(p0b)
    schema_path = Path(str(p0b["concept_schema_path"])).expanduser().resolve()
    receipt = {
        "schema_path": str(schema_path),
        "schema_sha256": str(p0b["concept_schema_sha256"]).lower(),
        "resolved_active_concept_heads": list(heads),
        "resolver_version": RESOLVER_VERSION,
    }
    if receipt_path is not None:
        target = Path(receipt_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return heads, dims, receipt


def derive_formal_p0b_heads(p0b: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, int]]:
    required = {"concept_schema_path", "concept_schema_sha256", "semantic_contract_path", "semantic_contract_sha256", "sparse_target_path", "concept_runtime_npz_path", "concept_runtime_manifest_path", "prototype_asset_path", "semantic_soft_label_v2"}
    missing = sorted(key for key in required if not p0b.get(key))
    if missing:
        raise ValueError("BLOCKED_CONCEPT_SCHEMA: formal_p0b misses " + ", ".join(missing))
    schema = load_frozen_concept_schema(p0b["concept_schema_path"], expected_sha256=str(p0b["concept_schema_sha256"]))
    contract_path = Path(str(p0b["semantic_contract_path"]))
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_hash != str(p0b["semantic_contract_sha256"]).lower():
        raise ValueError("Formal P0-B semantic contract SHA256 mismatch.")
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid formal P0-B semantic contract: {exc}") from exc
    authority = contract.get("schema_authority", {}) if isinstance(contract, dict) else {}
    if not isinstance(contract, dict) or contract.get("contract_id") != "P0B_FINAL_SEMANTIC_CONTRACT_V1_20260828" or contract.get("status") != "P0B_FINAL_CONTRACT_FROZEN_V1":
        raise ValueError("Formal P0-B semantic contract id/status mismatch.")
    if authority.get("yaml_sha256") != schema.yaml_sha256 or authority.get("concept_count") != 37:
        raise ValueError("Formal P0-B contract/schema authority mismatch.")
    expected = schema.direct_concept_ids
    configured = tuple(str(item) for item in p0b.get("active_direct_heads", ()))
    if configured and configured != expected:
        raise ValueError("BLOCKED_CONCEPT_SCHEMA: formal active heads must equal schema direct=true order.")
    if configured and set(configured) <= {"view", "laterality"}:
        raise ValueError("BLOCKED_CONCEPT_SCHEMA: legacy view/laterality-only formal head authority is forbidden.")
    soft = p0b["semantic_soft_label_v2"]
    if not isinstance(soft, dict) or not {"w_report", "w_graph", "w_concept"}.issubset(soft):
        raise ValueError("Formal P0-B requires explicit semantic soft-label V2 weights.")
    if float(soft["w_report"]) <= 0 or float(soft["w_graph"]) < 0 or float(soft["w_concept"]) < 0:
        raise ValueError("Formal P0-B semantic soft-label V2 weights are invalid.")
    if bool(p0b.get("allow_legacy_fallback", True)):
        raise ValueError("Formal P0-B legacy fallback must be false.")
    return expected, {concept_id: (1 if schema.concepts[concept_id].target_type == "binary" else len(schema.concepts[concept_id].value_space)) for concept_id in expected}


def validate_formal_p0b_mapping(config: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, int]]:
    p0b = config.get("formal_p0b")
    if not isinstance(p0b, dict):
        raise ValueError("BLOCKED_CONCEPT_SCHEMA: formal resolved config requires formal_p0b mapping.")
    return derive_formal_p0b_heads(p0b)
