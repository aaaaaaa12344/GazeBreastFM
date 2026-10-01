"""Authorization checks for a Stage E closure-only successor.

This adapter is deliberately separate from the legacy candidate/roster
authorization path.  It validates a successor built from an immutable Stage E
bundle and refuses to substitute absent historical registry artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from breast_pretrain.data.stage1_fullpool_binding_schema import canonical_json_bytes, is_sha256, sha256_bytes, sha256_file
from breast_pretrain.data.stage1_fullpool_final_materializer import load_final_bundle_receipt
from breast_pretrain.data.stage1_fullpool_final_validator import validate_final_bundle
from breast_pretrain.data.stage1_v6_contract import validate_stage1_v6_bundle


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _file(path_value: Any, label: str, blockers: list[str]) -> Path | None:
    value = _text(path_value)
    path = Path(value).expanduser().resolve() if value else None
    if path is None or not path.is_file():
        blockers.append(f"closure successor {label} is missing: {value!r}.")
        return None
    return path


def _exact_file(path_value: Any, sha_value: Any, label: str, blockers: list[str], gates: dict[str, str]) -> None:
    path = _file(path_value, label, blockers)
    expected = _text(sha_value)
    gate = f"closure_{label.replace(' ', '_')}"
    if path is None or not is_sha256(expected):
        if not is_sha256(expected):
            blockers.append(f"closure successor {label} SHA256 is malformed.")
        gates[gate] = "FAIL"
        return
    actual = sha256_file(str(path))
    gates[gate] = "PASS" if actual == expected else "FAIL"
    if actual != expected:
        blockers.append(f"closure successor {label} SHA256 does not match its recorded authority.")


def _closure_bindings(receipt: Mapping[str, Any], blockers: list[str], gates: dict[str, str]) -> dict[str, dict[str, str]]:
    input_receipts = receipt.get("input_receipts")
    semantic = receipt.get("semantic_receipts")
    if not isinstance(input_receipts, Mapping):
        blockers.append("closure successor receipt lacks input_receipts.")
        gates["closure_input_receipts"] = "FAIL"
        return {}
    required_flags = {
        "closure_only_successor": True,
        "closure_successor_mode": True,
        "semantic_change": False,
        "upstream_recomputation": False,
    }
    for key, expected in required_flags.items():
        if input_receipts.get(key) != expected:
            blockers.append(f"closure successor input_receipts.{key} must be {expected!r}.")
            gates["closure_input_receipts"] = "FAIL"
    universe_rows = input_receipts.get("formal_universe_rows")
    if not isinstance(universe_rows, int) or universe_rows <= 0:
        blockers.append("closure successor formal_universe_rows must be a positive authority value.")
        gates["closure_input_receipts"] = "FAIL"
    for key in ("corrected_source_manifest_sha256", "trainer_manifest_sha256"):
        if not is_sha256(_text(input_receipts.get(key))):
            blockers.append(f"closure successor {key} must be an exact SHA256 authority.")
            gates["closure_input_receipts"] = "FAIL"
    for label, path_key, sha_key in (
        ("parent Stage E receipt", "parent_stage_e_receipt_path", "parent_stage_e_receipt_sha256"),
    ):
        _exact_file(input_receipts.get(path_key), input_receipts.get(sha_key), label, blockers, gates)
    if gates.get("closure_input_receipts") != "FAIL":
        gates["closure_input_receipts"] = "PASS"
    if not isinstance(semantic, Mapping):
        blockers.append("closure successor receipt lacks semantic_receipts.")
        gates["closure_semantic_receipts"] = "FAIL"
        return {}
    pairs = (
        ("manifest reconciliation receipt", "manifest_reconciliation_receipt_path", "manifest_reconciliation_receipt_sha256"),
        ("graph index receipt", "graph_index_receipt_path", "graph_index_receipt_sha256"),
        ("successor config", "successor_config_path", "successor_config_sha256"),
        ("successor config receipt", "successor_config_receipt_path", "successor_config_receipt_sha256"),
        ("formal code SHA receipt", "formal_code_sha_receipt_path", "formal_code_sha_receipt_sha256"),
        ("concept target receipt", "concept_target_receipt_path", "concept_target_receipt_sha256"),
        ("prototype receipt", "prototype_receipt_path", "prototype_receipt_sha256"),
        ("soft label V2 receipt", "soft_label_v2_receipt_path", "soft_label_v2_receipt_sha256"),
        ("concept schema", "concept_schema_path", "concept_schema_sha256"),
        ("semantic contract", "semantic_contract_path", "semantic_contract_sha256"),
    )
    bindings: dict[str, dict[str, str]] = {}
    for label, path_key, sha_key in pairs:
        _exact_file(semantic.get(path_key), semantic.get(sha_key), label, blockers, gates)
        bindings[label] = {"path": _text(semantic.get(path_key)), "sha256": _text(semantic.get(sha_key))}
    gates["closure_semantic_receipts"] = "PASS" if not any(value == "FAIL" for key, value in gates.items() if key.startswith("closure_")) else "FAIL"
    return bindings


def evaluate_closure_successor_authorization(
    inputs: Any,
    *,
    authorization_schema_version: str,
    authorizer_version: str,
    utc_now: Callable[[], str],
    config_validator: Callable[[Any], list[str]],
    final_bundle_validation: Mapping[str, Any] | None = None,
    resolved_config_validation: list[str] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    gates: dict[str, str] = {}
    closure_path = _file(inputs.closure_successor_receipt_path, "receipt", blockers)
    closure_sha = _text(inputs.closure_successor_receipt_sha256)
    if closure_path is None or not is_sha256(closure_sha):
        if not is_sha256(closure_sha):
            blockers.append("closure successor receipt SHA256 is malformed.")
        gates["closure_successor_receipt"] = "FAIL"
        receipt: dict[str, Any] = {}
    else:
        actual = sha256_file(str(closure_path))
        gates["closure_successor_receipt"] = "PASS" if actual == closure_sha else "FAIL"
        if actual != closure_sha:
            blockers.append("closure successor receipt SHA256 does not match its recorded authority.")
        try:
            receipt = json.loads(closure_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            receipt = {}
            blockers.append(f"closure successor receipt is unreadable: {exc}")
            gates["closure_successor_receipt"] = "FAIL"
    if receipt:
        declared_self = _text(receipt.get("materialization_receipt_sha256"))
        recomputed_self = sha256_bytes(canonical_json_bytes({k: v for k, v in receipt.items() if k != "materialization_receipt_sha256"}))
        if declared_self != recomputed_self or receipt.get("status") != "PASS_FORMAL":
            blockers.append("closure successor receipt self-hash or status is invalid.")
            gates["closure_successor_receipt"] = "FAIL"
    runtime_bindings = _closure_bindings(receipt, blockers, gates) if receipt else {}
    bundle_root = str(Path(inputs.final_bundle_root).expanduser().resolve()) if _text(inputs.final_bundle_root) else ""
    if not bundle_root or not Path(bundle_root).is_dir():
        blockers.append("closure successor final bundle root is missing.")
        gates["final_bundle_validation"] = "FAIL"
        bundle_sha = ""
    else:
        try:
            bundle_receipt = load_final_bundle_receipt(bundle_root)
            bundle_sha = _text(bundle_receipt.get("bundle_sha256"))
            if not bundle_sha:
                blockers.append("closure successor final bundle receipt lacks bundle_sha256.")
                gates["final_bundle_sha"] = "FAIL"
            else:
                gates["final_bundle_sha"] = "PASS"
        except Exception as exc:  # noqa: BLE001
            bundle_sha = ""
            blockers.append(f"closure successor final bundle receipt is unreadable: {exc}")
            gates["final_bundle_sha"] = "FAIL"
    config_path = _file(inputs.resolved_config_path, "resolved config", blockers)
    config_sha = _text(inputs.resolved_config_sha256)
    if config_path is None or not is_sha256(config_sha) or sha256_file(str(config_path)) != config_sha:
        blockers.append("closure successor resolved config SHA is not exact.")
        gates["resolved_config_sha"] = "FAIL"
    else:
        gates["resolved_config_sha"] = "PASS"
    bundle_validation = dict(final_bundle_validation) if final_bundle_validation is not None else validate_final_bundle(bundle_root, resolved_config_path=inputs.resolved_config_path)
    gates["final_bundle_validation"] = "PASS" if bundle_validation.get("status") == "PASS_FORMAL" else f"FAIL:{bundle_validation.get('status')}"
    if bundle_validation.get("status") != "PASS_FORMAL":
        blockers.append("final bundle validation is not PASS_FORMAL: " + " | ".join(bundle_validation.get("errors", [])[:5]))
    try:
        v6 = validate_stage1_v6_bundle(bundle_root, resolved_config_path=inputs.resolved_config_path)
    except Exception as exc:  # noqa: BLE001
        v6 = {"status": "RAISED", "errors": [f"{type(exc).__name__}: {exc}"]}
    gates["v6_entry_validation"] = "PASS" if v6.get("status") == "PASS_FORMAL" else f"FAIL:{v6.get('status')}"
    if v6.get("status") != "PASS_FORMAL":
        blockers.append("actual V6 Entry bundle validation is not PASS_FORMAL: " + " | ".join(v6.get("errors", [])[:5]))
    config_issues = list(resolved_config_validation) if resolved_config_validation is not None else config_validator(inputs)
    gates["resolved_config_validation"] = "PASS" if not config_issues else "FAIL"
    if config_issues:
        blockers.append("resolved formal config validation failed: " + " | ".join(config_issues[:5]))
    gates["final_roster_formal_frozen"] = "NOT_APPLICABLE_CLOSURE_SUCCESSOR"
    authorities = inputs.as_record(bundle_root, bundle_sha, "NOT_APPLICABLE_CLOSURE_SUCCESSOR")
    authorities.update({
        "authorization_mode": "stage1_e_closure_successor_v1",
        "closure_successor_receipt_path": str(closure_path) if closure_path else "",
        "closure_successor_receipt_sha256": closure_sha,
        "closure_runtime_bindings": runtime_bindings,
    })
    record = {
        "authorization_schema_version": authorization_schema_version,
        "authorizer_version": authorizer_version,
        "authorized": not blockers,
        "generated_at_utc": utc_now(),
        "gates": gates,
        "blockers": blockers,
        "bound_authorities": authorities,
        "counts": {
            "image_count": input_receipts.get("formal_universe_rows"),
            "dataset_count": None,
            "case_count": None,
            "candidate_ready_rows": None,
            "pending_or_excluded_rows": None,
        },
        "decision": "authorized=true" if not blockers else "authorized=false",
    }
    record["authorization_sha256"] = sha256_bytes(canonical_json_bytes({k: v for k, v in record.items() if k != "authorization_sha256"}))
    return record


def assert_closure_successor_authorities(authorities: Mapping[str, Any]) -> None:
    path = _text(authorities.get("closure_successor_receipt_path"))
    expected = _text(authorities.get("closure_successor_receipt_sha256"))
    if not path or not is_sha256(expected) or not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError("closure successor receipt authority is not exact on disk.")
    bindings = authorities.get("closure_runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("closure successor runtime bindings are missing.")
    for label, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"closure successor runtime binding is malformed: {label}")
        binding_path = _text(binding.get("path"))
        binding_sha = _text(binding.get("sha256"))
        if not binding_path or not is_sha256(binding_sha) or not Path(binding_path).is_file() or sha256_file(binding_path) != binding_sha:
            raise ValueError(f"closure successor runtime binding is not exact on disk: {label}")
