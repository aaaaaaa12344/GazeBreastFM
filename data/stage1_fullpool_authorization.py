"""Independent formal pretraining authorizer (Work7 hotfix 1).

The authorizer reads only immutable inputs (validated candidate registry
receipt, final Image Branch roster receipt, final bundle receipt, resolved
formal config, Work456 frozen semantic receipts) and writes a single
``formal_pretraining_authorization.json`` into a dedicated authorization root.
It never writes into the immutable final bundle root and never starts
training.

Hotfix 1 changes:

* the authorizer cross-binds every upstream authority against the exact
  authorities recorded inside the final bundle receipt (bundle-bound A vs
  authorizer-bound B is a hard failure);
* ``assert_authorization_allows_launch`` re-verifies every frozen upstream
  receipt hash recorded in the authorization document (final roster,
  candidate registry/validation receipt, semantic contract, concept
  schema/targets, prototype, soft-label V2) before any Dataset / model /
  semantic runtime / CUDA initialization may proceed.

Hotfix 2 control-plane closure:

* ``evaluate_authorization`` never treats a missing resolved-config validation
  as PASS: the production path executes the real formal config validation
  (``validate_formal_stage1_config`` / ``validate_formal_p0b_contract`` /
  ``validate_loss_policy_contract``) against the resolved config file.  It
  constructs no Dataset, model, semantic runtime, or CUDA state;
* the candidate validation receipt is exactly bound: its self-hash is
  verified and its frozen ``candidate_registry_sha256`` / ``case_registry_sha256``
  must equal the real SHA256 of the current registry files;
* the authorizer requires BOTH the Work7 control-plane final-bundle validation
  and the actual V6 Entry bundle validation (``validate_stage1_v6_bundle`` on
  the final root) to be ``PASS_FORMAL`` before ``authorized=true``;
* the launcher guard re-verifies the final bundle receipt self-hash and the
  V6 asset index file hash recorded in the bundle receipt.

Hotfix 3 (micro):

* the actual V6 Entry validation call inside ``evaluate_authorization`` is
  fail-closed: the frozen ``validate_stage1_v6_bundle`` may raise (KeyError /
  ValueError / other contract exceptions) on malformed bundles; any exception
  becomes ``authorized=false`` with ``gates.v6_entry_validation=FAIL`` and an
  exact exception-class/message blocker -- the authorizer never crashes and
  the CLI always writes a false authorization receipt;
* the launcher guard recomputes the real final bundle SHA closure from the
  on-disk files (manifest / mapping / inventory / v6 asset index) through the
  validator's public ``verify_bundle_self_integrity`` (single SHA algorithm):
  ``actual_bundle_sha256 == receipt.bundle_sha256 ==
  authorization.bound_authorities.final_bundle_sha256`` must all hold before
  any runtime initialization; receipt fields alone are never trusted.

``authorized=true`` requires every hard gate to pass exactly:

* final Image Branch roster is formal/frozen,
* final bundle validation status == PASS_FORMAL,
* actual V6 Entry bundle validation status == PASS_FORMAL,
* resolved formal config validation == PASS (executed, never assumed),
* candidate validation receipt exactly binds the current registries,
* candidate registry SHA exact match,
* final roster receipt SHA exact match,
* final bundle SHA exact match,
* resolved config SHA exact match,
* all required Work456 authority hashes exact match,
* all upstream authorities cross-bind to the final bundle receipt.

Any missing or mismatched input produces ``authorized=false`` with an exact
blocker list.  The launcher must consume this file before constructing any
Dataset, model, semantic runtime, or CUDA state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from breast_pretrain.data.stage1_fullpool_binding_schema import (
    canonical_json_bytes,
    is_sha256,
    sha256_bytes,
    sha256_file,
)
from breast_pretrain.data.checkpoint_transition_authority import (
    validate_path_only_diff_receipt,
    validate_transition_contract,
)
from breast_pretrain.data.stage1_fullpool_final_materializer import (
    V6_ASSET_INDEX_FILENAME,
    load_final_bundle_receipt,
)
from breast_pretrain.data.stage1_fullpool_final_validator import (
    validate_final_bundle,
    validate_loss_policy_contract,
    verify_bundle_self_integrity,
)
from breast_pretrain.data.stage1_v6_contract import validate_stage1_v6_bundle

AUTHORIZATION_SCHEMA_VERSION = "formal_pretraining_authorization_v1"
AUTHORIZER_VERSION = "work7_authorizer_v1"
AUTHORIZATION_FILENAME = "formal_pretraining_authorization.json"


@dataclass(frozen=True)
class AuthorizationInputs:
    candidate_registry_path: str
    candidate_registry_sha256: str
    candidate_validation_receipt_path: str
    candidate_validation_receipt_sha256: str
    final_roster_receipt_path: str
    final_roster_receipt_sha256: str
    final_bundle_root: str
    resolved_config_path: str
    resolved_config_sha256: str
    semantic_contract_path: str | None = None
    semantic_contract_sha256: str | None = None
    concept_schema_path: str | None = None
    concept_schema_sha256: str | None = None
    concept_target_receipt_path: str | None = None
    concept_target_receipt_sha256: str | None = None
    prototype_receipt_path: str | None = None
    prototype_receipt_sha256: str | None = None
    soft_label_v2_receipt_path: str | None = None
    soft_label_v2_receipt_sha256: str | None = None
    case_registry_path: str | None = None
    case_registry_sha256: str | None = None
    closure_successor_receipt_path: str | None = None
    closure_successor_receipt_sha256: str | None = None

    def as_record(self, bundle_root: str, bundle_sha256: str, roster_terminal: str) -> dict[str, Any]:
        return {
            "candidate_registry_path": self.candidate_registry_path,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "candidate_validation_receipt_path": self.candidate_validation_receipt_path,
            "candidate_validation_receipt_sha256": self.candidate_validation_receipt_sha256,
            "final_roster_receipt_path": self.final_roster_receipt_path,
            "final_roster_receipt_sha256": self.final_roster_receipt_sha256,
            "final_roster_terminal_status": roster_terminal,
            "final_bundle_root": bundle_root,
            "final_bundle_sha256": bundle_sha256,
            "resolved_config_path": self.resolved_config_path,
            "resolved_config_sha256": self.resolved_config_sha256,
            "semantic_contract_path": self.semantic_contract_path or "",
            "semantic_contract_sha256": self.semantic_contract_sha256 or "",
            "concept_schema_path": self.concept_schema_path or "",
            "concept_schema_sha256": self.concept_schema_sha256 or "",
            "concept_target_receipt_path": self.concept_target_receipt_path or "",
            "concept_target_receipt_sha256": self.concept_target_receipt_sha256 or "",
            "prototype_receipt_path": self.prototype_receipt_path or "",
            "prototype_receipt_sha256": self.prototype_receipt_sha256 or "",
            "soft_label_v2_receipt_path": self.soft_label_v2_receipt_path or "",
            "soft_label_v2_receipt_sha256": self.soft_label_v2_receipt_sha256 or "",
            "case_registry_path": self.case_registry_path or "",
            "case_registry_sha256": self.case_registry_sha256 or "",
            "closure_successor_receipt_path": self.closure_successor_receipt_path or "",
            "closure_successor_receipt_sha256": self.closure_successor_receipt_sha256 or "",
        }


def _require_sha256(value: Any, label: str, blockers: list[str]) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not is_sha256(text):
        blockers.append(f"{label} must be a lowercase SHA256.")
        return ""
    return text


def _require_file(path_value: Any, label: str, blockers: list[str]) -> str:
    path = Path(str(path_value)).expanduser().resolve() if isinstance(path_value, str) and path_value else None
    if path is None:
        blockers.append(f"{label} is required.")
        return ""
    if not path.is_file():
        blockers.append(f"{label} does not exist: {path}")
        return ""
    return str(path)


def _verify_bundle_cross_bindings(
    inputs: AuthorizationInputs,
    bundle_receipt: Mapping[str, Any],
    blockers: list[str],
    gate_results: dict[str, str],
) -> None:
    """Every upstream authority must exactly match the bundle-bound authority.

    The final bundle receipt records the exact input receipts it was
    materialized from.  The authorizer must not authorize a bundle bound to A
    while it is handed authorities B.
    """
    input_receipts = bundle_receipt.get("input_receipts")
    if not isinstance(input_receipts, Mapping):
        blockers.append("final bundle receipt must record input_receipts for cross-binding.")
        gate_results["cross_bind_input_receipts"] = "FAIL"
        return
    pairs = (
        ("candidate_registry", "candidate_registry_sha256", inputs.candidate_registry_sha256),
        ("candidate_validation_receipt", "candidate_validation_receipt_sha256", inputs.candidate_validation_receipt_sha256),
        ("final_roster_receipt", "final_roster_receipt_sha256", inputs.final_roster_receipt_sha256),
    )
    for label, receipt_key, input_value in pairs:
        bound = _text_of(input_receipts.get(receipt_key))
        declared = _text_of(input_value)
        if not bound or not declared:
            blockers.append(f"cross-bind {label}: bundle receipt or authorization input lacks {receipt_key}.")
            gate_results[f"cross_bind_{label}"] = "FAIL"
            continue
        if bound != declared:
            blockers.append(
                f"cross-bind {label}: final bundle receipt records {bound}, "
                f"authorizer input declares {declared}; bundle-bound and authorizer-bound authorities disagree."
            )
            gate_results[f"cross_bind_{label}"] = "FAIL"
        else:
            gate_results[f"cross_bind_{label}"] = "PASS"
    semantic_receipts = bundle_receipt.get("semantic_receipts")
    if not isinstance(semantic_receipts, Mapping):
        blockers.append("final bundle receipt must record semantic_receipts for cross-binding.")
        gate_results["cross_bind_semantic_receipts"] = "FAIL"
        return
    semantic_pairs = (
        ("semantic_contract", "semantic_contract_sha256", inputs.semantic_contract_sha256),
        ("concept_schema", "concept_schema_sha256", inputs.concept_schema_sha256),
        ("concept_target_receipt", "concept_target_receipt_sha256", inputs.concept_target_receipt_sha256),
        ("prototype_receipt", "prototype_receipt_sha256", inputs.prototype_receipt_sha256),
        ("soft_label_v2_receipt", "soft_label_v2_receipt_sha256", inputs.soft_label_v2_receipt_sha256),
    )
    for label, receipt_key, input_value in semantic_pairs:
        bound = _text_of(semantic_receipts.get(receipt_key))
        declared = _text_of(input_value)
        if not bound or not declared:
            blockers.append(f"cross-bind {label}: bundle receipt or authorization input lacks {receipt_key}.")
            gate_results[f"cross_bind_{label}"] = "FAIL"
            continue
        if bound != declared:
            blockers.append(
                f"cross-bind {label}: final bundle receipt records {bound}, "
                f"authorizer input declares {declared}; semantic authorities disagree."
            )
            gate_results[f"cross_bind_{label}"] = "FAIL"
        else:
            gate_results[f"cross_bind_{label}"] = "PASS"


def _verify_candidate_receipt_exact_bind(
    inputs: AuthorizationInputs,
    blockers: list[str],
    gate_results: dict[str, str],
) -> None:
    """Exact-bind the candidate validation receipt to the current registries.

    Verifies the receipt self-hash and that the receipt's frozen
    ``candidate_registry_sha256`` / ``case_registry_sha256`` equal the real
    SHA256 of the registry files handed to the authorizer.  A receipt that
    does not bind the current files is never a PASS.
    """
    if not inputs.candidate_validation_receipt_path or not inputs.case_registry_path:
        blockers.append("candidate receipt exact bind: validation receipt and case registry paths are required.")
        gate_results["candidate_receipt_exact_bind"] = "FAIL"
        return
    try:
        receipt = json.loads(Path(inputs.candidate_validation_receipt_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        blockers.append(f"candidate receipt exact bind: validation receipt is unreadable: {exc}")
        gate_results["candidate_receipt_exact_bind"] = "FAIL"
        return
    if not isinstance(receipt, dict):
        blockers.append("candidate receipt exact bind: validation receipt must be an object.")
        gate_results["candidate_receipt_exact_bind"] = "FAIL"
        return
    declared_self = _text_of(receipt.get("validation_receipt_sha256"))
    recomputed_self = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "validation_receipt_sha256"})
    )
    if not declared_self or declared_self != recomputed_self:
        blockers.append("candidate receipt exact bind: validation receipt self-hash does not match its contents.")
        gate_results["candidate_receipt_exact_bind"] = "FAIL"
        return
    for label, path_value, field in (
        ("candidate registry", inputs.candidate_registry_path, "candidate_registry_sha256"),
        ("case registry", inputs.case_registry_path, "case_registry_sha256"),
    ):
        if not path_value or not Path(path_value).expanduser().resolve().is_file():
            blockers.append(f"candidate receipt exact bind: {label} file is missing: {path_value}")
            gate_results["candidate_receipt_exact_bind"] = "FAIL"
            continue
        frozen = _text_of(receipt.get(field))
        actual = sha256_file(str(Path(path_value).expanduser().resolve()))
        if not frozen or frozen != actual:
            blockers.append(
                f"candidate receipt exact bind: receipt {field} ({frozen!r}) does not match "
                f"the current {label} file SHA256 ({actual})."
            )
            gate_results["candidate_receipt_exact_bind"] = "FAIL"
    if gate_results.get("candidate_receipt_exact_bind") != "FAIL":
        gate_results["candidate_receipt_exact_bind"] = "PASS"


def _run_resolved_config_validation(inputs: AuthorizationInputs) -> list[str]:
    """Execute the real formal config validation against the resolved config.

    Never constructs a Dataset, model, semantic runtime, or CUDA state: this is
    pure YAML/config-object validation (``validate_formal_stage1_config`` which
    includes ``validate_formal_p0b_contract``) plus the frozen loss-policy
    contract check.
    """
    issues: list[str] = []
    if not inputs.resolved_config_path or not Path(inputs.resolved_config_path).expanduser().resolve().is_file():
        issues.append("resolved formal config file is missing.")
        return issues
    from breast_pretrain.train.stage1_joint.config import load_stage1_joint_trainer_config
    from breast_pretrain.train.stage1_joint.config_validation import validate_formal_stage1_config

    config_path = Path(inputs.resolved_config_path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = load_stage1_joint_trainer_config(config_path)
    except Exception as exc:  # noqa: BLE001 - any load failure must fail closed
        issues.append(f"resolved formal config could not be loaded: {exc}")
        return issues
    issues.extend(validate_formal_stage1_config(config))
    if isinstance(payload, Mapping):
        loss_issues: list[str] = []
        validate_loss_policy_contract(payload, loss_issues)
        issues.extend(loss_issues)
    return issues


def evaluate_authorization(
    inputs: AuthorizationInputs,
    *,
    final_bundle_validation: Mapping[str, Any] | None = None,
    resolved_config_validation: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate all hard gates and return the authorization record.

    ``final_bundle_validation`` may be precomputed (for tests); otherwise the
    authorizer runs the final bundle validator itself.
    """
    if inputs.closure_successor_receipt_path or inputs.closure_successor_receipt_sha256:
        from breast_pretrain.data.stage1_fullpool_closure_authorization import evaluate_closure_successor_authorization

        return evaluate_closure_successor_authorization(
            inputs,
            authorization_schema_version=AUTHORIZATION_SCHEMA_VERSION,
            authorizer_version=AUTHORIZER_VERSION,
            utc_now=_utc_now,
            config_validator=_run_resolved_config_validation,
            final_bundle_validation=final_bundle_validation,
            resolved_config_validation=resolved_config_validation,
        )
    blockers: list[str] = []
    gate_results: dict[str, str] = {}

    registry_sha = _require_sha256(inputs.candidate_registry_sha256, "candidate_registry_sha256", blockers)
    registry_path = _require_file(inputs.candidate_registry_path, "candidate_registry_path", blockers)
    if registry_sha and registry_path:
        actual = sha256_file(registry_path)
        gate_results["candidate_registry_sha"] = "PASS" if actual == registry_sha else "FAIL"
        if actual != registry_sha:
            blockers.append("candidate registry SHA does not match its declared authority hash.")

    validation_receipt_sha = _require_sha256(
        inputs.candidate_validation_receipt_sha256, "candidate_validation_receipt_sha256", blockers
    )
    validation_receipt_path = _require_file(
        inputs.candidate_validation_receipt_path, "candidate_validation_receipt_path", blockers
    )
    if validation_receipt_sha and validation_receipt_path:
        actual = sha256_file(validation_receipt_path)
        gate_results["candidate_validation_receipt_sha"] = "PASS" if actual == validation_receipt_sha else "FAIL"
        if actual != validation_receipt_sha:
            blockers.append("candidate validation receipt SHA does not match its declared authority hash.")
    # Hotfix 2: the receipt must exactly bind the current registry files.
    _verify_candidate_receipt_exact_bind(inputs, blockers, gate_results)

    roster_sha = _require_sha256(inputs.final_roster_receipt_sha256, "final_roster_receipt_sha256", blockers)
    roster_path = _require_file(inputs.final_roster_receipt_path, "final_roster_receipt_path", blockers)
    if roster_sha and roster_path:
        actual = sha256_file(roster_path)
        gate_results["final_roster_receipt_sha"] = "PASS" if actual == roster_sha else "FAIL"
        if actual != roster_sha:
            blockers.append("final Image Branch roster receipt SHA does not match its declared authority hash.")
    roster_terminal = ""
    if roster_path:
        try:
            roster_payload = json.loads(Path(roster_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            roster_payload = {}
        if isinstance(roster_payload, dict):
            roster_terminal = str(
                roster_payload.get("terminal_status") or roster_payload.get("status") or ""
            ).strip()
        gate_results["final_roster_formal_frozen"] = "PASS" if roster_terminal in {
            "FINAL_FROZEN", "FORMAL_FROZEN", "FINAL_ROSTER_FROZEN"
        } else "FAIL"
        if roster_terminal not in {"FINAL_FROZEN", "FORMAL_FROZEN", "FINAL_ROSTER_FROZEN"}:
            blockers.append(
                f"final Image Branch roster is not formal/frozen (terminal_status={roster_terminal!r})."
            )

    bundle_root = str(Path(inputs.final_bundle_root).expanduser().resolve()) if inputs.final_bundle_root else ""
    if not bundle_root or not Path(bundle_root).is_dir():
        blockers.append("final bundle root does not exist.")
        gate_results["final_bundle_receipt_sha"] = "FAIL"
        bundle_receipt: dict[str, Any] = {}
    else:
        try:
            bundle_receipt = load_final_bundle_receipt(bundle_root)
            gate_results["final_bundle_receipt_sha"] = "PASS"
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            bundle_receipt = {}
            blockers.append(f"final bundle materialization receipt is unreadable: {exc}")
            gate_results["final_bundle_receipt_sha"] = "FAIL"

    config_sha = _require_sha256(inputs.resolved_config_sha256, "resolved_config_sha256", blockers)
    config_path = _require_file(inputs.resolved_config_path, "resolved_config_path", blockers)
    if config_sha and config_path:
        actual = sha256_file(config_path)
        gate_results["resolved_config_sha"] = "PASS" if actual == config_sha else "FAIL"
        if actual != config_sha:
            blockers.append("resolved formal config SHA does not match its declared authority hash.")

    if final_bundle_validation is not None:
        bundle_validation = dict(final_bundle_validation)
    else:
        bundle_validation = validate_final_bundle(bundle_root, resolved_config_path=inputs.resolved_config_path)
    gate_results["final_bundle_validation"] = (
        "PASS" if bundle_validation.get("status") == "PASS_FORMAL" else f"FAIL:{bundle_validation.get('status')}"
    )
    if bundle_validation.get("status") != "PASS_FORMAL":
        blockers.append(
            "final bundle validation is not PASS_FORMAL: " + " | ".join(bundle_validation.get("errors", [])[:5])
        )

    # Hotfix 2: the actual V6 Entry bundle validation must also be PASS_FORMAL.
    # Hotfix 3: the call is fail-closed.  The frozen V6 validator may raise
    # (KeyError / ValueError / other contract exceptions) on malformed
    # bundles; any exception becomes authorized=false with an exact
    # exception-class/message blocker -- the authorizer never crashes and the
    # CLI always writes a false authorization receipt.
    if bundle_root and Path(bundle_root).is_dir():
        try:
            v6_result = validate_stage1_v6_bundle(bundle_root, resolved_config_path=inputs.resolved_config_path)
        except Exception as exc:  # noqa: BLE001 - any V6 validator exception fails closed
            v6_result = {"status": "RAISED", "errors": [f"{type(exc).__name__}: {exc}"]}
        gate_results["v6_entry_validation"] = (
            "PASS" if v6_result.get("status") == "PASS_FORMAL" else f"FAIL:{v6_result.get('status')}"
        )
        if v6_result.get("status") == "RAISED":
            blockers.append("actual V6 Entry bundle validation raised " + " | ".join(v6_result.get("errors", [])))
        elif v6_result.get("status") != "PASS_FORMAL":
            blockers.append(
                "actual V6 Entry bundle validation is not PASS_FORMAL: "
                + " | ".join(v6_result.get("errors", [])[:5])
            )
    else:
        gate_results["v6_entry_validation"] = "FAIL"
        blockers.append("actual V6 Entry bundle validation cannot run: final bundle root is missing.")

    # Hotfix 2: production must never treat a missing resolved-config
    # validation as PASS -- the real validation is always executed.
    if resolved_config_validation is not None:
        config_issues = list(resolved_config_validation)
    else:
        config_issues = _run_resolved_config_validation(inputs)
    gate_results["resolved_config_validation"] = "PASS" if not config_issues else "FAIL"
    if config_issues:
        blockers.append("resolved formal config validation failed: " + " | ".join(config_issues[:5]))

    semantic_authorities: list[tuple[str, str, str, str]] = [
        ("semantic_contract", inputs.semantic_contract_path, inputs.semantic_contract_sha256, "semantic_contract"),
        ("concept_schema", inputs.concept_schema_path, inputs.concept_schema_sha256, "concept_schema"),
        ("concept_targets", inputs.concept_target_receipt_path, inputs.concept_target_receipt_sha256, "concept_target_receipt"),
        ("prototype", inputs.prototype_receipt_path, inputs.prototype_receipt_sha256, "prototype_receipt"),
        ("soft_label_v2", inputs.soft_label_v2_receipt_path, inputs.soft_label_v2_receipt_sha256, "soft_label_v2_receipt"),
    ]
    for key, path_value, hash_value, label in semantic_authorities:
        if not path_value and not hash_value:
            blockers.append(f"{label} authority is missing (required Work456 receipt).")
            gate_results[f"{key}_authority"] = "FAIL"
            continue
        resolved = _require_file(path_value, f"{label} path", blockers)
        declared_sha = _require_sha256(hash_value, f"{label} sha256", blockers)
        if resolved and declared_sha:
            actual = sha256_file(resolved)
            gate_results[f"{key}_authority"] = "PASS" if actual == declared_sha else "FAIL"
            if actual != declared_sha:
                blockers.append(f"{label} SHA does not match its declared authority hash.")

    bundle_sha256 = _text_of(bundle_receipt.get("bundle_sha256"))
    if not bundle_sha256:
        blockers.append("final bundle receipt lacks bundle_sha256.")
        gate_results["final_bundle_sha"] = "FAIL"
    else:
        gate_results["final_bundle_sha"] = "PASS"

    _verify_bundle_cross_bindings(inputs, bundle_receipt, blockers, gate_results)

    counts = bundle_receipt.get("counts") if isinstance(bundle_receipt.get("counts"), Mapping) else {}
    record: dict[str, Any] = {
        "authorization_schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorizer_version": AUTHORIZER_VERSION,
        "authorized": False,
        "generated_at_utc": _utc_now(),
        "gates": gate_results,
        "blockers": blockers,
        "bound_authorities": inputs.as_record(bundle_root, bundle_sha256, roster_terminal),
        "counts": {
            "dataset_count": counts.get("dataset_count"),
            "case_count": counts.get("case_count"),
            "image_count": counts.get("image_count"),
            "candidate_ready_rows": counts.get("candidate_ready_rows"),
            "pending_or_excluded_rows": counts.get("candidate_rows", 0) - (counts.get("candidate_ready_rows", 0) or 0),
        },
        # decision is derived from authorized below; the two can never diverge.
        "decision": "authorized=false",
    }
    record["authorized"] = not blockers
    record["decision"] = "authorized=true" if record["authorized"] else "authorized=false"
    record["authorization_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in record.items() if key != "authorization_sha256"})
    )
    return record


def write_authorization(
    authorization_root: str | Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the authorization record into a dedicated authorization root.

    Never writes into the immutable final bundle root.  Existing authorization
    files are never overwritten.
    """
    root = Path(authorization_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / AUTHORIZATION_FILENAME
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite authorization file: {destination}")
    destination.write_text(
        json.dumps(dict(record), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"authorization_path": str(destination), "authorization_sha256": sha256_file(str(destination))}


def load_authorization(authorization_path: str | Path) -> dict[str, Any]:
    path = Path(authorization_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"formal_pretraining_authorization.json does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("formal_pretraining_authorization.json must contain an object.")
    return value


def validate_authorization_schema(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("authorization_schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        errors.append("unsupported authorization_schema_version.")
    if not isinstance(record.get("authorized"), bool):
        errors.append("authorized must be boolean.")
    if record.get("authorization_sha256") != sha256_bytes(
        canonical_json_bytes({key: value for key, value in record.items() if key != "authorization_sha256"})
    ):
        errors.append("authorization_sha256 does not match record contents.")
    if not isinstance(record.get("gates"), Mapping):
        errors.append("gates must be an object.")
    if not isinstance(record.get("blockers"), list):
        errors.append("blockers must be a list.")
    return errors


# Pairs of (authority label, path key, sha key) that the launcher must
# re-verify against the on-disk files before any runtime initialization.
_AUTHORITY_FILE_BINDINGS = (
    ("candidate registry", "candidate_registry_path", "candidate_registry_sha256"),
    ("candidate validation receipt", "candidate_validation_receipt_path", "candidate_validation_receipt_sha256"),
    ("final roster receipt", "final_roster_receipt_path", "final_roster_receipt_sha256"),
    ("semantic contract", "semantic_contract_path", "semantic_contract_sha256"),
    ("concept schema", "concept_schema_path", "concept_schema_sha256"),
    ("concept target receipt", "concept_target_receipt_path", "concept_target_receipt_sha256"),
    ("prototype receipt", "prototype_receipt_path", "prototype_receipt_sha256"),
    ("soft-label V2 receipt", "soft_label_v2_receipt_path", "soft_label_v2_receipt_sha256"),
    ("case registry", "case_registry_path", "case_registry_sha256"),
)


def _validate_authorization_scope(authorities: Mapping[str, Any]) -> None:
    run_mode = os.environ.get("HSM_STAGE1_RUN_MODE", "FORMAL_PRODUCTION").strip() or "FORMAL_PRODUCTION"
    scope = authorities.get("authorization_scope")
    if run_mode == "FORMAL_PRODUCTION":
        if scope != "FORMAL_PRODUCTION" or authorities.get("production_launch_allowed") is not True:
            raise ValueError("Formal production requires an explicit production authorization scope.")
    elif run_mode == "BOUNDED_REPAIR_VALIDATION":
        if scope != "BOUNDED_REPAIR_VALIDATION_ONLY" or authorities.get("bounded_repair_validation_allowed") is not True:
            raise ValueError("Bounded repair validation requires an explicit validation-only authorization scope.")
        if authorities.get("production_launch_allowed") is not False or authorities.get("candidate_runtime") is not True:
            raise ValueError("Validation-only authorization isolation fields are invalid.")
    else:
        raise ValueError(f"Unsupported HSM_STAGE1_RUN_MODE: {run_mode!r}.")


def _validate_receipt_only_numerical_policy(authorities: Mapping[str, Any]) -> None:
    """Validate the receipt-only runtime/code safety successor contract."""
    code_path = Path(_text_of(authorities.get("final_code_receipt_path")))
    runtime_path = Path(_text_of(authorities.get("runtime_binding_receipt_path")))
    if not code_path.is_file() or not runtime_path.is_file():
        raise ValueError("receipt-only numerical safety bindings must point to existing receipts.")
    code_receipt = json.loads(code_path.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(runtime_path.read_text(encoding="utf-8"))
    if code_receipt.get("schema_version") != "formal_final_code_sha_receipt_v1" or code_receipt.get("status") != "PASS":
        raise ValueError("receipt-only final code receipt is not a valid PASS receipt.")
    if not isinstance(code_receipt.get("files"), list):
        raise ValueError("receipt-only final code receipt files must be a list.")
    if len(code_receipt["files"]) != 21:
        raise ValueError("receipt-only final code receipt must contain exactly 21 files.")
    student_entries = [
        entry for entry in code_receipt["files"]
        if isinstance(entry, Mapping)
        and entry.get("file") == "src/breast_pretrain/train/stage1_joint/student_forward.py"
    ]
    if len(student_entries) != 1:
        raise ValueError("receipt-only final code receipt must contain exactly one student_forward.py entry.")
    student_sha = _text_of(student_entries[0].get("sha256"))
    bound_sha = _text_of(authorities.get("student_forward_code_sha256"))
    if not is_sha256(student_sha) or student_sha != bound_sha:
        raise ValueError("student_forward.py SHA does not match authorization bound authority.")
    if _text_of(code_receipt.get("final_code_sha256")) != _text_of(authorities.get("final_code_sha256")):
        raise ValueError("final code receipt aggregate SHA does not match authorization.")
    if runtime_receipt.get("schema_version") != "receipt_only_runtime_binding_v2" or runtime_receipt.get("status") != "PASS":
        raise ValueError("receipt-only runtime binding receipt is not a valid v2 PASS receipt.")
    validation_only = authorities.get("authorization_scope") == "BOUNDED_REPAIR_VALIDATION_ONLY"
    expected_policy = {
        "approved_numerical_safety_successor": True,
        "formal_method_drift": "NOT_YET_AUTHORIZED_CANDIDATE" if validation_only else False,
        "precision_policy_changed": True,
        "precision_policy_change_scope": "mammo_fm_timm_efficientnet_b5_encoder_only",
        "mammo_encoder_precision_policy": "fp32_safety_island",
        "mammo_encoder_fp32_safety_required": True,
        "required_environment": {"HSM_STAGE1_MAMMO_ENCODER_FP32_SAFETY": "1"},
        "student_forward_code_sha256": student_sha,
    }
    for key, expected in expected_policy.items():
        if runtime_receipt.get(key) != expected or authorities.get(key) != expected:
            raise ValueError(f"receipt-only numerical safety policy mismatch: {key}.")


def _validate_checkpoint_authority_transition(authorities: Mapping[str, Any]) -> None:
    transition = authorities.get("checkpoint_resume_authority_transition")
    if not isinstance(transition, Mapping):
        raise ValueError("receipt-only authorization is missing checkpoint authority transition.")
    validation_only = authorities.get("authorization_scope") == "BOUNDED_REPAIR_VALIDATION_ONLY"
    path_only, locator_only = validate_transition_contract(
        transition, authorities, validation_only=validation_only,
    )
    receipt_path = Path(_text_of(transition.get("parent_checkpoint_receipt_path"))).expanduser().resolve()
    if not receipt_path.is_file() or sha256_file(str(receipt_path)) != transition.get("parent_checkpoint_receipt_sha256"):
        raise ValueError("checkpoint authority transition parent receipt SHA/path mismatch.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checksums = receipt.get("run_checksums")
    if not isinstance(checksums, Mapping):
        raise ValueError("checkpoint authority transition parent run_checksums are missing.")
    bindings = {
        "status": "PASS",
        "checkpoint_schema_version": "formal_stage1_resume_v2",
        "global_step": transition.get("parent_global_step"),
        "checkpoint_sha256": transition.get("parent_checkpoint_sha256"),
    }
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise ValueError("checkpoint authority transition parent receipt fields mismatch.")
    checkpoint_path = Path(_text_of(transition.get("parent_checkpoint_path"))).expanduser().resolve()
    if not checkpoint_path.is_file() or sha256_file(str(checkpoint_path)) != transition.get("parent_checkpoint_sha256"):
        raise ValueError("checkpoint authority transition parent checkpoint SHA/path mismatch.")
    checksum_bindings = {
        "runtime_code_sha256": "parent_runtime_code_sha256",
        "authorization_sha256": "parent_authorization_sha256",
        "resolved_training_config_sha256": "parent_resolved_training_config_sha256",
    }
    if any(checksums.get(source) != transition.get(target) for source, target in checksum_bindings.items()):
        raise ValueError("checkpoint authority transition parent run checksum mismatch.")
    if sha256_bytes(canonical_json_bytes(dict(checksums))) != transition.get("parent_run_checksums_sha256"):
        raise ValueError("checkpoint authority transition parent run_checksums SHA mismatch.")
    if path_only:
        validate_path_only_diff_receipt(transition, authorities)
    elif transition.get("parent_resolved_training_config_sha256") != authorities.get("resolved_config_sha256"):
        raise ValueError("checkpoint authority transition changes the resolved training config.")
    if transition.get("successor_runtime_code_sha256") != authorities.get("final_code_sha256"):
        raise ValueError("checkpoint authority transition successor runtime SHA mismatch.")


def assert_authorization_allows_launch(
    authorization_path: str | Path,
    *,
    final_bundle_root: str | Path | None = None,
    resolved_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authorization-first launcher guard.

    Raises before any Dataset / model / semantic runtime / CUDA initialization
    when the authorization is missing, malformed, authorized=false, or when
    any bound authority (config SHA and every frozen upstream receipt hash)
    does not exactly match the on-disk authority.  For the final bundle, the
    real SHA closure is recomputed from the on-disk files (manifest / mapping
    / inventory / v6 asset index) via ``verify_bundle_self_integrity`` and must
    equal both the receipt ``bundle_sha256`` and the authorization-bound
    ``final_bundle_sha256`` -- receipt fields alone are never trusted.
    """
    record = load_authorization(authorization_path)
    schema_errors = validate_authorization_schema(record)
    if schema_errors:
        raise ValueError("formal pretraining authorization schema validation failed: " + "; ".join(schema_errors))
    if record.get("authorized") is not True:
        blockers = record.get("blockers") or []
        raise ValueError(
            "formal pretraining is NOT authorized: "
            + ("; ".join(blockers) if blockers else "authorized=false without recorded blockers.")
        )
    authorities = record.get("bound_authorities")
    if not isinstance(authorities, Mapping):
        raise ValueError("authorization record must bind authorities.")
    if authorities.get("authorization_mode") == "RECEIPT_ONLY_AUTHORIZATION_REBIND_V1":
        _validate_authorization_scope(authorities)
        receipt_only_bindings = (
            ("closure successor receipt", "closure_successor_receipt_path", "closure_successor_receipt_sha256"),
            ("Final Entry receipt", "final_entry_receipt_path", "final_entry_receipt_sha256"),
            ("manifest reconciliation receipt", "manifest_reconciliation_receipt_path", "manifest_reconciliation_receipt_sha256"),
            ("final code receipt", "final_code_receipt_path", "final_code_receipt_sha256"),
            ("authorization request", "authorization_request_path", "authorization_request_sha256"),
            ("Graph lazy receipt", "graph_lazy_receipt_path", "graph_lazy_receipt_sha256"),
            ("P0-B receipt", "p0b_receipt_path", "p0b_receipt_sha256"),
            ("semantic-unit successor receipt", "semantic_unit_successor_receipt_path", "semantic_unit_successor_receipt_sha256"),
            ("runtime binding receipt", "runtime_binding_receipt_path", "runtime_binding_receipt_sha256"),
            ("backbone", "backbone_path", "backbone_sha256"),
        )
        for label, path_key, sha_key in receipt_only_bindings:
            bound_path = _text_of(authorities.get(path_key))
            expected_sha = _text_of(authorities.get(sha_key))
            if not bound_path or not is_sha256(expected_sha):
                raise ValueError(f"receipt-only authorization {label} binding is malformed.")
            actual_sha = sha256_file(bound_path)
            if actual_sha != expected_sha:
                raise ValueError(f"receipt-only authorization {label} SHA mismatch ({actual_sha} != {expected_sha}).")
        if _text_of(authorities.get("final_bundle_sha256")) != "dd84f12d751a9fcd857e0c2d70c53939615b5364fda737eb7edc81c3914a8506":
            raise ValueError("receipt-only authorization bundle SHA is not the frozen authority.")
        _validate_receipt_only_numerical_policy(authorities)
        _validate_checkpoint_authority_transition(authorities)
        if resolved_config_path is not None:
            config_path = Path(resolved_config_path).expanduser().resolve()
            expected_path = Path(_text_of(authorities.get("resolved_config_path"))).expanduser().resolve()
            if config_path != expected_path:
                raise ValueError(f"receipt-only authorization config path mismatch ({config_path} != {expected_path}).")
            actual_config_sha = sha256_file(str(config_path))
            expected_config_sha = _text_of(authorities.get("resolved_config_sha256"))
            if actual_config_sha != expected_config_sha:
                raise ValueError(
                    "receipt-only authorization resolved config SHA mismatch "
                    f"({actual_config_sha} != {expected_config_sha})."
                )
        return record
    if final_bundle_root is not None:
        bundle_receipt = load_final_bundle_receipt(final_bundle_root)
        expected_bundle_sha = _text_of(authorities.get("final_bundle_sha256"))
        actual_bundle_sha = _text_of(bundle_receipt.get("bundle_sha256"))
        if not expected_bundle_sha or expected_bundle_sha != actual_bundle_sha:
            raise ValueError(
                "authorization bundle SHA does not match the final bundle receipt "
                f"({expected_bundle_sha!r} != {actual_bundle_sha!r})."
            )
        # The bundle receipt must be self-consistent.
        declared_receipt_sha = _text_of(bundle_receipt.get("materialization_receipt_sha256"))
        recomputed_receipt_sha = sha256_bytes(
            canonical_json_bytes(
                {key: value for key, value in bundle_receipt.items() if key != "materialization_receipt_sha256"}
            )
        )
        if not declared_receipt_sha or declared_receipt_sha != recomputed_receipt_sha:
            raise ValueError("final bundle materialization receipt self-hash does not match its contents.")
        # Hotfix 3: never trust the receipt fields alone.  Recompute the real
        # bundle SHA closure from the on-disk files with the validator's own
        # self-integrity helper (single SHA algorithm): per-file SHA256 of
        # manifest / mapping / inventory / v6 asset index verified against the
        # receipt and the SHA manifest, then the frozen closure.  The chain
        # on-disk closure == receipt.bundle_sha256 ==
        # authorization.bound_authorities.final_bundle_sha256 must all hold
        # before any Dataset / model / semantic runtime / CUDA initialization.
        integrity = verify_bundle_self_integrity(final_bundle_root)
        if integrity["status"] != "PASS":
            raise ValueError(
                "final bundle on-disk self-integrity closure failed: " + "; ".join(integrity["errors"][:6])
            )
        on_disk_bundle_sha = integrity.get("actual_bundle_sha256")
        if not on_disk_bundle_sha or on_disk_bundle_sha != actual_bundle_sha:
            raise ValueError(
                "final bundle on-disk SHA closure does not match the bundle receipt "
                f"({on_disk_bundle_sha!r} != {actual_bundle_sha!r})."
            )
    if resolved_config_path is not None:
        config_path = Path(resolved_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise ValueError(f"resolved formal config does not exist: {config_path}")
        expected_config_sha = _text_of(authorities.get("resolved_config_sha256"))
        actual_config_sha = sha256_file(str(config_path))
        if not expected_config_sha or expected_config_sha != actual_config_sha:
            raise ValueError(
                "authorization resolved config SHA does not match the config file "
                f"({expected_config_sha!r} != {actual_config_sha!r})."
            )
    if authorities.get("authorization_mode") == "RECEIPT_ONLY_AUTHORIZATION_REBIND_V1":
        receipt_only_bindings = (
            ("closure successor receipt", "closure_successor_receipt_path", "closure_successor_receipt_sha256"),
            ("Final Entry receipt", "final_entry_receipt_path", "final_entry_receipt_sha256"),
            ("manifest reconciliation receipt", "manifest_reconciliation_receipt_path", "manifest_reconciliation_receipt_sha256"),
            ("final code receipt", "final_code_receipt_path", "final_code_receipt_sha256"),
            ("authorization request", "authorization_request_path", "authorization_request_sha256"),
            ("Graph lazy receipt", "graph_lazy_receipt_path", "graph_lazy_receipt_sha256"),
            ("P0-B receipt", "p0b_receipt_path", "p0b_receipt_sha256"),
            ("semantic-unit successor receipt", "semantic_unit_successor_receipt_path", "semantic_unit_successor_receipt_sha256"),
            ("runtime binding receipt", "runtime_binding_receipt_path", "runtime_binding_receipt_sha256"),
            ("backbone", "backbone_path", "backbone_sha256"),
        )
        for label, path_key, sha_key in receipt_only_bindings:
            bound_path = _text_of(authorities.get(path_key))
            expected_sha = _text_of(authorities.get(sha_key))
            if not bound_path or not is_sha256(expected_sha):
                raise ValueError(f"receipt-only authorization {label} binding is malformed.")
            actual_sha = sha256_file(bound_path)
            if actual_sha != expected_sha:
                raise ValueError(f"receipt-only authorization {label} SHA mismatch ({actual_sha} != {expected_sha}).")
        if _text_of(authorities.get("final_bundle_sha256")) != "dd84f12d751a9fcd857e0c2d70c53939615b5364fda737eb7edc81c3914a8506":
            raise ValueError("receipt-only authorization bundle SHA is not the frozen authority.")
        return record
    if _text_of(authorities.get("closure_successor_receipt_path")):
        from breast_pretrain.data.stage1_fullpool_closure_authorization import assert_closure_successor_authorities

        assert_closure_successor_authorities(authorities)
        return record
    # Re-verify every frozen upstream receipt hash recorded in the
    # authorization document before any runtime may be constructed.
    for label, path_key, sha_key in _AUTHORITY_FILE_BINDINGS:
        path_value = _text_of(authorities.get(path_key))
        expected_sha = _text_of(authorities.get(sha_key))
        if not path_value:
            raise ValueError(f"authorization does not bind {label} path ({path_key}).")
        if not is_sha256(expected_sha):
            raise ValueError(f"authorization {label} sha256 ({sha_key}) is malformed.")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"authorization {label} file does not exist: {path}")
        actual_sha = sha256_file(str(path))
        if actual_sha != expected_sha:
            raise ValueError(
                f"authorization {label} SHA does not match the on-disk file "
                f"({expected_sha!r} != {actual_sha!r})."
            )
    return record


def _text_of(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "AUTHORIZATION_FILENAME",
    "AUTHORIZATION_SCHEMA_VERSION",
    "AUTHORIZER_VERSION",
    "AuthorizationInputs",
    "assert_authorization_allows_launch",
    "evaluate_authorization",
    "load_authorization",
    "validate_authorization_schema",
    "write_authorization",
]
