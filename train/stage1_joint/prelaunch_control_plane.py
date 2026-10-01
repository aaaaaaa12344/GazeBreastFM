"""Per-launch, rank-collapsed Formal Stage 1 control-plane validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml


SCHEMA_VERSION = "formal_stage1_prelaunch_receipt_v1"
RECEIPT_ENV = "HSM_FORMAL_PRELAUNCH_RECEIPT"
LAUNCH_ID_ENV = "HSM_FORMAL_PRELAUNCH_ID"
AUTHORIZATION_SHA_ENV = "HSM_FORMAL_AUTHORIZATION_SHA256"
AUTHORIZATION_PATH_ENV = "HSM_FORMAL_AUTHORIZATION_PATH"
RUNTIME_CODE_SHA_ENV = "HSM_FORMAL_RUNTIME_CODE_SHA256"
INIT_STATE_SHA_ENV = "HSM_FORMAL_INIT_STATE_SHA256"
MAMMO_SAFETY_ENV = "HSM_STAGE1_MAMMO_ENCODER_FP32_SAFETY"
RUN_MODE_ENV = "HSM_STAGE1_RUN_MODE"
PRELAUNCH_ROOT = Path("/tmp/hsm_formal_stage1_prelaunch")
_LAUNCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _authorization_path(config_path: Path) -> Path:
    override = os.environ.get(AUTHORIZATION_PATH_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{AUTHORIZATION_PATH_ENV} must be an absolute path.")
        return _absolute(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    block = payload.get("formal_authorization") if isinstance(payload, dict) else None
    value = block.get("authorization_path") if isinstance(block, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Formal config is missing formal_authorization.authorization_path.")
    path = Path(value).expanduser()
    return _absolute(path if path.is_absolute() else config_path.parent / path)


def _required_sha_env(name: str) -> str:
    value = os.environ.get(name, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be the exact 64-character SHA256 for this launch.")
    return value


def _receipt_path(launch_id: str) -> Path:
    if not _LAUNCH_ID.fullmatch(launch_id):
        raise ValueError(f"Invalid {LAUNCH_ID_ENV}: {launch_id!r}")
    runtime_root = os.environ.get("HSM_FORMAL_RUNTIME_ROOT", "").strip()
    root = Path(runtime_root) / "prelaunch" if runtime_root else PRELAUNCH_ROOT
    return root / launch_id / "prelaunch_receipt.json"


def _verify_self_hash(payload: dict[str, Any]) -> None:
    declared = payload.get("receipt_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if declared != actual:
        raise ValueError(f"Prelaunch receipt self-hash mismatch ({declared!r} != {actual}).")


def _validate_mammo_safety_binding(authorization: dict[str, Any]) -> dict[str, Any]:
    authorities = authorization.get("bound_authorities")
    if not isinstance(authorities, dict):
        raise ValueError("authorization does not bind numerical safety authorities.")
    run_mode = os.environ.get(RUN_MODE_ENV, "FORMAL_PRODUCTION").strip() or "FORMAL_PRODUCTION"
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
        raise ValueError(f"Unsupported {RUN_MODE_ENV}: {run_mode!r}.")
    expected = {
        "approved_numerical_safety_successor": True,
        "formal_method_drift": "NOT_YET_AUTHORIZED_CANDIDATE" if scope == "BOUNDED_REPAIR_VALIDATION_ONLY" else False,
        "precision_policy_change_scope": "mammo_fm_timm_efficientnet_b5_encoder_only",
        "mammo_encoder_precision_policy": "fp32_safety_island",
        "mammo_encoder_fp32_safety_required": True,
        "required_environment": {MAMMO_SAFETY_ENV: "1"},
    }
    for key, value in expected.items():
        if authorities.get(key) != value:
            raise ValueError(f"authorization numerical safety policy mismatch: {key}.")
    runtime_path = Path(str(authorities.get("runtime_binding_receipt_path", ""))).expanduser()
    runtime_sha = str(authorities.get("runtime_binding_receipt_sha256", ""))
    if not runtime_path.is_file() or _sha256(runtime_path) != runtime_sha:
        raise ValueError("authorization runtime binding receipt SHA/path mismatch.")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    for key, value in {"schema_version": "receipt_only_runtime_binding_v2", "status": "PASS", **expected}.items():
        if runtime.get(key) != value:
            raise ValueError(f"runtime numerical safety policy mismatch: {key}.")
    if runtime.get("student_forward_code_sha256") != authorities.get("student_forward_code_sha256"):
        raise ValueError("runtime and authorization student_forward.py SHA mismatch.")
    if os.environ.get(MAMMO_SAFETY_ENV) != "1":
        raise ValueError(f"{MAMMO_SAFETY_ENV} must be exactly '1' for formal production.")
    return authorities


def verify_prelaunch_receipt(
    config_path: str | Path,
    *,
    init_state_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a local PASS receipt against this launch and its exact bindings."""
    launch_id = os.environ.get(LAUNCH_ID_ENV, "").strip()
    expected_path = _receipt_path(launch_id)
    supplied_path = _absolute(receipt_path or os.environ.get(RECEIPT_ENV, ""))
    if supplied_path != expected_path:
        raise ValueError(f"Prelaunch receipt path is not bound to launch_id={launch_id}.")
    payload = json.loads(supplied_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prelaunch receipt must contain a JSON object.")
    _verify_self_hash(payload)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "PASS":
        raise ValueError("Prelaunch receipt is not a valid PASS receipt.")
    if payload.get("launch_id") != launch_id or payload.get("creator_rank") != 0:
        raise ValueError("Prelaunch receipt launch identity or creator rank mismatch.")

    config = _absolute(config_path)
    if payload.get("resolved_config_path") != str(config):
        raise ValueError("Prelaunch receipt config path mismatch.")
    if payload.get("resolved_config_sha256") != _sha256(config):
        raise ValueError("Prelaunch receipt config SHA mismatch.")
    authorization = _authorization_path(config)
    if payload.get("authorization_path") != str(authorization):
        raise ValueError("Prelaunch receipt authorization path mismatch.")
    if payload.get("authorization_sha256") != _required_sha_env(AUTHORIZATION_SHA_ENV):
        raise ValueError("Prelaunch receipt authorization SHA mismatch.")
    authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
    authorities = _validate_mammo_safety_binding(authorization_payload)
    if payload.get("runtime_code_sha256") != _required_sha_env(RUNTIME_CODE_SHA_ENV):
        raise ValueError("Prelaunch receipt runtime code SHA mismatch.")
    if payload.get("mammo_encoder_precision_policy") != authorities["mammo_encoder_precision_policy"]:
        raise ValueError("Prelaunch receipt Mammo precision policy mismatch.")
    if payload.get("mammo_encoder_fp32_safety_required") is not True:
        raise ValueError("Prelaunch receipt does not require Mammo FP32 safety.")
    if payload.get("mammo_encoder_fp32_safety_active") is not True or payload.get("mammo_encoder_fp32_safety_env_value") != "1":
        raise ValueError("Prelaunch receipt does not prove active Mammo FP32 safety environment.")
    if payload.get("student_forward_code_sha256") != authorities.get("student_forward_code_sha256"):
        raise ValueError("Prelaunch receipt student_forward.py SHA mismatch.")

    expected_init = _absolute(init_state_path or payload.get("init_state_path", ""))
    if payload.get("init_state_path") != str(expected_init):
        raise ValueError("Prelaunch receipt init-state path mismatch.")
    if payload.get("init_state_sha256") != _required_sha_env(INIT_STATE_SHA_ENV):
        raise ValueError("Prelaunch receipt init-state SHA mismatch.")
    if payload.get("final_model_compliance_status") not in {"pass", "pass_with_warnings"}:
        raise ValueError("Prelaunch receipt does not bind passing final-model compliance.")
    if payload.get("v6_entry_status") != "PASS":
        raise ValueError("Prelaunch receipt does not bind a passing V6 entry gate.")
    return payload


def verify_prelaunch_receipt_from_environment(config_path: str | Path) -> dict[str, Any] | None:
    receipt = os.environ.get(RECEIPT_ENV, "").strip()
    if not receipt:
        return None
    return verify_prelaunch_receipt(config_path, receipt_path=receipt)


def run_control_plane_once(
    config_path: str | Path,
    init_state_path: str | Path,
    full_validation: Callable[[Path], dict[str, Any]],
    *,
    wait_timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Run full validation on rank 0; all other ranks verify one local receipt."""
    rank = int(os.environ.get("RANK", "0"))
    launch_id = os.environ.get(LAUNCH_ID_ENV, "").strip()
    path = _receipt_path(launch_id)
    config = _absolute(config_path)
    init_state = _absolute(init_state_path)

    if rank == 0:
        if path.exists():
            raise FileExistsError(f"Per-launch receipt already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=False)
        try:
            report = full_validation(config)
            status = str(report.get("status", ""))
            if status not in {"pass", "pass_with_warnings"}:
                raise RuntimeError(f"Formal compliance status is {status!r}.")
            authorization = _authorization_path(config)
            authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
            authorities = _validate_mammo_safety_binding(authorization_payload)
            payload: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "launch_id": launch_id,
                "status": "PASS",
                "resolved_config_path": str(config),
                "resolved_config_sha256": _sha256(config),
                "authorization_path": str(authorization),
                "authorization_sha256": _sha256(authorization),
                "final_bundle_sha256": authorities.get("final_bundle_sha256"),
                "runtime_code_sha256": authorities.get("final_code_sha256"),
                "mammo_encoder_precision_policy": authorities.get("mammo_encoder_precision_policy"),
                "mammo_encoder_fp32_safety_required": True,
                "mammo_encoder_fp32_safety_active": os.environ.get(MAMMO_SAFETY_ENV) == "1",
                "mammo_encoder_fp32_safety_env_name": MAMMO_SAFETY_ENV,
                "mammo_encoder_fp32_safety_env_value": os.environ.get(MAMMO_SAFETY_ENV),
                "student_forward_code_sha256": authorities.get("student_forward_code_sha256"),
                "init_state_path": str(init_state),
                "init_state_sha256": _sha256(init_state),
                "final_model_compliance_status": status,
                "v6_entry_status": "PASS",
                "creator_rank": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "control_plane_full_validation_execution_count": 1,
            }
            for key, env_name in (
                ("authorization_sha256", AUTHORIZATION_SHA_ENV),
                ("runtime_code_sha256", RUNTIME_CODE_SHA_ENV),
                ("init_state_sha256", INIT_STATE_SHA_ENV),
            ):
                if payload[key] != _required_sha_env(env_name):
                    raise ValueError(f"Rank-0 {key} does not match {env_name}.")
            payload["receipt_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)
            os.environ[RECEIPT_ENV] = str(path)
        except Exception as exc:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "launch_id": launch_id,
                "status": "FAIL",
                "creator_rank": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            failure["receipt_sha256"] = hashlib.sha256(_canonical_bytes(failure)).hexdigest()
            path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
    else:
        os.environ[RECEIPT_ENV] = str(path)
        deadline = time.monotonic() + wait_timeout_seconds
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for rank-0 prelaunch receipt: {path}")
            time.sleep(0.25)
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed.get("status") != "PASS":
            raise RuntimeError(f"Rank-0 control-plane validation failed: {observed.get('error', 'unknown')}")

    return verify_prelaunch_receipt(config, init_state_path=init_state, receipt_path=path)
