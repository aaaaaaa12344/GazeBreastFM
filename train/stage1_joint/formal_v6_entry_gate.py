"""V6 semantic bundle gate invoked before any formal Stage 1 runtime is built."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.data.stage1_v6_contract import validate_stage1_v6_bundle
from breast_pretrain.data.stage1_fullpool_authorization import assert_authorization_allows_launch
from breast_pretrain.data_entry.stage1_v6_bridge import validate_stage1_v6_dataset_entry_v2_wrapper


def _resolved_manifest_path(raw: dict[str, Any], config_path: Path) -> Path | None:
    value = raw.get("image_manifest_path")
    if not isinstance(value, str) or not value.strip():
        return None
    root_value = raw.get("project_root", ".")
    project_root = Path(str(root_value)).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    manifest = Path(value).expanduser()
    return manifest if manifest.is_absolute() else (project_root / manifest).resolve()


def _resolved_path(value: object, config_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset Entry V2 formal config requires {field}.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _assert_dataset_entry_v2_extension(payload: dict[str, Any], config_path: Path) -> None:
    extension = payload.get("dataset_entry_v2")
    if extension in (None, False):
        return
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise ValueError("dataset_entry_v2 must be omitted or configure enabled=true.")
    wrapper = _resolved_path(extension.get("wrapper_root"), config_path, "dataset_entry_v2.wrapper_root")
    image = _resolved_path(extension.get("image_release_root"), config_path, "dataset_entry_v2.image_release_root")
    gaze = _resolved_path(extension.get("gaze_release_root"), config_path, "dataset_entry_v2.gaze_release_root")
    runtime_manifest = _resolved_path(extension.get("runtime_manifest_path"), config_path, "dataset_entry_v2.runtime_manifest_path")
    audit = _resolved_path(extension.get("audit_root"), config_path, "dataset_entry_v2.audit_root")
    if audit == wrapper or audit == image or audit == gaze:
        raise ValueError("dataset_entry_v2.audit_root must be independent of immutable releases.")
    if runtime_manifest != wrapper / "manifests" / "manifest_stage1_v6_dataset_entry_v2.csv" or not runtime_manifest.is_file():
        raise ValueError("dataset_entry_v2.runtime_manifest_path must be the immutable wrapper runtime manifest.")
    validate_stage1_v6_dataset_entry_v2_wrapper(
        wrapper_root=wrapper,
        image_release_root=image,
        gaze_release_root=gaze,
        resolved_formal_config=config_path,
        audit_root=audit,
    )


def _resolved_authorization_path(payload: dict[str, Any], config_path: Path) -> Path:
    override = os.environ.get("HSM_FORMAL_AUTHORIZATION_PATH", "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError("HSM_FORMAL_AUTHORIZATION_PATH must be an absolute path.")
        return path
    block = payload.get("formal_authorization")
    if not isinstance(block, dict) or not isinstance(block.get("authorization_path"), str) or not block["authorization_path"].strip():
        raise ValueError("Formal Stage 1 launch requires formal_authorization.authorization_path.")
    value = block["authorization_path"].strip()
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _resolved_audit_root(payload: dict[str, Any], config_path: Path, bundle_root: Path) -> Path:
    """Independent audit root for formal entry validation reports.

    Reports are never written into the frozen bundle root.  The
    dataset_entry_v2 extension already provides its own independent audit
    root (reused when enabled); otherwise ``formal_authorization.audit_root``
    is required and must be independent of the bundle root.
    """
    block = payload.get("formal_authorization")
    if not isinstance(block, dict) or not isinstance(block.get("audit_root"), str) or not block["audit_root"].strip():
        raise ValueError(
            "Formal Stage 1 launch requires formal_authorization.audit_root "
            "(dataset_entry_v2 is disabled and the frozen bundle root must stay immutable)."
        )
    value = block["audit_root"].strip()
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    root = root.resolve()
    if root == bundle_root or bundle_root in root.parents:
        raise ValueError("formal_authorization.audit_root must be independent of the frozen bundle root.")
    return root


def assert_formal_authorization_first(config_path: str | Path) -> None:
    """Authorization-first guard for the formal launcher.

    Runs before any Dataset / model / semantic runtime / CUDA initialization.
    Reads formal_pretraining_authorization.json, validates its schema, requires
    authorized==true, and re-checks the exact bundle/config SHA bindings.
    """
    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Formal config must contain a mapping: {path}")
    authorization_path = _resolved_authorization_path(payload, path)
    bundle_root_value = payload.get("formal_bundle_root")
    bundle_root = (
        Path(str(bundle_root_value)).expanduser().resolve()
        if isinstance(bundle_root_value, str) and bundle_root_value.strip()
        else None
    )
    if bundle_root is not None and not bundle_root.is_dir():
        raise ValueError(f"formal_bundle_root does not exist: {bundle_root}")
    assert_authorization_allows_launch(
        authorization_path,
        final_bundle_root=bundle_root,
        resolved_config_path=path,
    )



def assert_v6_formal_entry(config_path: str | Path) -> dict[str, Any] | None:
    """Require the V6 frozen bundle for every formal Stage 1 launch.

    A historic ``manifest_stage1_semantic.csv`` is not an alternative formal
    contract.  It can still be used by explicitly non-formal fixture/debug
    code, but it cannot cross the formal launcher boundary.
    """
    from breast_pretrain.train.stage1_joint.prelaunch_control_plane import (
        verify_prelaunch_receipt_from_environment,
    )

    prelaunch = verify_prelaunch_receipt_from_environment(config_path)
    if prelaunch is not None:
        return prelaunch
    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Formal config must contain a mapping: {path}")
    extension = payload.get("dataset_entry_v2")
    v2_enabled = isinstance(extension, dict) and extension.get("enabled") is True
    manifest = _resolved_manifest_path(payload, path)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    run_tier = str(payload.get("run_tier", metadata.get("run_tier", ""))).strip()
    contract_version = str(payload.get("entry_contract_version", "")).strip().lower()
    if run_tier != "formal_production" or contract_version != "v6":
        raise ValueError(
            "Formal Stage 1 launch requires run_tier=formal_production and "
            "entry_contract_version=v6; legacy/fixture configurations must not use this entry point."
        )
    assert_formal_authorization_first(path)
    if manifest is None or manifest.name != "manifest_stage1_v6.csv":
        raise ValueError("Formal Stage 1 launch requires image_manifest_path=.../manifest_stage1_v6.csv.")
    semantic = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
    clinical_graph = payload.get("clinical_graph") if isinstance(payload.get("clinical_graph"), dict) else {}
    required_suffixes = {
        "semantic.prompt_embedding_path": (semantic.get("prompt_embedding_path"), "stage1_prompt_embeddings.json"),
        "semantic.semantic_soft_label_topk_path": (semantic.get("semantic_soft_label_topk_path"), "stage1_semantic_soft_labels_topk.npz"),
        "semantic.semantic_manifest_path": (semantic.get("semantic_manifest_path"), "stage1_semantic_soft_label_manifest.csv"),
        "semantic.birads_prior_manifest_path": (semantic.get("birads_prior_manifest_path"), "stage1_birads_prior_manifest.csv"),
        "clinical_graph.nodes_path": (clinical_graph.get("nodes_path"), "stage1_report_derived_graph_nodes.csv"),
        "clinical_graph.edges_path": (clinical_graph.get("edges_path"), "stage1_report_derived_graph_edges.csv"),
        "clinical_graph.sidecar_case_concept_vector_path": (clinical_graph.get("sidecar_case_concept_vector_path"), "clinical_graph_v6_case_sidecar.jsonl"),
    }
    for field, (value, expected_name) in required_suffixes.items():
        if not isinstance(value, str) or Path(value).name != expected_name:
            raise ValueError(f"Formal V6 config {field} must point to {expected_name}.")
    authorization_path = _resolved_authorization_path(payload, path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("bound_authorities", {}).get("authorization_mode") == "RECEIPT_ONLY_AUTHORIZATION_REBIND_V1":
        return authorization
    bundle_root = manifest.parent
    if not bundle_root.is_dir():
        raise ValueError(
            "Formal V6 image_manifest_path must point to an existing "
            "manifest_stage1_v6.csv bundle directory; directory does not exist: "
            f"{bundle_root}"
        )
    result = validate_stage1_v6_bundle(bundle_root, path)
    # Reports are never written into the frozen bundle root.  Dataset Entry V2
    # releases are closed by their terminal receipts; the V2 extension writes
    # the combined V6/E1/E2 result into its independent audit root.  Without
    # the V2 extension the entry validation report goes to the dedicated
    # formal_authorization.audit_root, so a formal resume never mutates the
    # wrapper / v6_bundle / final bundle.
    if not v2_enabled:
        audit_root = _resolved_audit_root(payload, path, bundle_root)
        audit_root.mkdir(parents=True, exist_ok=True)
        report_path = audit_root / "stage1_entry_validation_report.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS_FORMAL":
        raise ValueError("V6 formal Stage 1 entry validation failed:\n  " + "\n  ".join(result["errors"]))
    # This executes after V6 method-contract validation and before any trainer
    # construction.  It reads real E1/E2 files; no Patch/token algorithm is
    # changed here.
    _assert_dataset_entry_v2_extension(payload, path)
    return result
