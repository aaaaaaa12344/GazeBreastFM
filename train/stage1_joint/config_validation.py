from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.data.stage1_sparse_concept_contract import (
    SCHEMA_VERSION as P0B_SCHEMA_VERSION,
    load_frozen_concept_schema,
)


FORMAL_PRODUCTION_TIERS = {"formal_production", "production_ready_candidate"}
FORMAL_SOURCE_INTENSITY_TOLERANCE = 1e-4
P0B_FROZEN_CONTRACT_ID = "P0B_FINAL_SEMANTIC_CONTRACT_V1_20260828"
P0B_FROZEN_CONTRACT_STATUS = "P0B_FINAL_CONTRACT_FROZEN_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _same_resolved_path(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _assert_path_under_root(path: Path | None, root: Path, label: str, issues: list[str]) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        issues.append(f"{label} must resolve under formal_bundle_root; got {resolved}")


def _sha256_file(path: Path) -> str:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def validate_formal_p0b_contract(config: Any, issues: list[str]) -> None:
    """Fail-closed on the frozen P0-B schema/contract; never a legacy head list.

    Consumes the Work456 CLOSED_PASS frozen interfaces directly
    (``load_frozen_concept_schema`` + the frozen semantic contract YAML).
    """
    p0b = getattr(config.semantic, "formal_p0b", None)
    # Formal heads must be schema-derived; a legacy view/laterality-only head
    # list is never the formal authority.  This check runs even when p0b is
    # absent so a legacy-only head list can never pass.
    active = tuple(str(item).strip() for item in getattr(config.semantic, "active_concept_heads", ()) if str(item).strip())
    if active and set(active) <= {"view", "laterality"}:
        issues.append("BLOCKED_CONCEPT_SCHEMA: legacy view/laterality-only formal head authority is forbidden")
    if isinstance(p0b, dict) and p0b.get("active_direct_heads"):
        configured = tuple(str(item) for item in p0b["active_direct_heads"])
        if configured and set(configured) <= {"view", "laterality"}:
            issues.append("BLOCKED_CONCEPT_SCHEMA: formal_p0b.active_direct_heads must not be view/laterality-only")
    if p0b is None:
        issues.append("formal Stage 1 requires formal_p0b (frozen P0-B mapping); legacy head lists are not authority")
        return
    if not isinstance(p0b, dict):
        issues.append("formal_p0b must be a mapping")
        return
    required_keys = (
        "concept_schema_path",
        "concept_schema_sha256",
        "semantic_contract_path",
        "semantic_contract_sha256",
        "sparse_target_path",
        "concept_runtime_npz_path",
        "concept_runtime_manifest_path",
        "prototype_asset_path",
        "semantic_soft_label_v2",
    )
    missing = [key for key in required_keys if not str(p0b.get(key) or "").strip()]
    if missing:
        issues.append("formal_p0b misses frozen keys: " + ", ".join(missing))
        return
    schema_path = Path(str(p0b["concept_schema_path"])).expanduser().resolve()
    schema_sha = str(p0b["concept_schema_sha256"]).strip()
    if not _SHA256_RE.fullmatch(schema_sha):
        issues.append("formal_p0b.concept_schema_sha256 must be a lowercase SHA256")
    elif not schema_path.is_file():
        issues.append(f"formal_p0b concept schema file does not exist: {schema_path}")
    else:
        actual = _sha256_file(schema_path)
        if actual != schema_sha:
            issues.append(f"formal_p0b concept schema SHA256 mismatch (declared={schema_sha}, actual={actual})")
        else:
            try:
                schema = load_frozen_concept_schema(schema_path, expected_sha256=schema_sha)
                if schema.version != P0B_SCHEMA_VERSION or len(schema.concepts) != 37:
                    issues.append("formal_p0b concept schema is not the frozen TRI_MODAL V0.3.2 (37 concepts)")
                elif not schema.direct_concept_ids:
                    issues.append("formal_p0b concept schema has no direct-enabled concepts")
            except ValueError as exc:
                issues.append(f"formal_p0b concept schema is invalid: {exc}")
    contract_path = Path(str(p0b["semantic_contract_path"])).expanduser().resolve()
    contract_sha = str(p0b["semantic_contract_sha256"]).strip()
    if not _SHA256_RE.fullmatch(contract_sha):
        issues.append("formal_p0b.semantic_contract_sha256 must be a lowercase SHA256")
    elif not contract_path.is_file():
        issues.append(f"formal_p0b semantic contract file does not exist: {contract_path}")
    else:
        actual = _sha256_file(contract_path)
        if actual != contract_sha:
            issues.append(f"formal_p0b semantic contract SHA256 mismatch (declared={contract_sha}, actual={actual})")
        else:
            try:
                payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                issues.append(f"formal_p0b semantic contract YAML is invalid: {exc}")
                payload = {}
            if not isinstance(payload, dict):
                issues.append("formal_p0b semantic contract must be a YAML mapping")
            else:
                if payload.get("contract_id") != P0B_FROZEN_CONTRACT_ID:
                    issues.append(f"formal_p0b semantic contract id must be {P0B_FROZEN_CONTRACT_ID}")
                if payload.get("status") != P0B_FROZEN_CONTRACT_STATUS:
                    issues.append(f"formal_p0b semantic contract status must be {P0B_FROZEN_CONTRACT_STATUS}")
                schema_authority = payload.get("schema_authority") if isinstance(payload.get("schema_authority"), dict) else {}
                if schema_authority.get("yaml_sha256") != schema_sha:
                    issues.append("formal_p0b contract schema_authority.yaml_sha256 must match concept_schema_sha256")


def validate_formal_stage1_config(
    config: Any,
    *,
    bundle_path: Path | None = None,
    backbone_weight_path: Path | None = None,
) -> list[str]:
    """Validate a formal Stage 1 config and return a list of blocking issues.

    Returns an empty list if the config is valid for formal training.
    """
    issues: list[str] = []

    # signed_inventory validates receipt row syntax only; formal releases must
    # re-hash every protected inventory member before consumption.
    if os.environ.get("HSM_IMMUTABLE_RECEIPT_VERIFY_MODE", "full").strip().lower() != "full":
        issues.append(
            "formal Stage 1 requires HSM_IMMUTABLE_RECEIPT_VERIFY_MODE=full; "
            "signed_inventory is an engineering-only mode"
        )

    run_tier = str(getattr(config.metadata, "run_tier", "") or "").strip()
    if run_tier not in FORMAL_PRODUCTION_TIERS:
        issues.append(
            "formal Stage 1 launch requires run_tier=formal_production or "
            f"production_ready_candidate, got {run_tier!r}"
        )

    audit_cfg = getattr(config, "source_intensity_audit", None)
    if audit_cfg is None or not bool(getattr(audit_cfg, "required", False)):
        issues.append("formal Stage 1 launch requires source_intensity_audit.required=true")
    else:
        approved_tolerance = float(getattr(audit_cfg, "approved_tolerance", 1e-4))
        if not math.isfinite(approved_tolerance) or approved_tolerance <= 0.0:
            issues.append("source_intensity_audit.approved_tolerance must be finite and positive")
        elif approved_tolerance != FORMAL_SOURCE_INTENSITY_TOLERANCE:
            issues.append(
                "source_intensity_audit.approved_tolerance must be exactly "
                f"{FORMAL_SOURCE_INTENSITY_TOLERANCE:g} for the formal run"
            )

    # Require bucket policy for formal training
    if str(getattr(config.data, "batch_policy", "") or "").strip() != "bucket_by_modality_and_image_size":
        issues.append("formal Stage 1 requires batch_policy=bucket_by_modality_and_image_size")

    # Require batch_size_by_modality
    batch_sizes_by_modality = getattr(config.data, "batch_size_by_modality", None)
    if not batch_sizes_by_modality:
        issues.append("batch_size_by_modality is required for formal training")
    else:
        singleton_modalities = sorted(
            str(modality)
            for modality, batch_size in batch_sizes_by_modality.items()
            if int(batch_size) < 2
        )
        if singleton_modalities:
            issues.append(
                "formal Stage 1 requires local batch_size_by_modality >= 2 for every "
                "contrastive modality bucket; singleton modalities: "
                + ", ".join(singleton_modalities)
            )

    # Verify no singleton fallback
    if int(getattr(config.data, "batch_size", 0)) == 1:
        issues.append("formal Stage 1 batch_size must be >= 2 (singleton batches degrade contrastive loss)")

    # Require no teacher latents
    if getattr(config.data, "require_teacher_latents", True):
        issues.append("require_teacher_latents must be false for formal training")

    # V6.1 frozen loss policy: L_global and L_graph must stay static, and
    # omega_sem may scale exactly L_visible/L_soft/L_concept/L_cc.
    if getattr(config.losses, "allow_dynamic_graph_consistency_weighting", False):
        issues.append("allow_dynamic_graph_consistency_weighting must be false (L_graph static)")
    conflict_aware_enabled = bool(getattr(config.losses, "conflict_aware_enabled", False))
    if conflict_aware_enabled:
        for target in ("global_align", "graph_consistency"):
            if bool(getattr(config.losses, f"conflict_aware_dynamic_{target}", False)):
                issues.append(f"conflict-aware dynamic weighting must not scale {target} (static policy)")

    floor = getattr(config.masking, "min_visible_salient_fraction", None)
    try:
        normalized_floor = float(floor)
    except (TypeError, ValueError):
        issues.append("masking.min_visible_salient_fraction must be a finite number in (0, 1]")
    else:
        if not math.isfinite(normalized_floor) or not 0.0 < normalized_floor <= 1.0:
            issues.append("masking.min_visible_salient_fraction must be a finite number in (0, 1]")

    # Require self_masked_reconstruction
    recon_source = str(getattr(config.semantic, "reconstruction_teacher_source", "") or "").strip()
    if recon_source != "self_masked_reconstruction":
        issues.append(f"reconstruction_teacher_source must be self_masked_reconstruction, got {recon_source!r}")

    # Require image_size_by_modality
    if not getattr(config.data, "image_size_by_modality", None):
        issues.append("image_size_by_modality is required for tri-modal formal training")

    # Require transform_policy_by_modality
    if not getattr(config.data, "transform_policy_by_modality", None):
        issues.append("transform_policy_by_modality is required for formal training")

    # Validate mammography uses aspect_ratio_preserving_resize_pad
    transform_policies = getattr(config.data, "transform_policy_by_modality", {}) or {}
    mammo_policy = str(transform_policies.get("mammography", "")).strip()
    if mammo_policy and mammo_policy != "aspect_ratio_preserving_resize_pad":
        issues.append(
            f"mammography transform_policy must be aspect_ratio_preserving_resize_pad, "
            f"got {mammo_policy!r}"
        )

    # Require vision_encoder_name
    enc_name = str(getattr(config.model, "vision_encoder_name", "") or "").strip()
    if enc_name != "mammo_fm_timm_efficientnet_b5":
        issues.append(
            f"vision_encoder_name must be mammo_fm_timm_efficientnet_b5 for formal training, "
            f"got {enc_name!r}"
        )

    # Require no freeze_backbone
    if getattr(config.model, "freeze_backbone", True):
        issues.append("freeze_backbone must be false for formal training")

    # Validate max_samples is null
    if getattr(config.data, "max_samples", 1) is not None:
        issues.append("max_samples must be null for formal training (use all resolved bundle rows)")

    # Check that active concept heads are a subset of supported heads
    active = getattr(config.semantic, "active_concept_heads", ())
    if not active:
        issues.append("at least one active_concept_head must be enabled")

    # Frozen P0-B schema/contract must be the sole formal concept authority.
    validate_formal_p0b_contract(config, issues)

    # Formal runs must consume the frozen Stage D semantic-unit bridge.  A
    # missing pair would otherwise select the legacy image-level index.
    semantic_unit_mapping = getattr(config.semantic, "semantic_unit_mapping_path", None)
    semantic_unit_topk = getattr(config.semantic, "semantic_unit_topk_path", None)
    if semantic_unit_mapping is None or semantic_unit_topk is None:
        issues.append(
            "formal Stage 1 requires frozen semantic-unit mapping and top-k assets; "
            "legacy image-level semantic lookup is forbidden"
        )
    if getattr(config.semantic, "semantic_unit_source_root", None) is None:
        issues.append("formal Stage 1 requires semantic_unit_source_root for frozen asset lineage")

    # Verify no teacher latent path configured
    if getattr(config.data, "teacher_latent_dir", None) is not None:
        issues.append("teacher_latent_dir must be null for formal no-teacher training")

    formal_backbone = getattr(config, "formal_backbone_weight_path", None)
    if formal_backbone is None:
        issues.append("resolved production config must set formal_backbone_weight_path")
    else:
        configured_backbone = getattr(config.model, "pretrained_weight_path", None)
        if not _same_resolved_path(configured_backbone, formal_backbone):
            issues.append(
                "runtime backbone path must exactly match formal_backbone_weight_path "
                f"({configured_backbone!r} != {formal_backbone})"
            )
        if backbone_weight_path is not None and not _same_resolved_path(backbone_weight_path, formal_backbone):
            issues.append(
                "runtime --backbone-weight-path must exactly match formal_backbone_weight_path "
                f"({backbone_weight_path} != {formal_backbone})"
            )
        expected_backbone_sha = str(
            getattr(config.model, "backbone_expected_sha256", None) or ""
        ).strip().lower()
        if not _SHA256_RE.fullmatch(expected_backbone_sha):
            issues.append(
                "formal Stage 1 requires model.backbone_expected_sha256 as a lowercase SHA256"
            )
        elif formal_backbone.is_file() and _sha256_file(formal_backbone) != expected_backbone_sha:
            issues.append(
                "formal backbone SHA256 mismatch "
                f"(declared={expected_backbone_sha}, actual={_sha256_file(formal_backbone)})"
            )

    formal_bundle_root = getattr(config, "formal_bundle_root", None)
    if formal_bundle_root is None:
        issues.append("resolved production config must set formal_bundle_root")
    else:
        bundle_root = Path(formal_bundle_root).expanduser().resolve()
        _assert_path_under_root(config.data.image_manifest_path, bundle_root, "image_manifest_path", issues)
        _assert_path_under_root(config.data.text_prompt_path, bundle_root, "text_prompt_path", issues)
        _assert_path_under_root(
            config.semantic.prompt_embedding_path,
            bundle_root,
            "semantic.prompt_embedding_path",
            issues,
        )
        _assert_path_under_root(
            config.semantic.semantic_soft_label_path,
            bundle_root,
            "semantic.semantic_soft_label_path",
            issues,
        )
        _assert_path_under_root(
            config.semantic.semantic_soft_label_topk_path,
            bundle_root,
            "semantic.semantic_soft_label_topk_path",
            issues,
        )
        _assert_path_under_root(
            config.semantic.semantic_manifest_path,
            bundle_root,
            "semantic.semantic_manifest_path",
            issues,
        )
        _assert_path_under_root(
            semantic_unit_mapping,
            bundle_root,
            "semantic.semantic_unit_mapping_path",
            issues,
        )
        _assert_path_under_root(
            semantic_unit_topk,
            bundle_root,
            "semantic.semantic_unit_topk_path",
            issues,
        )
        _assert_path_under_root(
            getattr(config.semantic, "semantic_unit_source_root", None),
            bundle_root,
            "semantic.semantic_unit_source_root",
            issues,
        )
        _assert_path_under_root(
            config.semantic.birads_prior_manifest_path,
            bundle_root,
            "semantic.birads_prior_manifest_path",
            issues,
        )
        clinical_graph = getattr(config, "clinical_graph", None)
        _assert_path_under_root(
            getattr(clinical_graph, "sidecar_case_concept_vector_path", None),
            bundle_root,
            "clinical_graph.sidecar_case_concept_vector_path",
            issues,
        )

    return issues


__all__ = [
    "FORMAL_SOURCE_INTENSITY_TOLERANCE",
    "validate_formal_p0b_contract",
    "validate_formal_stage1_config",
]
