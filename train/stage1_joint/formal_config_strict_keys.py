"""Fail-closed key validation for explicitly strict formal configurations.

The typed config parser intentionally keeps development and historical files
backward compatible.  This module is the narrow production guard that rejects
unknown keys before any dataset, model, DDP, or CUDA object is constructed.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


FORMAL_PRODUCTION_TIERS = frozenset({"formal_production", "production_ready_candidate"})

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "run_tier",
        "entry_contract_version",
        "dataset_entry_release_contract_version",
        "model_role",
        "compliance_status",
        "project_root",
        "formal_bundle_root",
        "formal_backbone_weight_path",
        "image_manifest_path",
        "text_prompt_path",
        "attention_map_dir",
        "teacher_latent_dir",
        "image_size",
        "image_size_by_modality",
        "transform_policy_by_modality",
        "batch_size",
        "batch_policy",
        "batch_size_by_modality",
        "num_workers",
        "device",
        "output_dir",
        "max_samples",
        "require_attention_prior_paths",
        "require_teacher_latents",
        "shuffle",
        "gaze_membership_path",
        "gaze_membership_expected_available",
        "gaze_membership_expected_disabled",
        "sampler_contract_version",
        "strict_formal_config",
        "model",
        "masking",
        "semantic",
        "losses",
        "train",
        "eval",
        "checkpoint",
        "reproducibility",
        "references",
        "clinical_graph",
        "graph_encoder",
        "formal_p0b",
        "formal_authorization",
        "source_intensity_audit",
        "metadata",
        "teacher",
        "dataset_entry_v2",
        # Resolved formal receipts materialize the typed data block and retain
        # the source path.  These are structural fields, not open-ended data.
        "config_path",
        "data",
        # These fields are retained for older formal receipts/config templates.
        "evaluation_config",
        "formal_bundle_expected_counts",
        "formal_stage1_authorized",
        "formal_stage1_started",
        "gaze_prior",
        "global_scale_name",
        "highres_encoder_contract_status",
        "is_local_scale_native",
        "local_scale_name",
        "multiscale_status",
        "primary_scale_name",
    }
)

_FIXED_KEY_BLOCKS: dict[str, frozenset[str]] = {
    "model": frozenset(
        {
            "vision_encoder_name",
            "pretrained_model_path",
            "pretrained_weight_path",
            "backbone_expected_sha256",
            "allow_missing_pretrained_fallback",
            "freeze_backbone",
            "output_patch_dim",
            "patch_size",
            "latent_dim",
            "text_dim",
            "align_dim",
            "modality_embedding",
            "modality_vocab",
            "modality_embedding_strategy",
            "modality_embedding_dim",
            "batch_norm_policy",
            "train_batch_norm_affine",
            "local_high_conf_branch",
            "high_resolution_encoder",
            "backbone",
            "encoder_contract",
            "local_high_conf_branch_enabled",
            "local_high_conf_branch_training_ready",
            "local_high_conf_branch_loss_weight",
            "local_high_conf_branch_input_size",
            "local_high_conf_branch_effective_stride",
            "local_high_conf_branch_roi_margin",
        }
    ),
    "model.local_high_conf_branch": frozenset(
        {
            "enabled",
            "local_branch_training_ready",
            "loss_weight",
            "local_input_size",
            "effective_stride",
            "roi_margin",
        }
    ),
    "model.high_resolution_encoder": frozenset(
        {"enabled", "name", "checkpoint_path", "expected_sha256", "output_patch_dim", "patch_size"}
    ),
    "model.backbone": frozenset({"name", "path", "expected_sha256", "freeze"}),
    "model.encoder_contract": frozenset({"name", "version", "expected_sha256", "status"}),
    "masking": frozenset(
        {
            "gaze_loss_mode",
            "mask_strategy",
            "mask_ratio",
            "gaze_weight_alpha",
            "gaze_mask_sampling_alpha",
            "high_conf_mask_quota",
            "min_random_mask_fraction",
            "mask_sampling_temperature",
            "gaze_mask_eps",
            "min_visible_salient_fraction",
        }
    ),
    "semantic": frozenset(
        {
            "clinical_graph_scope",
            "text_backend",
            "prompt_embedding_path",
            "text_encoder",
            "semantic_soft_label_format",
            "semantic_soft_label_path",
            "semantic_soft_label_topk_path",
            "semantic_soft_label_symmetrization",
            "semantic_manifest_path",
            "semantic_unit_mapping_path",
            "semantic_unit_topk_path",
            "semantic_unit_source_root",
            "semantic_unit_schema_version",
            "semantic_unit_binding_version",
            "birads_prior_manifest_path",
            "high_conf_weight_alpha",
            "reconstruction_teacher_source",
            "text_runtime",
            "active_concept_heads",
            "pending_concept_heads",
            "concept_head_policy",
            "concept_head_output_dims",
            "observed_mask_policy",
            "formal_p0b",
        }
    ),
    "semantic.text_encoder": frozenset(
        {"backend", "name", "checkpoint_path", "tokenizer_path", "pooling", "normalization"}
    ),
    "semantic.text_runtime": frozenset(
        {"allow_structured_prompt_fallback", "require_effective_report", "reject_legacy_prompt_cache_without_receipt"}
    ),
    "losses": frozenset(
        {
            "reconstruction_weight",
            "global_align_weight",
            "visible_align_weight",
            "semantic_soft_weight",
            "concept_loss_weight",
            "concept_consistency_weight",
            "graph_consistency_weight",
            "conflict_aware_enabled",
            "conflict_aware_semantic_visible_coverage_target",
            "conflict_aware_reconstruction_masked_gaze_target",
            "conflict_aware_min_weight",
            "conflict_aware_max_weight",
            "conflict_aware_warmup_steps",
            "allow_dynamic_graph_consistency_weighting",
            "concept_head_weights",
            "concept_consistency_head_weights",
            "conflict_aware",
        }
    ),
    "losses.conflict_aware": frozenset(
        {
            "enabled",
            "semantic_visible_coverage_target",
            "reconstruction_masked_gaze_target",
            "min_weight",
            "max_weight",
            "warmup_steps",
            "allow_dynamic_graph_consistency_weighting",
            "graph_consistency_enabled",
        }
    ),
    "train": frozenset(
        {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "max_steps",
            "grad_clip_norm",
            "log_every_n_steps",
            "resume_from_checkpoint",
            "differential_lr",
            "use_amp",
            "scheduler",
        }
    ),
    "train.differential_lr": frozenset({"enabled", "backbone_lr", "head_lr"}),
    "train.scheduler": frozenset({"type", "warmup_ratio", "min_lr_ratio"}),
    "eval": frozenset({"enabled", "every_n_steps", "every_n_epochs", "attention_top_fraction", "eval_on_train_set", "eval_label"}),
    "checkpoint": frozenset({"save_every_n_steps", "save_every_n_epochs", "save_last", "save_best", "monitor", "mode", "resume_from"}),
    "reproducibility": frozenset({"seed", "deterministic_ablation", "reuse_initial_model", "reuse_patch_mask"}),
    "references": frozenset({"panderm_repo_path", "fgclip_repo_path", "cogaze_repo_path", "ultrasound_clip_repo_path", "import_policy"}),
    "clinical_graph": frozenset(
        {
            "version",
            "nodes_path",
            "edges_path",
            "mapping_rules_path",
            "consistency_rules_path",
            "prompt_templates_path",
            "sidecar_case_concept_vector_path",
            "sidecar_index_path",
            "use_for_structured_prompt",
            "use_for_semantic_soft_labels",
            "use_for_concept_targets",
            "use_for_concept_consistency",
            "forbid_direct_graph_node_alignment",
        }
    ),
    "graph_encoder": frozenset(
        {
            "enabled",
            "implementation",
            "mode",
            "no_runtime_external_repo_dependency",
            "alignment_graph_text_fusion_enabled",
            "text_fusion_weight",
            "fusion_target",
            "forbid_image_region_graph_node_alignment",
            "nodes_path",
            "edges_path",
            "dropout",
            "graph_tensor_path",
            "hidden_dim",
            "node_feature_dim",
            "num_layers",
            "semantic_prior_weight",
            "source_model_family",
        }
    ),
    "formal_p0b": frozenset(
        {
            "concept_schema_path",
            "concept_schema_sha256",
            "semantic_contract_path",
            "semantic_contract_sha256",
            "sparse_target_path",
            "concept_runtime_npz_path",
            "concept_runtime_manifest_path",
            "prototype_asset_path",
            "semantic_soft_label_v2",
            "allow_legacy_fallback",
            "embedding_authority_hash",
            "concept_target_authority_hash",
        }
    ),
    "formal_p0b.semantic_soft_label_v2": frozenset({"w_report", "w_graph", "w_concept"}),
    "semantic.formal_p0b": frozenset(
        {
            "concept_schema_path",
            "concept_schema_sha256",
            "semantic_contract_path",
            "semantic_contract_sha256",
            "sparse_target_path",
            "concept_runtime_npz_path",
            "concept_runtime_manifest_path",
            "prototype_asset_path",
            "semantic_soft_label_v2",
            "allow_legacy_fallback",
            "embedding_authority_hash",
            "concept_target_authority_hash",
        }
    ),
    "formal_authorization": frozenset({"authorization_path", "authorization_sha256", "audit_root"}),
    "source_intensity_audit": frozenset({"required", "approved_tolerance", "audit_report_path"}),
    "metadata": frozenset({"run_tier", "model_role", "compliance_status", "known_limitations", "allowed_claims", "forbidden_claims", "strict_formal_config"}),
    "dataset_entry_v2": frozenset({"enabled", "runtime_manifest_path", "image_release_root", "gaze_release_root", "canonical_runtime", "source_runtime_cache"}),
    "data": frozenset(
        {
            "attention_map_dir",
            "batch_policy",
            "batch_size",
            "batch_size_by_modality",
            "canonical_runtime",
            "dataset_entry_v2_enabled",
            "dataset_entry_v2_gaze_release_root",
            "dataset_entry_v2_image_release_root",
            "device",
            "gaze_membership_expected_available",
            "gaze_membership_expected_disabled",
            "gaze_membership_path",
            "image_manifest_path",
            "image_size",
            "image_size_by_modality",
            "max_samples",
            "num_workers",
            "output_dir",
            "project_root",
            "require_attention_prior_paths",
            "require_teacher_latents",
            "shuffle",
            "source_runtime_cache",
            "teacher_latent_dir",
            "text_prompt_path",
            "transform_policy_by_modality",
        }
    ),
    "ddp": frozenset({"backend", "world_size", "rank", "local_rank", "find_unused_parameters", "static_graph", "master_addr", "master_port"}),
    "runtime_lineage": frozenset({"code_sha256", "asset_sha256", "manifest_sha256", "effective_report_sha256", "embedding_sha256", "graph_sha256", "concept_target_sha256", "patch_geometry_sha256", "gaze_projection_sha256"}),
}

_DYNAMIC_MAPPING_PATHS = frozenset(
    {
        "image_size_by_modality",
        "transform_policy_by_modality",
        "batch_size_by_modality",
        "losses.concept_head_weights",
        "losses.concept_consistency_head_weights",
        "semantic.concept_head_output_dims",
        "dataset_entry_v2.canonical_runtime",
        "dataset_entry_v2.source_runtime_cache",
        "formal_bundle_expected_counts",
        "evaluation_config",
        "gaze_prior",
        "runtime_lineage",
    }
)


def _strict_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"strict formal config flag {path} must be a boolean; value omitted from error output")


def strict_formal_config_enabled(raw_config: Mapping[str, Any]) -> bool:
    metadata = raw_config.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    tier = str(raw_config.get("run_tier", metadata.get("run_tier", ""))).strip()
    explicit = raw_config.get("strict_formal_config", metadata.get("strict_formal_config", False))
    return tier in FORMAL_PRODUCTION_TIERS or _strict_bool(explicit, "strict_formal_config")


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be a mapping")
        return None
    return value


def validate_formal_config_keys(raw_config: Mapping[str, Any], *, source: str = "config") -> None:
    """Reject unknown keys only for production/explicit strict configs.

    Error messages contain key paths and structural reasons only; values are
    deliberately never rendered because config values may contain credentials.
    """

    if not isinstance(raw_config, Mapping) or not strict_formal_config_enabled(raw_config):
        return
    errors: list[str] = []
    for key in raw_config:
        if str(key) not in ALLOWED_TOP_LEVEL_KEYS:
            errors.append(f"unknown key: {key}")

    def check_block(path: str, parent: Mapping[str, Any]) -> None:
        allowed = _FIXED_KEY_BLOCKS[path]
        for key in parent:
            if str(key) not in allowed:
                errors.append(f"unknown key: {path}.{key}")

    for path, allowed in _FIXED_KEY_BLOCKS.items():
        if "." in path:
            parent_path, child_key = path.rsplit(".", 1)
            parent = raw_config
            for part in parent_path.split("."):
                value = parent.get(part) if isinstance(parent, Mapping) else None
                parent = value if isinstance(value, Mapping) else {}
            block = parent.get(child_key) if isinstance(parent, Mapping) else None
            if isinstance(block, Mapping):
                for key in block:
                    if str(key) not in allowed:
                        errors.append(f"unknown key: {path}.{key}")
            continue
        block = raw_config.get(path)
        if isinstance(block, Mapping):
            check_block(path, block)
        elif block is not None and path not in _DYNAMIC_MAPPING_PATHS:
            errors.append(f"{path} must be a mapping")

    for path in _DYNAMIC_MAPPING_PATHS:
        value: Any = raw_config
        for part in path.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is not None and not isinstance(value, Mapping):
            errors.append(f"{path} must be a mapping")
        elif isinstance(value, Mapping) and any(not str(key).strip() for key in value):
            errors.append(f"{path} contains an empty dynamic key")

    # Nested dynamic mappings live inside fixed blocks and remain intentionally
    # open-ended for modality/concept/runtime identity namespaces.
    for path in ("semantic.concept_head_output_dims", "losses.concept_head_weights", "losses.concept_consistency_head_weights", "dataset_entry_v2.canonical_runtime", "dataset_entry_v2.source_runtime_cache"):
        value: Any = raw_config
        for part in path.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if isinstance(value, Mapping) and any(not isinstance(key, str) or not key.strip() for key in value):
            errors.append(f"{path} contains an invalid dynamic key")

    if errors:
        details = "\n".join(f"  - {item}" for item in sorted(set(errors)))
        raise ValueError(f"Formal config strict-key validation failed for {source}:\n{details}")


def normalize_resolved_formal_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a serialized typed config receipt back to parser input shape.

    Resolved receipts intentionally group data fields under ``data`` and
    flatten a few typed sub-configs.  The normalizer is structural only; it
    does not resolve paths, load assets, or loosen strict-key validation.
    """

    normalized = deepcopy(dict(raw_config))
    data = normalized.get("data")
    if not isinstance(data, Mapping):
        return normalized

    # Only fields consumed by the source-shaped parser are promoted.  Runtime
    # receipt-only fields stay nested so they cannot become an accidental new
    # top-level schema surface.
    for key in (
        "attention_map_dir",
        "batch_policy",
        "batch_size",
        "batch_size_by_modality",
        "device",
        "gaze_membership_expected_available",
        "gaze_membership_expected_disabled",
        "gaze_membership_path",
        "image_manifest_path",
        "image_size",
        "image_size_by_modality",
        "max_samples",
        "num_workers",
        "output_dir",
        "project_root",
        "require_attention_prior_paths",
        "require_teacher_latents",
        "shuffle",
        "teacher_latent_dir",
        "text_prompt_path",
        "transform_policy_by_modality",
    ):
        if key in data:
            normalized.setdefault(key, data[key])

    dataset_entry = dict(normalized.get("dataset_entry_v2") or {})
    if "enabled" not in dataset_entry and "dataset_entry_v2_enabled" in data:
        dataset_entry["enabled"] = data["dataset_entry_v2_enabled"]
    if "image_release_root" not in dataset_entry and "dataset_entry_v2_image_release_root" in data:
        dataset_entry["image_release_root"] = data["dataset_entry_v2_image_release_root"]
    if "gaze_release_root" not in dataset_entry and "dataset_entry_v2_gaze_release_root" in data:
        dataset_entry["gaze_release_root"] = data["dataset_entry_v2_gaze_release_root"]
    for key in ("canonical_runtime", "source_runtime_cache"):
        if key not in dataset_entry and key in data:
            dataset_entry[key] = data[key]
    if dataset_entry:
        normalized["dataset_entry_v2"] = dataset_entry

    model = dict(normalized.get("model") or {})
    local_branch = dict(model.get("local_high_conf_branch") or {})
    aliases = {
        "enabled": "local_high_conf_branch_enabled",
        "local_branch_training_ready": "local_high_conf_branch_training_ready",
        "loss_weight": "local_high_conf_branch_loss_weight",
        "local_input_size": "local_high_conf_branch_input_size",
        "effective_stride": "local_high_conf_branch_effective_stride",
        "roi_margin": "local_high_conf_branch_roi_margin",
    }
    for nested_key, flat_key in aliases.items():
        if nested_key not in local_branch and flat_key in model:
            local_branch[nested_key] = model[flat_key]
    if local_branch:
        model["local_high_conf_branch"] = local_branch
    normalized["model"] = model

    losses = dict(normalized.get("losses") or {})
    conflict = dict(losses.get("conflict_aware") or {})
    conflict_aliases = {
        "enabled": "conflict_aware_enabled",
        "semantic_visible_coverage_target": "conflict_aware_semantic_visible_coverage_target",
        "reconstruction_masked_gaze_target": "conflict_aware_reconstruction_masked_gaze_target",
        "min_weight": "conflict_aware_min_weight",
        "max_weight": "conflict_aware_max_weight",
        "warmup_steps": "conflict_aware_warmup_steps",
        "allow_dynamic_graph_consistency_weighting": "allow_dynamic_graph_consistency_weighting",
    }
    for nested_key, flat_key in conflict_aliases.items():
        if nested_key not in conflict and flat_key in losses:
            conflict[nested_key] = losses[flat_key]
    if conflict:
        losses["conflict_aware"] = conflict
    normalized["losses"] = losses

    semantic = normalized.get("semantic")
    if isinstance(semantic, Mapping) and "formal_p0b" not in normalized:
        formal_p0b = semantic.get("formal_p0b")
        if isinstance(formal_p0b, Mapping):
            normalized["formal_p0b"] = dict(formal_p0b)

    return normalized


__all__ = [
    "ALLOWED_TOP_LEVEL_KEYS",
    "FORMAL_PRODUCTION_TIERS",
    "strict_formal_config_enabled",
    "normalize_resolved_formal_config",
    "validate_formal_config_keys",
]
