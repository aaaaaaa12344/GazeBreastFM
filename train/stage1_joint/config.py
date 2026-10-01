from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any
import warnings

import yaml

from breast_pretrain.clinical_graph_encoder.config import GraphEncoderConfig
from breast_pretrain.data.transforms.stage1_transform_spec import (
    ImageSize,
    normalize_image_size,
)
from breast_pretrain.text.clinical_concepts import (
    SUPPORTED_STAGE1_CONCEPT_HEADS,
    concept_head_output_dim,
)
from breast_pretrain.train.stage1_joint.concept_head_policy import (
    parse_concept_head_policy,
    validate_concept_head_policy,
)
from breast_pretrain.train.stage1_joint.formal_config_strict_keys import normalize_resolved_formal_config, validate_formal_config_keys
from breast_pretrain.train.stage1_joint.stage1_sampler_contract import resolve_sampler_contract_version
from breast_pretrain.train.stage1_joint.types import (
    CheckpointConfig,
    ClinicalGraphConfig,
    DataConfig,
    EvalConfig,
    LossConfig,
    MaskingConfig,
    ModelConfig,
    ReferencePaths,
    ReproducibilityConfig,
    RunMetadataConfig,
    SemanticConfig,
    SourceIntensityAuditConfig,
    Stage1JointTrainerConfig,
    TrainConfig,
)


_SUPPORTED_CONCEPT_HEAD_SET = set(SUPPORTED_STAGE1_CONCEPT_HEADS)


def _resolve_path(raw_value: Any, base_dir: Path) -> Path:
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path(raw_value: Any, base_dir: Path) -> Path | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return _resolve_path(value, base_dir)


def _resolve_optional_model_path_string(raw_value: Any, base_dir: Path) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    if value.startswith("/"):
        return value
    return str(_resolve_path(value, base_dir))


def _as_bool(raw_value: Any, field_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value, got: {raw_value!r}")


def _as_optional_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _as_tuple_of_strings(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple)):
        return tuple(str(item).strip() for item in raw_value if str(item).strip())
    value = str(raw_value).strip()
    return (value,) if value else ()


def _as_optional_positive_int(raw_value: Any, field_name: str) -> int | None:
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided.")
    return value


def _parse_image_size(raw_value: Any, field_name: str = "image_size") -> ImageSize:
    if isinstance(raw_value, int):
        return int(raw_value)
    if isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
        return normalize_image_size(raw_value)
    raise ValueError(f"{field_name} must be a positive int or [height, width].")


def _parse_image_size_by_modality(raw_value: Any) -> dict[str, tuple[int, int]] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("image_size_by_modality must be a mapping when provided.")
    parsed: dict[str, tuple[int, int]] = {}
    for raw_modality, raw_size in raw_value.items():
        modality = str(raw_modality).strip().lower()
        if modality == "mammo":
            modality = "mammography"
        if modality == "us":
            modality = "ultrasound"
        if not modality:
            raise ValueError("image_size_by_modality contains an empty modality key.")
        parsed[modality] = normalize_image_size(_parse_image_size(raw_size, f"image_size_by_modality.{modality}"))
    return parsed


def _parse_transform_policy_by_modality(raw_value: Any) -> dict[str, str] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("transform_policy_by_modality must be a mapping when provided.")
    parsed: dict[str, str] = {}
    for raw_modality, raw_policy in raw_value.items():
        modality = str(raw_modality).strip().lower()
        if modality == "mammo":
            modality = "mammography"
        if modality == "us":
            modality = "ultrasound"
        policy = str(raw_policy).strip()
        if modality and policy:
            parsed[modality] = policy
    return parsed


def _parse_batch_size_by_modality(raw_value: Any) -> dict[str, int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("batch_size_by_modality must be a mapping when provided.")
    parsed: dict[str, int] = {}
    for raw_modality, raw_batch_size in raw_value.items():
        modality = str(raw_modality).strip().lower()
        if modality == "mammo":
            modality = "mammography"
        if modality == "us":
            modality = "ultrasound"
        batch_size = int(raw_batch_size)
        if not modality:
            raise ValueError("batch_size_by_modality contains an empty modality key.")
        if batch_size <= 0:
            raise ValueError(f"batch_size_by_modality.{modality} must be positive.")
        parsed[modality] = batch_size
    return parsed


def _as_optional_positive_float(raw_value: Any, field_name: str) -> float | None:
    if raw_value is None:
        return None
    value = float(raw_value)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive when provided.")
    return value


def _parse_semantic_soft_label_format(raw_value: Any) -> str:
    value = str(raw_value or "dense").strip().lower()
    if value not in {"dense", "sparse_topk"}:
        raise ValueError("semantic.semantic_soft_label_format must be 'dense' or 'sparse_topk'.")
    return value


def _parse_vision_encoder_name(raw_value: Any) -> str:
    value = str(raw_value or "minimal_patch_encoder").strip().lower()
    return value or "minimal_patch_encoder"


def _warn_reference_path(path: Path | None, label: str) -> None:
    if path is None:
        return
    if not path.exists():
        warnings.warn(f"{label} reference path does not exist: {path}", stacklevel=2)
        return
    if not path.is_dir():
        warnings.warn(f"{label} reference path is not a directory: {path}", stacklevel=2)


def _normalize_head_name(raw_value: Any) -> str:
    return str(raw_value).strip()


def _parse_concept_head_list(
    raw_value: Any,
    *,
    field_name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if raw_value is None:
        candidate_values = list(default)
    elif isinstance(raw_value, (list, tuple)):
        candidate_values = [_normalize_head_name(item) for item in raw_value]
    else:
        raise ValueError(f"{field_name} must be a list of concept head names.")

    result: list[str] = []
    seen: set[str] = set()
    for value in candidate_values:
        if not value:
            continue
        if value not in _SUPPORTED_CONCEPT_HEAD_SET:
            supported = ", ".join(SUPPORTED_STAGE1_CONCEPT_HEADS)
            raise ValueError(
                f"{field_name} contains unsupported concept head {value!r}. Supported: {supported}."
            )
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    if not result and field_name.endswith("active_concept_heads"):
        raise ValueError(f"{field_name} must enable at least one concept head.")
    return tuple(result)


def _default_head_weight(head_name: str, active_heads: tuple[str, ...]) -> float:
    return 1.0 if head_name in active_heads and head_name in {"view", "laterality"} else 0.0


def _parse_concept_head_weights(
    raw_value: Any,
    *,
    field_name: str,
    active_heads: tuple[str, ...],
    default_all_zero: bool,
) -> dict[str, float]:
    if raw_value is None:
        raw_mapping: dict[str, Any] = {}
    elif isinstance(raw_value, dict):
        raw_mapping = raw_value
    else:
        raise ValueError(f"{field_name} must be a mapping when provided.")

    weights: dict[str, float] = {}
    for head_name in SUPPORTED_STAGE1_CONCEPT_HEADS:
        if head_name in raw_mapping:
            weights[head_name] = float(raw_mapping[head_name])
        elif default_all_zero:
            weights[head_name] = 0.0
        else:
            weights[head_name] = _default_head_weight(head_name, active_heads)

    for raw_head_name in raw_mapping:
        head_name = _normalize_head_name(raw_head_name)
        if head_name not in _SUPPORTED_CONCEPT_HEAD_SET:
            raise ValueError(f"{field_name} contains unsupported concept head key: {raw_head_name!r}")
        if weights[head_name] < 0.0:
            raise ValueError(f"{field_name}.{head_name} must be non-negative.")

    for head_name, weight in weights.items():
        if weight > 0.0 and head_name not in active_heads:
            raise ValueError(
                f"{field_name}.{head_name} is positive but {head_name!r} is not enabled in semantic.active_concept_heads."
            )
    return weights


def _concept_head_output_dims(head_names: tuple[str, ...]) -> dict[str, int]:
    return {head_name: concept_head_output_dim(head_name) for head_name in head_names}


def _parse_differential_lr(raw_value: Any) -> Any | None:
    from breast_pretrain.train.stage1_joint.types import DifferentialLRConfig
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        return None
    return DifferentialLRConfig(
        enabled=_as_bool(raw_value.get("enabled", False), "differential_lr.enabled"),
        backbone_lr=float(raw_value.get("backbone_lr", 1.0e-5)),
        head_lr=float(raw_value.get("head_lr", 1.0e-4)),
    )


def _parse_scheduler(raw_value: Any) -> Any | None:
    from breast_pretrain.train.stage1_joint.types import SchedulerConfig
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        return None
    return SchedulerConfig(
        type=str(raw_value.get("type", "cosine")).strip() or "cosine",
        warmup_ratio=float(raw_value.get("warmup_ratio", 0.05)),
        min_lr_ratio=float(raw_value.get("min_lr_ratio", 0.01)),
    )


def _pick_resume_path(
    train_resume_from_checkpoint: Any,
    checkpoint_resume_from: Any,
    project_root: Path,
) -> tuple[Path | None, Path | None]:
    train_resume = _resolve_optional_path(train_resume_from_checkpoint, project_root)
    checkpoint_resume = _resolve_optional_path(checkpoint_resume_from, project_root)
    if train_resume is not None and checkpoint_resume is not None and train_resume != checkpoint_resume:
        raise ValueError(
            "train.resume_from_checkpoint and checkpoint.resume_from must match when both are provided."
        )
    return train_resume, checkpoint_resume or train_resume


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return raw_config


def _build_reference_paths(
    raw_config: dict[str, Any],
    project_root: Path,
) -> ReferencePaths:
    references_block = raw_config.get("references")
    if references_block is not None and not isinstance(references_block, dict):
        raise ValueError("references must be a mapping when provided.")

    references = ReferencePaths(
        panderm_repo_path=_resolve_optional_path(
            (references_block or {}).get("panderm_repo_path", raw_config.get("panderm_repo_path")),
            project_root,
        ),
        fgclip_repo_path=_resolve_optional_path(
            (references_block or {}).get("fgclip_repo_path", raw_config.get("fgclip_repo_path")),
            project_root,
        ),
        cogaze_repo_path=_resolve_optional_path(
            (references_block or {}).get("cogaze_repo_path"),
            project_root,
        ),
        ultrasound_clip_repo_path=_resolve_optional_path(
            (references_block or {}).get("ultrasound_clip_repo_path"),
            project_root,
        ),
        import_policy=str(
            (references_block or {}).get("import_policy", "read_only_reference_only")
        ).strip()
        or "read_only_reference_only",
    )
    _allowed_import_policies = {"read_only_reference_only", "no_runtime_external_repo_dependency"}
    if references.import_policy not in _allowed_import_policies:
        raise ValueError(
            f"references.import_policy must be one of {sorted(_allowed_import_policies)}, "
            f"got {references.import_policy!r}."
        )

    _warn_reference_path(references.panderm_repo_path, "PanDerm")
    _warn_reference_path(references.fgclip_repo_path, "FG-CLIP")
    _warn_reference_path(references.cogaze_repo_path, "CoGaze")
    _warn_reference_path(references.ultrasound_clip_repo_path, "Ultrasound-CLIP")
    return references


def _build_run_metadata(raw_config: dict[str, Any]) -> RunMetadataConfig:
    metadata_block = raw_config.get("metadata") if isinstance(raw_config.get("metadata"), dict) else {}
    run_tier = str(raw_config.get("run_tier", metadata_block.get("run_tier", "development"))).strip()
    model_role = str(raw_config.get("model_role", metadata_block.get("model_role", "unspecified"))).strip()
    compliance_status = str(
        raw_config.get("compliance_status", metadata_block.get("compliance_status", "not_final_model"))
    ).strip()
    return RunMetadataConfig(
        run_tier=run_tier or "development",
        model_role=model_role or "unspecified",
        compliance_status=compliance_status or "not_final_model",
        known_limitations=_as_tuple_of_strings(metadata_block.get("known_limitations")),
        allowed_claims=_as_tuple_of_strings(metadata_block.get("allowed_claims")),
        forbidden_claims=_as_tuple_of_strings(metadata_block.get("forbidden_claims")),
    )


def _parse_conflict_aware_loss(loss_block: dict[str, Any]) -> dict[str, Any]:
    conflict_block = loss_block.get("conflict_aware")
    if conflict_block is None:
        conflict_block = {}
    if not isinstance(conflict_block, dict):
        raise ValueError("losses.conflict_aware must be a mapping when provided.")

    min_weight = float(conflict_block.get("min_weight", 0.5))
    max_weight = float(conflict_block.get("max_weight", 2.0))
    if min_weight <= 0.0 or max_weight <= 0.0:
        raise ValueError("losses.conflict_aware min_weight/max_weight must be positive.")
    if min_weight > max_weight:
        raise ValueError("losses.conflict_aware.min_weight cannot exceed max_weight.")
    warmup_steps = int(conflict_block.get("warmup_steps", 0))
    if warmup_steps < 0:
        raise ValueError("losses.conflict_aware.warmup_steps must be non-negative.")
    return {
        "conflict_aware_enabled": _as_bool(
            conflict_block.get("enabled", False),
            "losses.conflict_aware.enabled",
        ),
        "conflict_aware_semantic_visible_coverage_target": float(
            conflict_block.get("semantic_visible_coverage_target", 0.25)
        ),
        "conflict_aware_reconstruction_masked_gaze_target": float(
            conflict_block.get("reconstruction_masked_gaze_target", 0.25)
        ),
        "conflict_aware_min_weight": min_weight,
        "conflict_aware_max_weight": max_weight,
        "conflict_aware_warmup_steps": warmup_steps,
        "allow_dynamic_graph_consistency_weighting": _as_bool(
            conflict_block.get(
                "allow_dynamic_graph_consistency_weighting",
                conflict_block.get("graph_consistency_enabled", False),
            ),
            "losses.conflict_aware.allow_dynamic_graph_consistency_weighting",
        ),
    }


def _parse_clinical_graph(
    raw_config: dict[str, Any],
    project_root: Path,
) -> ClinicalGraphConfig | None:
    cg_block = raw_config.get("clinical_graph")
    if cg_block is None:
        return None
    if not isinstance(cg_block, dict):
        return None

    version = str(cg_block.get("version", "tri_modal_clinical_graph_v1")).strip()
    nodes_path = _resolve_optional_path(cg_block.get("nodes_path"), project_root)
    edges_path = _resolve_optional_path(cg_block.get("edges_path"), project_root)
    mapping_rules_path = _resolve_optional_path(cg_block.get("mapping_rules_path"), project_root)
    consistency_rules_path = _resolve_optional_path(cg_block.get("consistency_rules_path"), project_root)
    prompt_templates_path = _resolve_optional_path(cg_block.get("prompt_templates_path"), project_root)
    sidecar_path = _resolve_optional_path(cg_block.get("sidecar_case_concept_vector_path"), project_root)
    index_path = _resolve_optional_path(cg_block.get("sidecar_index_path"), project_root)

    return ClinicalGraphConfig(
        version=version,
        nodes_path=nodes_path,
        edges_path=edges_path,
        mapping_rules_path=mapping_rules_path,
        consistency_rules_path=consistency_rules_path,
        prompt_templates_path=prompt_templates_path,
        sidecar_case_concept_vector_path=sidecar_path,
        sidecar_index_path=index_path,
        use_for_structured_prompt=_as_bool(
            cg_block.get("use_for_structured_prompt", False), "clinical_graph.use_for_structured_prompt"
        ),
        use_for_semantic_soft_labels=_as_bool(
            cg_block.get("use_for_semantic_soft_labels", False), "clinical_graph.use_for_semantic_soft_labels"
        ),
        use_for_concept_targets=_as_bool(
            cg_block.get("use_for_concept_targets", False), "clinical_graph.use_for_concept_targets"
        ),
        use_for_concept_consistency=_as_bool(
            cg_block.get("use_for_concept_consistency", False), "clinical_graph.use_for_concept_consistency"
        ),
        forbid_direct_graph_node_alignment=_as_bool(
            cg_block.get("forbid_direct_graph_node_alignment", True),
            "clinical_graph.forbid_direct_graph_node_alignment",
        ),
    )


def _parse_graph_encoder(
    raw_config: dict[str, Any],
    project_root: Path,
    clinical_graph: ClinicalGraphConfig | None,
) -> GraphEncoderConfig | None:
    return GraphEncoderConfig.from_mapping(
        raw_config.get("graph_encoder") if isinstance(raw_config.get("graph_encoder"), dict) else None,
        project_root=project_root,
        clinical_graph_nodes_path=clinical_graph.nodes_path if clinical_graph is not None else None,
        clinical_graph_edges_path=clinical_graph.edges_path if clinical_graph is not None else None,
    )


def _parse_source_intensity_audit(
    raw_config: dict[str, Any],
    project_root: Path,
) -> SourceIntensityAuditConfig | None:
    block = raw_config.get("source_intensity_audit")
    if block is None:
        return None
    if not isinstance(block, dict):
        return None
    required = _as_bool(block.get("required", False), "source_intensity_audit.required")
    approved_tolerance = float(block.get("approved_tolerance", 1e-4))
    audit_report_path = _resolve_optional_path(block.get("audit_report_path"), project_root)
    return SourceIntensityAuditConfig(
        required=required,
        approved_tolerance=approved_tolerance,
        audit_report_path=audit_report_path,
    )


def _load_formal_config(
    raw_config: dict[str, Any],
    config_path: Path,
) -> Stage1JointTrainerConfig:
    project_root = _resolve_path(raw_config["project_root"], config_path.parent)
    model_block = raw_config.get("model") or {}
    dataset_entry_release_contract_version = str(
        raw_config.get("dataset_entry_release_contract_version", "")
    ).strip().lower()
    image_size_by_modality = _parse_image_size_by_modality(
        raw_config.get("image_size_by_modality")
    )
    if dataset_entry_release_contract_version == "v2":
        if "image_size" in raw_config:
            raise ValueError(
                "Dataset Entry V2 formal config must not configure image_size; "
                "it is derived read-only from image_size_by_modality.mammography."
            )
        if image_size_by_modality is None:
            raise ValueError("Dataset Entry V2 formal config requires image_size_by_modality.")
        from breast_pretrain.data_entry.geometry_contract import resolve_geometry_contract

        geometry_input = {
            "image_size_by_modality": image_size_by_modality,
            "patch_size": model_block.get("patch_size"),
        }
        resolved_geometry = resolve_geometry_contract(geometry_input)
        image_size = resolved_geometry.legacy_image_size
    else:
        image_size = _parse_image_size(raw_config["image_size"])
    local_branch_block = (
        model_block.get("local_high_conf_branch")
        if isinstance(model_block.get("local_high_conf_branch"), dict)
        else {}
    )
    masking_block = raw_config.get("masking") or {}
    semantic_block = raw_config.get("semantic") or {}
    loss_block = raw_config.get("losses") or {}
    train_block = raw_config.get("train") or {}
    eval_block = raw_config.get("eval") or {}
    checkpoint_block = raw_config.get("checkpoint") or {}
    reproducibility_block = raw_config.get("reproducibility") or {}
    run_metadata = _build_run_metadata(raw_config)
    is_formal_tier = run_metadata.run_tier in {"formal_production", "production_ready_candidate"}
    sampler_contract_version = resolve_sampler_contract_version(raw_config.get("sampler_contract_version"), formal=is_formal_tier)
    if is_formal_tier and "min_visible_salient_fraction" not in masking_block:
        raise ValueError(
            "Formal Stage 1 config must explicitly set masking.min_visible_salient_fraction."
        )
    min_visible_salient_fraction = float(masking_block.get("min_visible_salient_fraction", 0.0))
    if not math.isfinite(min_visible_salient_fraction):
        raise ValueError("masking.min_visible_salient_fraction must be finite.")
    if is_formal_tier and not 0.0 < min_visible_salient_fraction <= 1.0:
        raise ValueError(
            "Formal Stage 1 masking.min_visible_salient_fraction must be in (0, 1]."
        )
    clinical_graph_config = _parse_clinical_graph(raw_config, project_root)
    graph_encoder_config = _parse_graph_encoder(raw_config, project_root, clinical_graph_config)

    formal_p0b = (
        dict(raw_config.get("formal_p0b") or {})
        if isinstance(raw_config.get("formal_p0b"), dict)
        else None
    )
    if formal_p0b is not None:
        for field in ("concept_schema_path", "semantic_contract_path"):
            if formal_p0b.get(field):
                formal_p0b[field] = str(_resolve_path(formal_p0b[field], project_root))
    p0b_output_dims: dict[str, int] | None = None
    if is_formal_tier and formal_p0b is not None:
        if formal_p0b is None:
            raise ValueError("Formal Stage 1 requires formal_p0b; legacy concept heads are not authority.")
        # The frozen P0-B schema, not a legacy image-level list, is the sole
        # authority for formal heads and their output dimensions.
        from breast_pretrain.train.stage1_joint.p0b_config_contract import derive_formal_p0b_heads

        active_heads, p0b_output_dims = derive_formal_p0b_heads(formal_p0b)
        configured_heads = tuple(
            _normalize_head_name(item)
            for item in semantic_block.get("active_concept_heads", ())
            if _normalize_head_name(item)
        )
        if configured_heads and configured_heads != active_heads:
            raise ValueError(
                "Formal P0-B active concept heads disagree with the frozen schema-derived order."
            )
    else:
        active_heads = _parse_concept_head_list(
            semantic_block.get("active_concept_heads"),
            field_name="semantic.active_concept_heads",
            default=("view", "laterality"),
        )
    pending_heads = _parse_concept_head_list(
        semantic_block.get("pending_concept_heads"),
        field_name="semantic.pending_concept_heads",
        default=("density", "finding", "birads"),
    )
    concept_head_policy = parse_concept_head_policy(semantic_block.get("concept_head_policy"))
    train_resume, checkpoint_resume = _pick_resume_path(
        train_block.get("resume_from_checkpoint"),
        checkpoint_block.get("resume_from"),
        project_root,
    )
    if p0b_output_dims is not None:
        # Formal P0-B direct concepts are schema-derived rather than members of
        # the legacy clinical-head registry.  Empty YAML mappings therefore
        # mean unit weight for every frozen direct concept, never silent loss
        # disablement or a legacy head fallback.
        concept_head_weights = {head_name: 1.0 for head_name in active_heads}
        concept_consistency_head_weights = {head_name: 0.0 for head_name in active_heads}
    else:
        concept_head_weights = _parse_concept_head_weights(
            loss_block.get("concept_head_weights"),
            field_name="losses.concept_head_weights",
            active_heads=active_heads,
            default_all_zero=False,
        )
        concept_consistency_head_weights = _parse_concept_head_weights(
            loss_block.get("concept_consistency_head_weights"),
            field_name="losses.concept_consistency_head_weights",
            active_heads=active_heads,
            default_all_zero=True,
        )
    validate_concept_head_policy(
        policy=concept_head_policy,
        active_heads=active_heads,
        pending_heads=pending_heads,
        concept_head_weights=concept_head_weights,
        concept_consistency_head_weights=concept_consistency_head_weights,
        run_tier=str(raw_config.get("run_tier", "")),
    )

    text_backend = str(semantic_block.get("text_backend", "")).strip().lower()
    if text_backend != "prompt_embedding_cache":
        raise ValueError("Phase 1 formal trainer only supports semantic.text_backend=prompt_embedding_cache.")

    semantic_soft_label_format = _parse_semantic_soft_label_format(
        semantic_block.get("semantic_soft_label_format")
    )
    semantic_soft_label_topk_path = _resolve_optional_path(
        semantic_block.get("semantic_soft_label_topk_path"),
        project_root,
    )
    if semantic_soft_label_format == "sparse_topk" and semantic_soft_label_topk_path is None:
        raise ValueError(
            "semantic.semantic_soft_label_topk_path is required when semantic_soft_label_format=sparse_topk."
        )

    return Stage1JointTrainerConfig(
        config_path=config_path,
        metadata=run_metadata,
        data=DataConfig(
            project_root=project_root,
            image_manifest_path=_resolve_path(
                (raw_config.get("dataset_entry_v2") or {}).get("runtime_manifest_path")
                if isinstance(raw_config.get("dataset_entry_v2"), dict)
                and (raw_config.get("dataset_entry_v2") or {}).get("enabled") is True
                and (raw_config.get("dataset_entry_v2") or {}).get("runtime_manifest_path")
                else raw_config["image_manifest_path"],
                project_root,
            ),
            attention_map_dir=_resolve_optional_path(raw_config.get("attention_map_dir"), project_root),
            teacher_latent_dir=_resolve_optional_path(raw_config.get("teacher_latent_dir"), project_root),
            text_prompt_path=_resolve_optional_path(raw_config.get("text_prompt_path"), project_root),
            image_size=image_size,
            batch_size=int(raw_config["batch_size"]),
            num_workers=int(raw_config["num_workers"]),
            device=str(raw_config["device"]).strip(),
            output_dir=_resolve_path(raw_config["output_dir"], project_root),
            max_samples=_as_optional_positive_int(raw_config.get("max_samples"), "max_samples"),
            require_attention_prior_paths=_as_bool(
                raw_config.get("require_attention_prior_paths", False),
                "require_attention_prior_paths",
            ),
            require_teacher_latents=_as_bool(
                raw_config.get("require_teacher_latents", False),
                "require_teacher_latents",
            ),
            image_size_by_modality=image_size_by_modality,
            transform_policy_by_modality=_parse_transform_policy_by_modality(
                raw_config.get("transform_policy_by_modality")
            ),
            batch_policy=str(raw_config.get("batch_policy", "fixed_batch_size")).strip().lower()
            or "fixed_batch_size",
            batch_size_by_modality=_parse_batch_size_by_modality(
                raw_config.get("batch_size_by_modality")
            ),
            sampler_contract_version=sampler_contract_version,
            shuffle=_as_bool(raw_config.get("shuffle", False), "shuffle"),
            dataset_entry_v2_enabled=_as_bool(
                (raw_config.get("dataset_entry_v2") or {}).get("enabled", False),
                "dataset_entry_v2.enabled",
            ) if isinstance(raw_config.get("dataset_entry_v2"), dict) else False,
            dataset_entry_v2_image_release_root=_resolve_optional_path(
                (raw_config.get("dataset_entry_v2") or {}).get("image_release_root"), project_root
            ) if isinstance(raw_config.get("dataset_entry_v2"), dict) else None,
            dataset_entry_v2_gaze_release_root=_resolve_optional_path(
                (raw_config.get("dataset_entry_v2") or {}).get("gaze_release_root"), project_root
            ) if isinstance(raw_config.get("dataset_entry_v2"), dict) else None,
            canonical_runtime=dict((raw_config.get("dataset_entry_v2") or {}).get("canonical_runtime") or {})
            if isinstance(raw_config.get("dataset_entry_v2"), dict) else None,
            source_runtime_cache=dict((raw_config.get("dataset_entry_v2") or {}).get("source_runtime_cache") or {})
            if isinstance(raw_config.get("dataset_entry_v2"), dict) else None,
            gaze_membership_path=_resolve_optional_path(
                raw_config.get("gaze_membership_path"), project_root
            ),
            gaze_membership_expected_available=_as_optional_positive_int(
                raw_config.get("gaze_membership_expected_available"),
                "gaze_membership_expected_available",
            ),
            gaze_membership_expected_disabled=_as_optional_positive_int(
                raw_config.get("gaze_membership_expected_disabled"),
                "gaze_membership_expected_disabled",
            ),
        ),
        references=_build_reference_paths(raw_config, project_root),
        model=ModelConfig(
            patch_size=int(model_block["patch_size"]),
            latent_dim=int(model_block["latent_dim"]),
            text_dim=int(model_block.get("text_dim", 0)),
            align_dim=int(model_block["align_dim"]),
            vision_encoder_name=_parse_vision_encoder_name(
                model_block.get("vision_encoder_name")
            ),
            pretrained_model_path=_as_optional_string(
                model_block.get("pretrained_model_path")
            ),
            pretrained_weight_path=_resolve_optional_model_path_string(
                model_block.get("pretrained_weight_path", model_block.get("pretrained_model_path")),
                project_root,
            ),
            backbone_expected_sha256=_as_optional_string(
                model_block.get("backbone_expected_sha256")
            ),
            allow_missing_pretrained_fallback=_as_bool(
                model_block.get("allow_missing_pretrained_fallback", False),
                "model.allow_missing_pretrained_fallback",
            ),
            freeze_backbone=_as_bool(
                model_block.get("freeze_backbone", False),
                "model.freeze_backbone",
            ),
            output_patch_dim=_as_optional_positive_int(
                model_block.get("output_patch_dim", model_block["latent_dim"]),
                "model.output_patch_dim",
            ),
            modality_embedding=_as_bool(
                model_block.get("modality_embedding", False),
                "model.modality_embedding",
            ),
            modality_vocab=_as_tuple_of_strings(
                model_block.get("modality_vocab", ["mammography", "mri", "ultrasound"])
            ) or ("mammography", "mri", "ultrasound"),
            modality_embedding_strategy=str(
                model_block.get("modality_embedding_strategy", "add_to_global")
            ).strip().lower() or "add_to_global",
            modality_embedding_dim=int(
                model_block.get("modality_embedding_dim", 32)
            ),
            batch_norm_policy=str(
                model_block.get("batch_norm_policy", "freeze_running_stats")
            ).strip() or "freeze_running_stats",
            train_batch_norm_affine=_as_bool(
                model_block.get("train_batch_norm_affine", True),
                "model.train_batch_norm_affine",
            ),
            local_high_conf_branch_enabled=_as_bool(
                local_branch_block.get("enabled", False),
                "model.local_high_conf_branch.enabled",
            ),
            local_high_conf_branch_training_ready=_as_bool(
                local_branch_block.get("local_branch_training_ready", False),
                "model.local_high_conf_branch.local_branch_training_ready",
            ),
            local_high_conf_branch_loss_weight=float(
                local_branch_block.get("loss_weight", 0.0)
            ),
            local_high_conf_branch_input_size=tuple(
                int(v)
                for v in local_branch_block.get("local_input_size", [512, 512])
            ),
            local_high_conf_branch_effective_stride=int(
                local_branch_block.get("effective_stride", 8)
            ),
            local_high_conf_branch_roi_margin=int(
                local_branch_block.get("roi_margin", 32)
            ),
        ),
        masking=MaskingConfig(
            mask_ratio=float(masking_block["mask_ratio"]),
            gaze_loss_mode=str(masking_block["gaze_loss_mode"]),
            gaze_weight_alpha=float(masking_block.get("gaze_weight_alpha", 1.0)),
            mask_strategy=str(masking_block.get("mask_strategy", "random")),
            gaze_mask_sampling_alpha=float(masking_block.get("gaze_mask_sampling_alpha", 0.7)),
            high_conf_mask_quota=float(masking_block.get("high_conf_mask_quota", 0.6)),
            min_random_mask_fraction=float(masking_block.get("min_random_mask_fraction", 0.3)),
            mask_sampling_temperature=float(masking_block.get("mask_sampling_temperature", 1.0)),
            gaze_mask_eps=float(masking_block.get("gaze_mask_eps", 1e-6)),
            min_visible_salient_fraction=min_visible_salient_fraction,
        ),
        semantic=SemanticConfig(
            text_backend=text_backend,
            prompt_embedding_path=_resolve_path(semantic_block["prompt_embedding_path"], project_root),
            semantic_soft_label_path=_resolve_path(semantic_block["semantic_soft_label_path"], project_root),
            semantic_manifest_path=_resolve_path(semantic_block["semantic_manifest_path"], project_root),
            semantic_soft_label_format=semantic_soft_label_format,
            semantic_soft_label_topk_path=semantic_soft_label_topk_path,
            semantic_soft_label_symmetrization=str(
                semantic_block.get("semantic_soft_label_symmetrization", "max")
            ).strip()
            or "max",
            semantic_unit_mapping_path=_resolve_optional_path(
                semantic_block.get("semantic_unit_mapping_path"), project_root
            ),
            semantic_unit_topk_path=_resolve_optional_path(
                semantic_block.get("semantic_unit_topk_path"), project_root
            ),
            semantic_unit_source_root=_resolve_optional_path(
                semantic_block.get("semantic_unit_source_root"), project_root
            ),
            semantic_unit_schema_version=str(
                semantic_block.get("semantic_unit_schema_version", "")
            ).strip(),
            semantic_unit_binding_version=str(
                semantic_block.get("semantic_unit_binding_version", "")
            ).strip(),
            birads_prior_manifest_path=_resolve_path(
                semantic_block["birads_prior_manifest_path"],
                project_root,
            ),
            high_conf_weight_alpha=float(semantic_block.get("high_conf_weight_alpha", 1.0)),
            reconstruction_teacher_source=str(
                semantic_block.get("reconstruction_teacher_source", "self_masked_reconstruction")
            ).strip()
            or "self_masked_reconstruction",
            active_concept_heads=active_heads,
            pending_concept_heads=pending_heads,
            concept_head_policy=concept_head_policy,
            concept_head_output_dims={
                str(key): int(value)
                for key, value in (semantic_block.get("concept_head_output_dims") or {}).items()
            } or p0b_output_dims or _concept_head_output_dims(active_heads),
            formal_p0b=formal_p0b,
        ),
        losses=LossConfig(
            reconstruction_weight=float(loss_block.get("reconstruction_weight", 1.0)),
            global_align_weight=float(loss_block.get("global_align_weight", 1.0)),
            visible_align_weight=float(loss_block.get("visible_align_weight", 1.0)),
            semantic_soft_weight=float(loss_block.get("semantic_soft_weight", 1.0)),
            concept_loss_weight=float(loss_block.get("concept_loss_weight", 1.0)),
            concept_consistency_weight=float(loss_block.get("concept_consistency_weight", 1.0)),
            graph_consistency_weight=float(loss_block.get("graph_consistency_weight", 0.0)),
            concept_head_weights=concept_head_weights,
            concept_consistency_head_weights=concept_consistency_head_weights,
            **_parse_conflict_aware_loss(loss_block),
        ),
        train=TrainConfig(
            optimizer=str(train_block.get("optimizer", "adamw")).strip().lower(),
            learning_rate=float(train_block["learning_rate"]),
            weight_decay=float(train_block.get("weight_decay", 0.0)),
            max_epochs=int(train_block.get("max_epochs", 1)),
            max_steps=int(train_block["max_steps"]),
            grad_clip_norm=_as_optional_positive_float(
                train_block.get("grad_clip_norm"),
                "train.grad_clip_norm",
            ),
            log_every_n_steps=max(1, int(train_block.get("log_every_n_steps", 10))),
            resume_from_checkpoint=train_resume,
            differential_lr=_parse_differential_lr(train_block.get("differential_lr")),
            use_amp=_as_bool(train_block.get("use_amp", False), "train.use_amp"),
            scheduler=_parse_scheduler(train_block.get("scheduler")),
        ),
        eval=EvalConfig(
            enabled=_as_bool(eval_block.get("enabled", True), "eval.enabled"),
            every_n_steps=_as_optional_positive_int(eval_block.get("every_n_steps"), "eval.every_n_steps"),
            every_n_epochs=_as_optional_positive_int(
                eval_block.get("every_n_epochs"),
                "eval.every_n_epochs",
            ),
            attention_top_fraction=float(eval_block.get("attention_top_fraction", 0.2)),
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=_as_optional_positive_int(
                checkpoint_block.get("save_every_n_steps"),
                "checkpoint.save_every_n_steps",
            ),
            save_every_n_epochs=_as_optional_positive_int(
                checkpoint_block.get("save_every_n_epochs"),
                "checkpoint.save_every_n_epochs",
            ),
            save_last=_as_bool(checkpoint_block.get("save_last", True), "checkpoint.save_last"),
            save_best=_as_bool(checkpoint_block.get("save_best", False), "checkpoint.save_best"),
            monitor=str(checkpoint_block.get("monitor", "loss_total")).strip() or "loss_total",
            mode=str(checkpoint_block.get("mode", "min")).strip().lower() or "min",
            resume_from=checkpoint_resume,
        ),
        reproducibility=ReproducibilityConfig(
            seed=int(reproducibility_block.get("seed", 42)),
            deterministic_ablation=_as_bool(
                reproducibility_block.get("deterministic_ablation", True),
                "reproducibility.deterministic_ablation",
            ),
            reuse_initial_model=_as_bool(
                reproducibility_block.get("reuse_initial_model", False),
                "reproducibility.reuse_initial_model",
            ),
            reuse_patch_mask=_as_bool(
                reproducibility_block.get("reuse_patch_mask", False),
                "reproducibility.reuse_patch_mask",
            ),
        ),
        source_intensity_audit=_parse_source_intensity_audit(raw_config, project_root),
        clinical_graph=clinical_graph_config,
        graph_encoder=graph_encoder_config,
        teacher=raw_config.get("teacher") if isinstance(raw_config.get("teacher"), dict) else None,
        formal_bundle_root=_resolve_optional_path(raw_config.get("formal_bundle_root"), project_root),
        formal_backbone_weight_path=_resolve_optional_path(
            raw_config.get("formal_backbone_weight_path"),
            project_root,
        ),
    )


def _load_legacy_config(
    raw_config: dict[str, Any],
    config_path: Path,
) -> Stage1JointTrainerConfig:
    from breast_pretrain.train.stage1_joint.config_legacy import load_legacy_config
    return load_legacy_config(raw_config, config_path)


def load_stage1_joint_trainer_config(
    config_path: str | Path,
) -> Stage1JointTrainerConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_config_path}")
    raw_config = _load_raw_config(resolved_config_path)
    validate_formal_config_keys(raw_config, source=str(resolved_config_path))
    raw_config = normalize_resolved_formal_config(raw_config); validate_formal_config_keys(raw_config, source=f"{resolved_config_path} (normalized)")

    if "model" in raw_config or "masking" in raw_config or "train" in raw_config:
        return _load_formal_config(raw_config, resolved_config_path)
    return _load_legacy_config(raw_config, resolved_config_path)


def override_stage1_joint_trainer_config(
    config: Stage1JointTrainerConfig,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
    text_dim: int | None = None,
    pretrained_weight_path: str | None = None,
    init_state_path: Path | None = None,
    artifact_dir: Path | None = None,
    high_conf_prior_npz: Path | None = None,
    high_conf_prior_manifest: Path | None = None,
    resume_from: Path | str | None = None,
) -> Stage1JointTrainerConfig:
    validate_formal_config_keys(_load_raw_config(Path(config.config_path).expanduser().resolve()), source=str(config.config_path))
    data_config = config.data
    train_config = config.train
    model_config = config.model
    if output_dir is not None:
        data_config = replace(data_config, output_dir=Path(output_dir).expanduser().resolve())
    if max_samples is not None:
        data_config = replace(data_config, max_samples=int(max_samples))
    if max_steps is not None:
        train_config = replace(train_config, max_steps=int(max_steps))
    if text_dim is not None and text_dim > 0:
        model_config = replace(model_config, text_dim=int(text_dim))
    if pretrained_weight_path is not None:
        model_config = replace(model_config, pretrained_weight_path=str(pretrained_weight_path))
    checkpoint_config = config.checkpoint
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        train_config = replace(train_config, resume_from_checkpoint=resume_path)
        checkpoint_config = replace(checkpoint_config, resume_from=resume_path)
    if (
        data_config is config.data
        and train_config is config.train
        and model_config is config.model
        and checkpoint_config is config.checkpoint
    ):
        return config
    return replace(config, data=data_config, train=train_config, model=model_config, checkpoint=checkpoint_config)
