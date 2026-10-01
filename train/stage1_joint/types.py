from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from breast_pretrain.data.transforms.stage1_transform_spec import ImageSize
from breast_pretrain.clinical_graph_encoder.config import GraphEncoderConfig
from breast_pretrain.clinical_graph_encoder.hetero_graph_encoder import GraphEncoderOutput


@dataclass(frozen=True)
class ReferencePaths:
    panderm_repo_path: Path | None
    fgclip_repo_path: Path | None
    cogaze_repo_path: Path | None
    ultrasound_clip_repo_path: Path | None
    import_policy: str = "read_only_reference_only"


@dataclass(frozen=True)
class RunMetadataConfig:
    run_tier: str = "development"
    model_role: str = "unspecified"
    compliance_status: str = "not_final_model"
    known_limitations: tuple[str, ...] = ()
    allowed_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataConfig:
    project_root: Path
    image_manifest_path: Path
    attention_map_dir: Path | None
    teacher_latent_dir: Path | None
    text_prompt_path: Path | None
    image_size: ImageSize
    batch_size: int
    num_workers: int
    device: str
    output_dir: Path
    max_samples: int | None = None
    require_attention_prior_paths: bool = False
    require_teacher_latents: bool = False
    image_size_by_modality: dict[str, tuple[int, int]] | None = None
    transform_policy_by_modality: dict[str, str] | None = None
    batch_policy: str = "fixed_batch_size"
    batch_size_by_modality: dict[str, int] | None = None
    sampler_contract_version: str = "legacy_historical_sampler_v1"
    shuffle: bool = False
    dataset_entry_v2_enabled: bool = False
    dataset_entry_v2_image_release_root: Path | None = None
    dataset_entry_v2_gaze_release_root: Path | None = None
    canonical_runtime: dict[str, Any] | None = None
    source_runtime_cache: dict[str, Any] | None = None
    gaze_membership_path: Path | None = None
    # These counts belong to the supplied frozen membership authority. There
    # is deliberately no project-specific default in the public package.
    gaze_membership_expected_available: int | None = None
    gaze_membership_expected_disabled: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int
    latent_dim: int
    text_dim: int
    align_dim: int
    vision_encoder_name: str = "minimal_patch_encoder"
    pretrained_model_path: Optional[str] = None
    pretrained_weight_path: Optional[str] = None
    backbone_expected_sha256: Optional[str] = None
    allow_missing_pretrained_fallback: bool = False
    freeze_backbone: bool = False
    output_patch_dim: Optional[int] = None
    modality_embedding: bool = False
    modality_vocab: tuple[str, ...] = ("mammography", "mri", "ultrasound")
    modality_embedding_strategy: str = "add_to_global"
    modality_embedding_dim: int = 32
    batch_norm_policy: str = "freeze_running_stats"
    train_batch_norm_affine: bool = True
    local_high_conf_branch_enabled: bool = False
    local_high_conf_branch_training_ready: bool = False
    local_high_conf_branch_loss_weight: float = 0.0
    local_high_conf_branch_input_size: tuple[int, int] = (512, 512)
    local_high_conf_branch_effective_stride: int = 8
    local_high_conf_branch_roi_margin: int = 32


@dataclass(frozen=True)
class MaskingConfig:
    mask_ratio: float
    gaze_loss_mode: str
    gaze_weight_alpha: float
    mask_strategy: str
    gaze_mask_sampling_alpha: float
    high_conf_mask_quota: float
    min_random_mask_fraction: float
    mask_sampling_temperature: float
    gaze_mask_eps: float
    min_visible_salient_fraction: float = 0.0


@dataclass(frozen=True)
class SemanticConfig:
    text_backend: str
    prompt_embedding_path: Path
    semantic_soft_label_path: Path
    semantic_manifest_path: Path
    semantic_soft_label_format: str
    semantic_soft_label_topk_path: Path | None
    semantic_soft_label_symmetrization: str
    birads_prior_manifest_path: Path
    high_conf_weight_alpha: float
    reconstruction_teacher_source: str
    active_concept_heads: tuple[str, ...]
    pending_concept_heads: tuple[str, ...]
    concept_head_output_dims: dict[str, int]
    semantic_unit_mapping_path: Path | None = None
    semantic_unit_topk_path: Path | None = None
    semantic_unit_source_root: Path | None = None
    semantic_unit_schema_version: str = ""
    semantic_unit_binding_version: str = ""
    concept_head_policy: str = "confirmed_labels_only_with_observed_mask_strict"
    formal_p0b: dict[str, Any] | None = None


@dataclass(frozen=True)
class LossConfig:
    reconstruction_weight: float
    global_align_weight: float
    visible_align_weight: float
    semantic_soft_weight: float
    concept_loss_weight: float
    concept_consistency_weight: float
    concept_head_weights: dict[str, float]
    concept_consistency_head_weights: dict[str, float]
    graph_consistency_weight: float = 0.0
    conflict_aware_enabled: bool = False
    conflict_aware_semantic_visible_coverage_target: float = 0.25
    conflict_aware_reconstruction_masked_gaze_target: float = 0.25
    conflict_aware_min_weight: float = 0.5
    conflict_aware_max_weight: float = 2.0
    conflict_aware_warmup_steps: int = 0
    allow_dynamic_graph_consistency_weighting: bool = False


@dataclass(frozen=True)
class DifferentialLRConfig:
    enabled: bool = False
    backbone_lr: float = 1.0e-5
    head_lr: float = 1.0e-4


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    max_steps: int
    grad_clip_norm: float | None
    log_every_n_steps: int
    resume_from_checkpoint: Path | None = None
    differential_lr: DifferentialLRConfig | None = None
    use_amp: bool = False
    scheduler: SchedulerConfig | None = None


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool
    every_n_steps: int | None
    every_n_epochs: int | None
    attention_top_fraction: float = 0.2


@dataclass(frozen=True)
class CheckpointConfig:
    save_every_n_steps: int | None
    save_every_n_epochs: int | None
    save_last: bool
    save_best: bool
    monitor: str
    mode: str
    resume_from: Path | None = None


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic_ablation: bool
    reuse_initial_model: bool
    reuse_patch_mask: bool


@dataclass(frozen=True)
class SourceIntensityAuditConfig:
    required: bool = False
    approved_tolerance: float = 1e-4
    audit_report_path: Path | None = None


@dataclass(frozen=True)
class ClinicalGraphConfig:
    version: str = "tri_modal_clinical_graph_v1"
    nodes_path: Path | None = None
    edges_path: Path | None = None
    mapping_rules_path: Path | None = None
    consistency_rules_path: Path | None = None
    prompt_templates_path: Path | None = None
    sidecar_case_concept_vector_path: Path | None = None
    sidecar_index_path: Path | None = None
    use_for_structured_prompt: bool = False
    use_for_semantic_soft_labels: bool = False
    use_for_concept_targets: bool = False
    use_for_concept_consistency: bool = False
    forbid_direct_graph_node_alignment: bool = True


@dataclass(frozen=True)
class Stage1JointTrainerConfig:
    config_path: Path
    metadata: RunMetadataConfig
    data: DataConfig
    references: ReferencePaths
    model: ModelConfig
    masking: MaskingConfig
    semantic: SemanticConfig
    losses: LossConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig
    reproducibility: ReproducibilityConfig
    source_intensity_audit: SourceIntensityAuditConfig | None = None
    clinical_graph: ClinicalGraphConfig | None = None
    graph_encoder: GraphEncoderConfig | None = None
    teacher: dict[str, object] | None = None
    formal_bundle_root: Path | None = None
    formal_backbone_weight_path: Path | None = None


@dataclass(frozen=True)
class Stage1JointBatch:
    image: torch.Tensor
    attention_map: torch.Tensor
    high_conf_mask: torch.Tensor
    image_ids: list[str]
    dataset_ids: list[str]
    case_ids: list[str]
    manifest_row_indices: list[int]
    teacher_latent_paths: list[str]
    teacher_source_types: list[str]
    prompts: list[str]
    prompt_warnings: list[str]
    concept_targets: dict[str, torch.Tensor]
    concept_valid_masks: dict[str, torch.Tensor]
    modalities: list[str]
    modality_ids: torch.Tensor
    gaze_supervision_sources: list[str]
    audit_statuses: list[str]
    attention_map_paths: list[str]
    high_conf_mask_paths: list[str]
    patch_gaze_weight_paths: list[str]
    prior_statuses: list[str]
    coverage_ratios: list[float]
    high_conf_area_ratios: list[float]
    inside_ratios: list[float]
    patch_gaze_weight: torch.Tensor | None = None
    high_conf_patch_prior: torch.Tensor | None = None
    valid_content_patch_mask: torch.Tensor | None = None
    concept_value_valid_masks: dict[str, torch.Tensor] | None = None
    transform_policies: list[str] | None = None
    transform_spec_checksums: list[str] | None = None
    image_transform_geometry_checksums: list[str] | None = None
    image_transform_geometry_jsons: list[str] | None = None
    attention_transform_spec_checksums: list[str] | None = None
    attention_transform_geometry_checksums: list[str] | None = None
    attention_transform_geometry_jsons: list[str] | None = None
    high_conf_mask_transform_spec_checksums: list[str] | None = None
    high_conf_mask_transform_geometry_checksums: list[str] | None = None
    high_conf_mask_transform_geometry_jsons: list[str] | None = None
    clinical_graph_v2_node_values: torch.Tensor | None = None
    clinical_graph_v2_observed_mask: torch.Tensor | None = None
    clinical_graph_v2_node_index: torch.Tensor | None = None
    clinical_graph_v2_node_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MaskingRuntimeState:
    normalized_gaze_loss_mode: str
    configured_mask_strategy: str
    normalized_mask_strategy: str
    mask_prior_mode: str
    mask_checksum: str
    shared_patch_masks: list[torch.Tensor] | None
    mask_generator: torch.Generator | None


@dataclass(frozen=True)
class MaskingStepOutput:
    patch_mask: torch.Tensor
    patch_weights: torch.Tensor
    attention_tokens: torch.Tensor
    high_conf_tokens: torch.Tensor
    mask_policy_used: tuple[str, ...]
    adaptive_gaze_quota: torch.Tensor
    adaptive_random_fraction: torch.Tensor
    fallback_reason: tuple[str, ...]
    masked_gaze_coverage: torch.Tensor
    visible_gaze_coverage: torch.Tensor
    high_conf_mask_quota_actual: torch.Tensor
    random_mask_fraction_actual: torch.Tensor
    warnings: list[str]
    total_salient_count: torch.Tensor | None = None
    masked_salient_count: torch.Tensor | None = None
    visible_salient_count: torch.Tensor | None = None
    q_vis: torch.Tensor | None = None
    visible_salient_floor_violation_count: torch.Tensor | None = None
    adaptive_gaze_guided_by_modality: dict[str, int] | None = None
    adaptive_weak_qc_by_modality: dict[str, int] | None = None
    random_fallback_by_modality: dict[str, int] | None = None
    fallback_reason_histogram: dict[str, int] | None = None


@dataclass(frozen=True)
class TeacherLatentBatch:
    teacher_latent: torch.Tensor
    teacher_sources: list[str]
    missing_teacher_flags: list[bool]


@dataclass(frozen=True)
class StudentForwardOutput:
    patch_tokens: torch.Tensor
    global_image_feature: torch.Tensor
    patch_grid: tuple[int, int] | None = None
    num_patches: int | None = None


@dataclass(frozen=True)
class SemanticForwardOutput:
    global_image_embedding: torch.Tensor
    visible_image_embedding: torch.Tensor
    case_embedding: torch.Tensor
    concept_feature: torch.Tensor
    concept_logits: dict[str, torch.Tensor]
    semantic_batch_positions: list[int]
    semantic_image_ids: list[str]
    semantic_target_matrix: torch.Tensor
    prior_records: Sequence[object]
    prior_schema_versions: tuple[str, ...]
    graph_encoder_output: GraphEncoderOutput | None = None
    graph_encoder_consumed: bool = False
    graph_prototype_projection: torch.nn.ModuleDict | None = None
    p0b_projected_prototypes: dict[str, dict[str, torch.Tensor]] | None = None
    diagnostic_tensors: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class LossComputationResult:
    losses: dict[str, torch.Tensor]
    semantic_valid_count: int
    semantic_skipped_batch_count: int
    concept_head_losses: dict[str, torch.Tensor]
    concept_head_correct_counts: dict[str, float]
    concept_head_valid_label_counts: dict[str, int]
    concept_head_missing_label_counts: dict[str, int]
    dynamic_loss_weights: dict[str, float]
    loss_weight_audit: dict[str, dict[str, float]]
    local_branch_metrics: dict[str, Any]
    warnings: list[str]
    graph_consistency_metrics: dict[str, Any]


@dataclass(frozen=True)
class EvalHookResult:
    payload: dict[str, object]
    output_path: Path
