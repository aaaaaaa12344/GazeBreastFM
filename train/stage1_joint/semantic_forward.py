from __future__ import annotations

from dataclasses import dataclass

import torch

from breast_pretrain.clinical_graph_encoder.semantic_prior import blend_graph_semantic_prior
from breast_pretrain.semantics import SemanticSoftLabelIndex, Stage1BiradsPriorIndex
from breast_pretrain.semantics.semantic_unit_soft_label_adapter import SemanticUnitSoftLabelAdapter
from breast_pretrain.semantics.concept_prototypes import PrototypeRuntimeIndex
from breast_pretrain.data.stage1_sparse_concept_contract import load_frozen_concept_schema
from breast_pretrain.text.prompt_embedding import PromptEmbeddingCacheEncoder
from breast_pretrain.train.stage1_joint.types import (
    MaskingStepOutput,
    SemanticForwardOutput,
    Stage1JointBatch,
    Stage1JointTrainerConfig,
    StudentForwardOutput,
)
from breast_pretrain.train.stage1_joint.graph_encoder_forward import (
    Stage1GraphEncoderRuntime,
    build_graph_encoder_runtime,
    forward_graph_encoder,
)
from breast_pretrain.train.stage1_semantic_branch import (
    Stage1SemanticBranch,
    build_visible_region_weights,
)
from breast_pretrain.train.stage1_joint.ddp_components import unwrap_ddp


@dataclass
class Stage1JointSemanticRuntime:
    branch: Stage1SemanticBranch
    prompt_encoder: PromptEmbeddingCacheEncoder
    semantic_index: SemanticSoftLabelIndex | SemanticUnitSoftLabelAdapter
    birads_prior_index: Stage1BiradsPriorIndex | None
    graph_runtime: Stage1GraphEncoderRuntime | None = None
    prototype_index: PrototypeRuntimeIndex | None = None


def build_semantic_runtime(
    config: Stage1JointTrainerConfig,
    device: torch.device,
) -> Stage1JointSemanticRuntime:
    formal_mode = str(config.metadata.run_tier).strip() in {
        "formal_production",
        "production_ready_candidate",
    }
    required_prompt_path = config.data.text_prompt_path if formal_mode else None
    if formal_mode and required_prompt_path is None:
        raise ValueError(
            "Formal Stage 1 requires the frozen Effective Report prompt authority path."
        )
    visual_dim = config.model.output_patch_dim or config.model.latent_dim
    branch = Stage1SemanticBranch(
            visual_dim=visual_dim,
            text_dim=config.model.text_dim,
            align_dim=config.model.align_dim,
            active_concept_heads=config.semantic.active_concept_heads,
            concept_head_output_dims=config.semantic.concept_head_output_dims,
        ).to(device)
    graph_runtime = build_graph_encoder_runtime(config, device)
    if graph_runtime is not None:
        branch.graph_encoder = graph_runtime.encoder
        branch.graph_text_fusion = graph_runtime.text_fusion
        branch.configure_graph_prototype_projectors(
            config.graph_encoder.hidden_dim,
            prototype_head_names=config.semantic.active_concept_heads,
        )
    prototype_index = None
    if config.semantic.formal_p0b is not None:
        p0b = config.semantic.formal_p0b
        schema = load_frozen_concept_schema(str(p0b["concept_schema_path"]), expected_sha256=str(p0b["concept_schema_sha256"]))
        prototype_index = PrototypeRuntimeIndex.load(str(p0b["prototype_asset_path"]), schema=schema, embedding_authority_hash=p0b.get("embedding_authority_hash"), target_authority_hash=p0b.get("concept_target_authority_hash"))
    semantic_mapping = config.semantic.semantic_unit_mapping_path
    semantic_topk = config.semantic.semantic_unit_topk_path
    if config.semantic.formal_p0b is not None and (semantic_mapping is None or semantic_topk is None):
        raise ValueError(
            "Formal Stage 1 requires both frozen semantic-unit mapping and top-k assets; "
            "legacy image-level semantic lookup is forbidden."
        )
    if semantic_mapping is not None or semantic_topk is not None:
        if semantic_mapping is None or semantic_topk is None:
            raise ValueError("Semantic-unit runtime requires both mapping and top-k paths.")
        semantic_index = SemanticUnitSoftLabelAdapter(
            mapping_path=semantic_mapping,
            topk_path=semantic_topk,
            source_root=config.semantic.semantic_unit_source_root,
            expected_schema_version=config.semantic.semantic_unit_schema_version or None,
            expected_binding_version=config.semantic.semantic_unit_binding_version or None,
        )
    else:
        semantic_index = SemanticSoftLabelIndex(
            matrix_path=config.semantic.semantic_soft_label_path,
            manifest_path=config.semantic.semantic_manifest_path,
            label_format=config.semantic.semantic_soft_label_format,
            topk_path=config.semantic.semantic_soft_label_topk_path,
            symmetrization_strategy=config.semantic.semantic_soft_label_symmetrization,
        )
    return Stage1JointSemanticRuntime(
        branch=branch,
        prompt_encoder=PromptEmbeddingCacheEncoder(
            cache_path=config.semantic.prompt_embedding_path,
            text_dim=config.model.text_dim,
            allow_missing_fallback=not formal_mode,
            required_prompt_path=required_prompt_path,
        ),
        semantic_index=semantic_index,
        birads_prior_index=(None if config.semantic.formal_p0b is not None else Stage1BiradsPriorIndex(
            manifest_path=config.semantic.birads_prior_manifest_path,
        )),
        graph_runtime=graph_runtime,
        prototype_index=prototype_index,
    )


def forward_semantic(
    runtime: Stage1JointSemanticRuntime,
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    student_output: StudentForwardOutput,
    masking_output: MaskingStepOutput,
    device: torch.device,
    *,
    visible_patch_tokens: torch.Tensor | None = None,
    valid_content_patch_mask: torch.Tensor | None = None,
    diagnostic: bool = False,
) -> tuple[SemanticForwardOutput, list[str]]:
    effective_report_embeddings, prompt_warnings = runtime.prompt_encoder.encode_prompts(batch.prompts)
    effective_report_embeddings = effective_report_embeddings.to(device=device, dtype=torch.float32)
    graph_output, alignment_text_embeddings, graph_warnings = forward_graph_encoder(
        runtime.graph_runtime,
        effective_report_embeddings,
        prototype_head_names=config.semantic.active_concept_heads,
        clinical_graph_v2_node_values=batch.clinical_graph_v2_node_values,
        clinical_graph_v2_observed_mask=batch.clinical_graph_v2_observed_mask,
        clinical_graph_v2_node_index=batch.clinical_graph_v2_node_index,
    )

    visible_weights = build_visible_region_weights(
        attention_tokens=masking_output.attention_tokens,
        high_conf_tokens=masking_output.high_conf_tokens,
        patch_mask=masking_output.patch_mask,
        high_conf_weight_alpha=config.semantic.high_conf_weight_alpha,
    )
    # Apply valid_content_patch_mask to exclude padding patches from visible pooling
    if valid_content_patch_mask is not None:
        content_mask = valid_content_patch_mask.to(
            device=visible_weights.device, dtype=visible_weights.dtype
        )
        visible_weights = visible_weights * content_mask
        denom = visible_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        visible_weights = visible_weights / denom

    # Use masked-view context patches for visible feature (NOT clean view patches)
    patch_source = visible_patch_tokens if visible_patch_tokens is not None else student_output.patch_tokens
    visible_image_feature = (
        visible_weights.unsqueeze(-1) * patch_source
    ).sum(dim=1)
    outputs = runtime.branch(
        global_image_feature=student_output.global_image_feature,
        visible_image_feature=visible_image_feature,
        case_feature=alignment_text_embeddings,
    )
    diagnostic_tensors = None
    if diagnostic:
        diagnostic_tensors = unwrap_ddp(runtime.branch).diagnostic_projection_tensors(
            student_output.global_image_feature,
            visible_image_feature,
            alignment_text_embeddings,
        )
        diagnostic_tensors["visible_context_representation"] = visible_image_feature

    if isinstance(runtime.semantic_index, SemanticUnitSoftLabelAdapter):
        semantic_batch = runtime.semantic_index.resolve_batch(
            batch.image_ids,
            row_indices=batch.manifest_row_indices,
            semantic_target_indices=getattr(batch, "semantic_target_indices", None),
        )
    else:
        semantic_batch = runtime.semantic_index.resolve_batch(
            batch.image_ids,
            row_indices=batch.manifest_row_indices,
        )
    warnings_list = list(prompt_warnings) + list(graph_warnings) + list(semantic_batch.warnings)
    prior_records = []
    prior_schema_versions: tuple[str, ...] = ()
    semantic_target_matrix = semantic_batch.matrix.to(device=device)
    if semantic_batch.valid_count >= 2 and runtime.birads_prior_index is not None:
        prior_records, prior_warnings = runtime.birads_prior_index.resolve_batch(
            semantic_batch.image_ids
        )
        warnings_list.extend(prior_warnings)
        prior_schema_versions = tuple(
            sorted({record.schema_version for record in prior_records})
        )
        if graph_output is not None and config.graph_encoder is not None:
            semantic_target_matrix = blend_graph_semantic_prior(
                semantic_targets=semantic_target_matrix,
                graph_output=graph_output,
                concept_targets=batch.concept_targets,
                concept_valid_masks=batch.concept_valid_masks,
                semantic_batch_positions=list(semantic_batch.batch_positions),
                weight=config.graph_encoder.semantic_prior_weight,
            )

    projected_prototypes: dict[str, dict[str, torch.Tensor]] = {}
    if runtime.prototype_index is not None:
        by_concept: dict[str, dict[str, torch.Tensor]] = {}
        for key, vector in runtime.prototype_index.vectors.items():
            concept_id, value = key.split("::", 1)
            by_concept.setdefault(concept_id, {})[value] = vector.to(device=device, dtype=effective_report_embeddings.dtype)
        branch_module = unwrap_ddp(runtime.branch)
        projected_prototypes = {concept_id: {value: branch_module.project_case(vector.unsqueeze(0))[0] for value, vector in values.items()} for concept_id, values in by_concept.items()}
    return (
        SemanticForwardOutput(
            global_image_embedding=outputs["global_image_embedding"],
            visible_image_embedding=outputs["visible_image_embedding"],
            case_embedding=outputs["case_embedding"],
            concept_feature=outputs["concept_feature"],
            concept_logits=outputs["concept_logits"],
            semantic_batch_positions=list(semantic_batch.batch_positions),
            semantic_image_ids=list(semantic_batch.image_ids),
            semantic_target_matrix=semantic_target_matrix,
            prior_records=prior_records,
            prior_schema_versions=prior_schema_versions,
            graph_encoder_output=graph_output,
            graph_encoder_consumed=graph_output is not None,
            graph_prototype_projection=(
                unwrap_ddp(runtime.branch).graph_prototype_projectors
                if len(unwrap_ddp(runtime.branch).graph_prototype_projectors) > 0
                else None
            ),
            p0b_projected_prototypes=projected_prototypes or None,
            diagnostic_tensors=diagnostic_tensors,
        ),
        warnings_list,
    )
