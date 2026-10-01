from __future__ import annotations

import torch
from torch.nn import functional as F

from breast_pretrain.clinical_graph_sidecar.activation_audit import (
    build_node_id_head_mapping,
)
from breast_pretrain.losses.graph_consistency import (
    compute_graph_consistency_loss,
    compute_graph_encoder_consistency_loss,
    compute_graph_prototype_alignment_loss,
)
from breast_pretrain.semantics import (
    compute_concept_consistency_loss,
    compute_prior_head_consistency_loss,
)
from breast_pretrain.semantics.concept_prototypes import compute_p0b_pair_consistency_loss, compute_p0b_prototype_loss, combine_p0b_cc_loss
from breast_pretrain.data.stage1_sparse_concept_contract import load_frozen_concept_schema
from breast_pretrain.train.masked_latent_smoke import compute_masked_latent_loss
from breast_pretrain.train.stage1_joint.teacher_latents import (
    RECONSTRUCTION_SOURCE_SELF,
    RECONSTRUCTION_SOURCE_TEACHER_NPY,
)
from breast_pretrain.train.stage1_joint.dynamic_loss_weighting import (
    build_loss_weight_audit,
    compute_conflict_aware_weights,
    loss_weight_static_config,
)
from breast_pretrain.train.stage1_joint.types import (
    LossComputationResult,
    MaskingStepOutput,
    SemanticForwardOutput,
    Stage1JointBatch,
    Stage1JointTrainerConfig,
    StudentForwardOutput,
    TeacherLatentBatch,
)


RECONSTRUCTION_LATENT_NORMALIZATION_CONTRACT = "RECONSTRUCTION_LATENT_NORMALIZATION_V1"
RECONSTRUCTION_LATENT_NORMALIZATION_EPS = 1.0e-6


def _normalize_reconstruction_latent_v1(latent: torch.Tensor) -> torch.Tensor:
    """Normalize each patch token in FP32 without a learnable affine."""
    if latent.ndim < 1 or int(latent.shape[-1]) <= 0:
        raise ValueError(
            f"reconstruction latent must have a non-empty feature axis, got {tuple(latent.shape)}"
        )
    return F.layer_norm(
        latent.float(),
        normalized_shape=(int(latent.shape[-1]),),
        weight=None,
        bias=None,
        eps=RECONSTRUCTION_LATENT_NORMALIZATION_EPS,
    )


def _paired_contrastive_loss(
    image_embeddings: torch.Tensor,
    case_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    # Keep the contrastive objective unchanged while avoiding FP16 matmul
    # gradient overflow under AMP.
    logits = (image_embeddings.float() @ case_embeddings.float().transpose(0, 1)) / float(temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
    )


def _soft_pair_contrastive_loss(
    image_embeddings: torch.Tensor,
    case_embeddings: torch.Tensor,
    semantic_targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if image_embeddings.shape[0] < 2:
        return image_embeddings.new_zeros(())
    targets = semantic_targets.to(device=image_embeddings.device, dtype=image_embeddings.dtype)
    targets = torch.maximum(
        targets.clamp(0.0, 1.0),
        torch.eye(targets.shape[0], device=targets.device, dtype=targets.dtype),
    )
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-6)
    logits = (image_embeddings.float() @ case_embeddings.float().transpose(0, 1)) / float(temperature)
    image_to_case = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    case_to_image = -(
        targets.transpose(0, 1) * F.log_softmax(logits.transpose(0, 1), dim=1)
    ).sum(dim=1).mean()
    return 0.5 * (image_to_case + case_to_image)


def _compute_unweighted_masked_reconstruction_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    per_patch_loss = (student_latent - teacher_latent).pow(2).mean(dim=-1)
    mask = patch_mask.to(dtype=per_patch_loss.dtype)
    return (per_patch_loss * mask).sum() / mask.sum().clamp_min(1.0)


def _build_graph_node_id_head_mapping(
    config: Stage1JointTrainerConfig,
) -> dict[str, str]:
    """Build graph node -> active concept head mapping without direct image-node alignment."""
    node_ids: list[str] = []
    if config.clinical_graph is not None and config.clinical_graph.nodes_path is not None:
        import csv

        with config.clinical_graph.nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
            node_ids = [str(row.get("node_id", "")).strip() for row in csv.DictReader(handle)]
    return build_node_id_head_mapping(set(config.semantic.active_concept_heads), node_ids)


def _compute_single_head_supervised_loss(
    head_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    value_valid_mask: torch.Tensor | None = None,
    target_type: str | None = None,
) -> tuple[torch.Tensor, float, int, int]:
    if value_valid_mask is not None:
        value_mask = value_valid_mask.to(device=logits.device, dtype=torch.bool)
        if value_mask.shape != logits.shape or targets.shape != logits.shape:
            raise ValueError(f"Formal multilabel target/mask shape mismatch for {head_name}.")
        count = int(value_mask.sum().item())
        if count <= 0:
            return logits.new_zeros(()), 0.0, 0, int(value_mask.numel())
        target = targets.to(device=logits.device, dtype=logits.dtype)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (losses * value_mask).sum() / value_mask.sum().clamp_min(1)
        correct = float((((logits >= 0.0).to(target.dtype) == target) & value_mask).sum().item())
        return loss, correct, count, int(value_mask.numel() - count)
    valid_tensor = valid_mask.to(dtype=torch.bool, device=logits.device)
    valid_count = int(valid_tensor.sum().item())
    missing_count = int(valid_tensor.numel() - valid_count)
    if valid_count <= 0:
        return logits.new_zeros(()), 0.0, 0, missing_count

    if target_type == "binary":
        selected_logits = logits[valid_tensor].squeeze(-1)
        selected_targets = targets[valid_tensor].to(device=logits.device, dtype=logits.dtype)
        loss_value = F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
        correct_count = float(((selected_logits >= 0.0).to(selected_targets.dtype) == selected_targets).sum().item())
        return loss_value, correct_count, valid_count, missing_count
    if target_type == "categorical_multilabel" or (target_type is None and head_name == "finding"):
        selected_logits = logits[valid_tensor]
        selected_targets = targets[valid_tensor].to(device=logits.device, dtype=logits.dtype)
        loss_value = F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
        predictions = (selected_logits >= 0.0).to(dtype=selected_targets.dtype)
        correct_count = float((predictions == selected_targets).all(dim=1).sum().item())
        return loss_value, correct_count, valid_count, missing_count

    if head_name == "cancer_label":
        selected_logits = logits[valid_tensor].squeeze(-1)
        selected_targets = targets[valid_tensor].to(device=logits.device, dtype=logits.dtype)
        loss_value = F.binary_cross_entropy_with_logits(selected_logits, selected_targets)
        predictions = (selected_logits >= 0.0).to(dtype=selected_targets.dtype)
        correct_count = float((predictions == selected_targets).sum().item())
        return loss_value, correct_count, valid_count, missing_count

    selected_logits = logits[valid_tensor]
    selected_targets = targets[valid_tensor].to(device=logits.device, dtype=torch.long)
    loss_value = F.cross_entropy(selected_logits, selected_targets)
    predictions = selected_logits.argmax(dim=1)
    correct_count = float((predictions == selected_targets).sum().item())
    return loss_value, correct_count, valid_count, missing_count


def _compute_concept_supervised_losses(
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    semantic_output: SemanticForwardOutput,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], dict[str, int], dict[str, int]]:
    reference_tensor = semantic_output.concept_feature
    concept_head_losses: dict[str, torch.Tensor] = {}
    concept_head_correct_counts: dict[str, float] = {}
    concept_head_valid_label_counts: dict[str, int] = {}
    concept_head_missing_label_counts: dict[str, int] = {}

    weighted_loss_terms: list[torch.Tensor] = []
    weighted_loss_weights: list[float] = []
    formal_specs = None
    if config.semantic.formal_p0b is not None:
        p0b = config.semantic.formal_p0b
        formal_specs = load_frozen_concept_schema(str(p0b["concept_schema_path"]), expected_sha256=str(p0b["concept_schema_sha256"])).concepts
    for head_name in config.semantic.active_concept_heads:
        logits = semantic_output.concept_logits[head_name]
        targets = batch.concept_targets[head_name]
        valid_mask = batch.concept_valid_masks[head_name]
        value_valid_mask = (batch.concept_value_valid_masks or {}).get(head_name)
        head_loss, correct_count, valid_count, missing_count = _compute_single_head_supervised_loss(
            head_name=head_name,
            logits=logits,
            targets=targets,
            valid_mask=valid_mask,
            value_valid_mask=value_valid_mask,
            target_type=formal_specs[head_name].target_type if formal_specs is not None else None,
        )
        concept_head_losses[head_name] = head_loss
        concept_head_correct_counts[head_name] = correct_count
        concept_head_valid_label_counts[head_name] = valid_count
        concept_head_missing_label_counts[head_name] = missing_count
        head_weight = float(config.losses.concept_head_weights.get(head_name, 0.0))
        if head_weight > 0.0 and valid_count > 0:
            weighted_loss_terms.append(head_weight * head_loss)
            weighted_loss_weights.append(head_weight)

    if not weighted_loss_terms:
        concept_loss = reference_tensor.new_zeros(())
    else:
        concept_loss = torch.stack(weighted_loss_terms).sum() / max(sum(weighted_loss_weights), 1e-6)
    return (
        concept_loss,
        concept_head_losses,
        concept_head_correct_counts,
        concept_head_valid_label_counts,
        concept_head_missing_label_counts,
    )


def _has_legal_off_diagonal_pair(mask: torch.Tensor) -> bool:
    """Mirror P0-B pair-valid semantics for scalar and multilabel supports."""
    valid = mask.to(dtype=torch.bool)
    if valid.ndim == 1:
        return int(valid.sum().item()) >= 2
    if valid.ndim == 2:
        joint = valid[:, None, :] & valid[None, :, :]
        pair_valid = joint.any(dim=-1)
        pair_valid.fill_diagonal_(False)
        return bool(pair_valid.any().item())
    raise ValueError("P0-B pair availability mask must be [B] or [B,K].")


def compute_stage1_joint_losses(
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    student_output: StudentForwardOutput,
    teacher_batch: TeacherLatentBatch,
    semantic_output: SemanticForwardOutput,
    masking_output: MaskingStepOutput,
    *,
    valid_content_patch_mask: torch.Tensor | None = None,
    local_branch_loss: torch.Tensor | None = None,
    local_branch_metrics: dict[str, Any] | None = None,
) -> LossComputationResult:
    source = config.semantic.reconstruction_teacher_source
    if source not in (RECONSTRUCTION_SOURCE_SELF, RECONSTRUCTION_SOURCE_TEACHER_NPY):
        raise ValueError(
            f"Unsupported reconstruction_teacher_source: {source!r}"
        )

    dynamic_weights, conflict_warnings = compute_conflict_aware_weights(config, masking_output)

    # Build effective reconstruction mask: only masked AND valid-content patches
    recon_mask = masking_output.patch_mask
    if valid_content_patch_mask is not None:
        content_mask = valid_content_patch_mask.to(device=recon_mask.device, dtype=torch.bool)
        # content_mask is [N], broadcast to [B, N]
        if content_mask.ndim == 1:
            content_mask = content_mask.unsqueeze(0)
        recon_mask = recon_mask & content_mask

    reconstruction_prediction = _normalize_reconstruction_latent_v1(
        student_output.patch_tokens
    )
    reconstruction_target = _normalize_reconstruction_latent_v1(
        teacher_batch.teacher_latent.detach()
    )
    reconstruction_loss = compute_masked_latent_loss(
        student_latent=reconstruction_prediction,
        teacher_latent=reconstruction_target,
        patch_mask=recon_mask,
        patch_weights=masking_output.patch_weights,
    )
    reconstruction_unweighted_loss = _compute_unweighted_masked_reconstruction_loss(
        student_latent=reconstruction_prediction,
        teacher_latent=reconstruction_target,
        patch_mask=recon_mask,
    )
    global_align_loss = _paired_contrastive_loss(
        semantic_output.global_image_embedding,
        semantic_output.case_embedding,
    )
    visible_align_loss = _paired_contrastive_loss(
        semantic_output.visible_image_embedding,
        semantic_output.case_embedding,
    )
    (
        concept_loss,
        concept_head_losses,
        concept_head_correct_counts,
        concept_head_valid_label_counts,
        concept_head_missing_label_counts,
    ) = _compute_concept_supervised_losses(
        config=config,
        batch=batch,
        semantic_output=semantic_output,
    )

    semantic_soft_loss = batch.image.new_zeros(())
    concept_consistency_loss = batch.image.new_zeros(())
    warnings_list: list[str] = list(conflict_warnings)
    semantic_valid_count = int(len(semantic_output.semantic_batch_positions))
    semantic_skipped_batch_count = 0
    if semantic_valid_count >= 2:
        selected_positions = semantic_output.semantic_batch_positions
        selected_image_embeddings = 0.5 * (
            semantic_output.global_image_embedding[selected_positions]
            + semantic_output.visible_image_embedding[selected_positions]
        )
        selected_case_embeddings = semantic_output.case_embedding[selected_positions]
        selected_concept_logits = {
            head_name: head_logits[selected_positions]
            for head_name, head_logits in semantic_output.concept_logits.items()
        }
        semantic_soft_loss = _soft_pair_contrastive_loss(
            image_embeddings=selected_image_embeddings,
            case_embeddings=selected_case_embeddings,
            semantic_targets=semantic_output.semantic_target_matrix,
        )
        if config.semantic.formal_p0b is not None:
            p0b = config.semantic.formal_p0b
            schema = load_frozen_concept_schema(
                str(p0b["concept_schema_path"]), expected_sha256=str(p0b["concept_schema_sha256"])
            )
            comparable = {
                concept_id: (
                    batch.concept_targets[concept_id][selected_positions],
                    ((batch.concept_value_valid_masks or {}).get(concept_id, batch.concept_valid_masks[concept_id]))[selected_positions],
                )
                for concept_id, spec in schema.concepts.items()
                if spec.supervision_modes.get("consistency", False) and concept_id in batch.concept_targets
            }
            # Availability means an actual legal off-diagonal pair, not merely
            # one observed target whose pair loss happens to reduce to zero.
            pair_available = any(_has_legal_off_diagonal_pair(mask) for _, mask in comparable.values())
            pair_loss = compute_p0b_pair_consistency_loss(
                semantic_output.concept_feature[selected_positions], comparable
            )
            positive_support: dict[str, torch.Tensor] = {}
            prototype_tensors: dict[str, torch.Tensor] = {}
            for concept_id, values in (semantic_output.p0b_projected_prototypes or {}).items():
                spec = schema.concepts[concept_id]
                ordered = [value for value in spec.value_space if value in values]
                if not ordered:
                    continue
                target = batch.concept_targets[concept_id][selected_positions]
                valid = (batch.concept_value_valid_masks or {}).get(concept_id, batch.concept_valid_masks[concept_id])[selected_positions].to(device=target.device)
                if spec.target_type == "categorical_multilabel":
                    support = torch.stack([target[:, spec.value_space.index(value)].to(torch.bool) & valid[:, spec.value_space.index(value)].to(torch.bool) for value in ordered], dim=1)
                elif spec.target_type == "binary":
                    support = torch.stack([(target.to(torch.bool) & valid.to(torch.bool)) if value == "present" else ((~target.to(torch.bool)) & valid.to(torch.bool)) for value in ordered], dim=1)
                else:
                    support = torch.stack([(target == spec.value_space.index(value)) & valid.to(torch.bool) for value in ordered], dim=1)
                positive_support[concept_id] = support
                prototype_tensors[concept_id] = torch.stack([values[value] for value in ordered])
            prototype_available = any(bool(value.any().item()) for value in positive_support.values())
            prototype_loss = compute_p0b_prototype_loss(semantic_output.concept_feature[selected_positions], prototype_tensors, positive_support)
            concept_consistency_loss = combine_p0b_cc_loss(pair_loss, prototype_loss, pair_available=pair_available, prototype_available=prototype_available)
        else:
            concept_consistency_loss = compute_concept_consistency_loss(
                image_embeddings=selected_image_embeddings,
                semantic_targets=semantic_output.semantic_target_matrix,
                prior_records=list(semantic_output.prior_records),
            )
            prior_head_consistency_loss = compute_prior_head_consistency_loss(
                concept_logits=selected_concept_logits,
                prior_records=list(semantic_output.prior_records),
                head_weights=config.losses.concept_consistency_head_weights,
            )
            concept_consistency_loss = concept_consistency_loss + prior_head_consistency_loss
    else:
        semantic_skipped_batch_count = 1
        warnings_list.append("semantic_soft_contrastive_skipped:valid_semantic_batch_size<2")

    losses = {
        "reconstruction_gaze_weighted": reconstruction_loss,
        "reconstruction_unweighted": reconstruction_unweighted_loss,
        "reconstruction_total": reconstruction_loss,
        "local_loss": local_branch_loss if local_branch_loss is not None else batch.image.new_zeros(()),
        "global_align": global_align_loss,
        "visible_align": visible_align_loss,
        "semantic_soft": semantic_soft_loss,
        "concept_cls": concept_loss,
        "concept_consistency": concept_consistency_loss,
    }

    # Graph consistency loss (clinical graph V1 semantic sidecar regularization)
    graph_consistency_loss = batch.image.new_zeros(())
    graph_prototype_loss = batch.image.new_zeros(())
    graph_consistency_weight = float(config.losses.graph_consistency_weight)
    graph_consistency_metrics: dict[str, Any] = {}
    if (
        graph_consistency_weight > 0.0
        and config.clinical_graph is not None
        and config.clinical_graph.use_for_concept_consistency
        and config.clinical_graph.consistency_rules_path is not None
    ):
        _node_id_map = _build_graph_node_id_head_mapping(config)
        if semantic_output.graph_encoder_consumed and semantic_output.graph_encoder_output is not None:
            graph_consistency_loss, graph_consistency_metrics = compute_graph_encoder_consistency_loss(
                concept_logits=semantic_output.concept_logits,
                concept_valid_masks=batch.concept_valid_masks,
                consistency_rules_path=config.clinical_graph.consistency_rules_path,
                graph_encoder_output=semantic_output.graph_encoder_output,
                node_id_mapping=_node_id_map,
                node_observed_mask=batch.clinical_graph_v2_observed_mask,
            )
            if config.semantic.formal_p0b is None:
                graph_prototype_loss = compute_graph_prototype_alignment_loss(
                    concept_feature=semantic_output.concept_feature,
                    graph_prototypes=semantic_output.graph_encoder_output.graph_concept_prototypes,
                    concept_targets=batch.concept_targets,
                    concept_valid_masks=batch.concept_valid_masks,
                    prototype_projection=semantic_output.graph_prototype_projection,
                    prototype_alignment_metrics=graph_consistency_metrics,
                    warnings=warnings_list,
                )
        else:
            graph_consistency_loss, graph_consistency_metrics = compute_graph_encoder_consistency_loss(
                concept_logits=semantic_output.concept_logits,
                concept_valid_masks=batch.concept_valid_masks,
                consistency_rules_path=config.clinical_graph.consistency_rules_path,
                graph_encoder_output=None,
                node_id_mapping=_node_id_map,
                node_observed_mask=batch.clinical_graph_v2_observed_mask,
            )
    elif graph_consistency_weight <= 0.0:
        warnings_list.append("graph_consistency_disabled:graph_consistency_weight<=0.0")
    elif config.clinical_graph is None:
        warnings_list.append("graph_consistency_disabled:clinical_graph_missing")
    elif not config.clinical_graph.use_for_concept_consistency:
        warnings_list.append("graph_consistency_disabled:use_for_concept_consistency=false")
    elif config.clinical_graph.consistency_rules_path is None:
        warnings_list.append("graph_consistency_disabled:consistency_rules_path_missing")
    losses["graph_consistency"] = graph_consistency_loss + graph_prototype_loss
    losses["total"] = (
        config.losses.reconstruction_weight * dynamic_weights["reconstruction"] * losses["reconstruction_total"]
        + config.losses.global_align_weight * dynamic_weights["global_align"] * losses["global_align"]
        + config.losses.visible_align_weight * dynamic_weights["visible_align"] * losses["visible_align"]
        + config.losses.semantic_soft_weight * dynamic_weights["semantic_soft"] * losses["semantic_soft"]
        + config.losses.concept_loss_weight * dynamic_weights["concept_loss"] * losses["concept_cls"]
        + config.losses.concept_consistency_weight * dynamic_weights["concept_consistency"] * losses["concept_consistency"]
        + config.losses.graph_consistency_weight * dynamic_weights["graph_consistency"] * losses["graph_consistency"]
        + config.model.local_high_conf_branch_loss_weight * losses["local_loss"]
    )
    loss_weight_audit = build_loss_weight_audit(
        losses=losses,
        static_weights=loss_weight_static_config(config),
        dynamic_weights=dynamic_weights,
    )
    return LossComputationResult(
        losses=losses,
        semantic_valid_count=semantic_valid_count,
        semantic_skipped_batch_count=semantic_skipped_batch_count,
        concept_head_losses=concept_head_losses,
        concept_head_correct_counts=concept_head_correct_counts,
        concept_head_valid_label_counts=concept_head_valid_label_counts,
        concept_head_missing_label_counts=concept_head_missing_label_counts,
        dynamic_loss_weights=dynamic_weights,
        loss_weight_audit=loss_weight_audit,
        local_branch_metrics=local_branch_metrics or {
            "local_branch_used": False,
            "local_token_count": 0,
            "local_loss": 0.0,
            "local_high_conf_coverage": 0.0,
        },
        warnings=warnings_list,
        graph_consistency_metrics=graph_consistency_metrics,
    )
