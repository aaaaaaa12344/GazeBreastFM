from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from breast_pretrain.train.stage1_joint.dynamic_patch_utils import (
    build_dynamic_patch_sampling_priors,
    expand_patch_mask_to_spatial,
    resolve_patch_grid_from_batch,
    validate_patch_token_grid_consistency,
)
from breast_pretrain.train.stage1_joint.gaze_masking import build_masking_step_output
from breast_pretrain.train.stage1_joint.high_conf_sidecar import HighConfSidecarLoader
from breast_pretrain.train.stage1_joint.local_branch_forward import (
    run_local_high_conf_branch_step,
)
from breast_pretrain.train.stage1_joint.losses import compute_stage1_joint_losses
from breast_pretrain.train.stage1_joint.semantic_forward import (
    Stage1JointSemanticRuntime,
    forward_semantic,
)
from breast_pretrain.train.stage1_joint.ddp_components import unwrap_ddp
from breast_pretrain.train.stage1_joint.formal_runtime_safety import (
    aggregate_has_invalid_value,
    batch_failure_context,
    coordinate_failure,
    first_invalid_value,
    module_gradient_values,
    module_parameter_values,
    model_state_at_failure,
    raise_coordinated_failure,
    scaler_found_inf_after_unscale,
    should_recover_amp_overflow,
)
from breast_pretrain.train.stage1_joint.formal_diagnostic import (
    diagnostic_enabled,
    gradient_aggregate_stats,
    write_diagnostic_summary,
)
from breast_pretrain.train.stage1_joint.student_forward import forward_student
from breast_pretrain.train.stage1_joint.teacher_latents import (
    RECONSTRUCTION_SOURCE_SELF,
    RECONSTRUCTION_SOURCE_TEACHER_NPY,
    build_no_teacher_latent_batch,
    load_teacher_latent_batch,
)
from breast_pretrain.train.stage1_joint.types import (
    LossComputationResult,
    MaskingRuntimeState,
    MaskingStepOutput,
    SemanticForwardOutput,
    Stage1JointBatch,
    Stage1JointTrainerConfig,
    StudentForwardOutput,
    TeacherLatentBatch,
)


@dataclass(frozen=True)
class FormalTrainStepOutput:
    loss_result: LossComputationResult
    masking_output: MaskingStepOutput
    semantic_warnings: list[str]
    semantic_output: SemanticForwardOutput
    teacher_batch: TeacherLatentBatch
    clean_output: StudentForwardOutput
    masked_context_tokens: torch.Tensor
    amp_state: dict[str, Any]


def _autocast_context(device: torch.device, use_amp: bool):
    return torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda"))


def _build_teacher_batch(
    *,
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    device: torch.device,
    clean_patches_detached: torch.Tensor,
) -> TeacherLatentBatch:
    if config.semantic.reconstruction_teacher_source == RECONSTRUCTION_SOURCE_SELF:
        return TeacherLatentBatch(
            teacher_latent=clean_patches_detached,
            teacher_sources=[RECONSTRUCTION_SOURCE_SELF] * int(batch.image.shape[0]),
            missing_teacher_flags=[False] * int(batch.image.shape[0]),
        )
    if config.semantic.reconstruction_teacher_source == RECONSTRUCTION_SOURCE_TEACHER_NPY:
        return load_teacher_latent_batch(batch, config, device)
    raise ValueError(
        "Unsupported reconstruction_teacher_source: "
        f"{config.semantic.reconstruction_teacher_source!r}"
    )


def _assert_formal_gaze_output(
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    masking_output: MaskingStepOutput,
) -> None:
    if str(config.metadata.run_tier).strip() not in {
        "formal_production",
        "production_ready_candidate",
    }:
        return
    invalid_policies = {
        policy
        for policy in masking_output.mask_policy_used
        if policy in {"random_fallback", "adaptive_weak_qc"}
    }
    invalid_reasons = tuple(reason for reason in masking_output.fallback_reason if str(reason).strip())
    if invalid_policies or invalid_reasons:
        invalid_samples = [
            {
                "image_id": batch.image_ids[index],
                "dataset_id": batch.dataset_ids[index],
                "case_id": batch.case_ids[index],
                "gaze_source": batch.gaze_supervision_sources[index],
                "attention_map_path": batch.attention_map_paths[index],
                "high_conf_mask_path": batch.high_conf_mask_paths[index],
                "patch_gaze_weight_path": batch.patch_gaze_weight_paths[index],
                "fallback_reason": masking_output.fallback_reason[index],
                "mask_policy": masking_output.mask_policy_used[index],
            }
            for index in range(len(batch.image_ids))
            if masking_output.mask_policy_used[index] in {"random_fallback", "adaptive_weak_qc"}
            or str(masking_output.fallback_reason[index]).strip()
        ]
        raise RuntimeError(
            "Formal Stage 1 requires usable Stage0 gaze for every sample; "
            f"invalid mask policies={sorted(invalid_policies)}, "
            f"fallback_reasons={list(invalid_reasons)}, "
            f"rank={__import__('os').environ.get('RANK', 'unknown')}, "
            f"invalid_samples={invalid_samples}."
        )


def formal_train_step(
    *,
    config: Stage1JointTrainerConfig,
    batch: Stage1JointBatch,
    model: torch.nn.Module,
    semantic_runtime: Stage1JointSemanticRuntime,
    mask_regressor: torch.nn.Module,
    masking_state: MaskingRuntimeState,
    device: torch.device,
    step_index: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: Any | None = None,
    projection_metadata: dict[str, Any] | None = None,
    high_conf_sidecar: HighConfSidecarLoader | None = None,
    use_amp: bool | None = None,
    last_committed_global_step: int = 0,
    epoch: int = 0,
    sampler_position: dict[str, object] | None = None,
    consecutive_amp_overflow_recoveries: int = 0,
) -> FormalTrainStepOutput:
    """Run the single production Stage 1 training step.

    The formal trainer, safety gate, and integration tests call this function
    so that dual-view forward, masking, semantic/graph losses, MaskRegressor,
    AMP, backward, clipping, scaler, scheduler, and padding exclusion remain a
    single implementation.
    """
    effective_amp = bool(config.train.use_amp if use_amp is None else use_amp)
    active_scaler = scaler or torch.cuda.amp.GradScaler(enabled=False)

    formal_runtime = str(config.metadata.run_tier).strip() in {
        "formal_production",
        "production_ready_candidate",
    }
    diagnostic_active = formal_runtime and diagnostic_enabled(step_index)

    failure_context = batch_failure_context(
        batch,
        attempt_step=step_index,
        global_step=last_committed_global_step,
        epoch=epoch,
    )
    failure_context.update({
        "current_checkpoint_parent": __import__("os").environ.get("HSM_FORMAL_RESUME_CHECKPOINT"),
        "sampler_position": sampler_position or {},
        "old_scale": float(active_scaler.get_scale()),
        "new_scale": float(active_scaler.get_scale()),
        "local_nonfinite_grad": False,
        "found_inf_any_rank": False,
        "optimizer_step_executed": False,
        "formal_step_committed": False,
    })
    diagnostic_tensors: dict[str, Any] = {}
    diagnostic_internal_module_summaries: dict[str, Any] = {}

    def numerical_gate(
        boundary: str,
        named_values: dict[str, Any],
        *,
        positive_names: set[str] | None = None,
    ) -> None:
        local_invalid = aggregate_has_invalid_value(named_values, positive_names=positive_names)
        local_failure = (
            first_invalid_value(named_values, positive_names=positive_names)
            if local_invalid else None
        )
        if local_failure is not None:
            local_failure["source_context"] = dict(failure_context)
            if diagnostic_active:
                local_failure["model_state_at_failure"] = model_state_at_failure({
                    "model": model,
                    "semantic_branch": semantic_runtime.branch,
                    "mask_regressor": mask_regressor,
                })
        coordinated = coordinate_failure(local_failure, device)
        if coordinated is not None:
            failure_context["failure_boundary"] = boundary
            raise_coordinated_failure(
                output_dir=config.data.output_dir,
                context=failure_context,
                failure=coordinated,
            )

    def diagnostic_boundary_gate(
        boundary: str, named_values: dict[str, Any], *, positive_names: set[str] | None = None
    ) -> None:
        """Write accumulated compact summaries before the diagnostic-only gate."""
        if not diagnostic_active:
            return
        diagnostic_tensors.update({name: value for name, value in named_values.items() if value is not None})
        write_diagnostic_summary(
            output_dir=config.data.output_dir,
            context={**failure_context, "diagnostic_boundary": boundary, "gradient_finite": None},
            tensors=diagnostic_tensors,
            internal_module_summaries=diagnostic_internal_module_summaries,
        )
        numerical_gate(boundary, named_values, positive_names=positive_names)
    if formal_runtime and batch.valid_content_patch_mask is None:
        raise ValueError(
            "Formal Stage 1 requires the frozen Dataset Entry valid_content_patch_mask; "
            "runtime geometry reconstruction is not a formal fallback."
        )
    if formal_runtime:
        numerical_gate(
            "grad_scaler_state",
            {"grad_scaler_scale": float(active_scaler.get_scale())},
            positive_names={"grad_scaler_scale"},
        )
    if formal_runtime:
        patch_weight = batch.patch_gaze_weight
        if patch_weight is None:
            raise ValueError("Formal Stage 1 requires frozen patch_gaze_weight; zero fallback is forbidden.")
        if patch_weight.ndim != 2 or int(patch_weight.shape[0]) != int(batch.image.shape[0]):
            raise ValueError(
                "Formal Stage 1 patch_gaze_weight must have shape [B, N]; "
                f"got {tuple(patch_weight.shape)} for batch={int(batch.image.shape[0])}."
            )
        if not diagnostic_active:
            numerical_gate(
                "input",
                {
                    "input_patch_gaze_weight": patch_weight,
                    "input_patch_gaze_weight_mass": patch_weight.sum(dim=1),
                },
                positive_names={"input_patch_gaze_weight_mass"},
            )

    resolved = resolve_patch_grid_from_batch(
        image=batch.image,
        modalities=batch.modalities,
        patch_size=config.model.patch_size,
        image_size_by_modality=config.data.image_size_by_modality,
        image_ids=batch.image_ids,
        projection_metadata=projection_metadata,
        frozen_valid_content_patch_mask=batch.valid_content_patch_mask,
    )
    if formal_runtime and int(batch.patch_gaze_weight.shape[1]) != int(resolved.num_patches):
        raise ValueError(
            "Formal Stage 1 patch_gaze_weight token count mismatch: "
            f"weights={int(batch.patch_gaze_weight.shape[1])} runtime={int(resolved.num_patches)}."
        )
    dynamic_gaze, dynamic_high_conf = build_dynamic_patch_sampling_priors(
        batch_patch_gaze_weight=batch.patch_gaze_weight,
        batch_high_conf_patch_prior=batch.high_conf_patch_prior,
        resolved=resolved,
    )
    if high_conf_sidecar is not None:
        dynamic_high_conf = high_conf_sidecar.lookup(
            batch.image_ids,
            num_patches=resolved.num_patches,
            modalities=batch.modalities,
            patch_grid=resolved.patch_grid,
        ).to(device=device)

    masking_output = build_masking_step_output(
        config=config,
        batch=batch,
        device=device,
        num_patches=resolved.num_patches,
        step_index=step_index,
        runtime_state=masking_state,
        patch_gaze_scores=dynamic_gaze.to(device=device),
        high_conf_patch_scores=dynamic_high_conf.to(device=device),
        valid_content_patch_mask=resolved.valid_content_patch_mask.to(device=device),
    )
    _assert_formal_gaze_output(config, batch, masking_output)

    if formal_runtime:
        failure_context.update({
            "q_vis": float(masking_output.q_vis.mean().item()),
            "omega_rec": None,
            "omega_sem": None,
            "lr": float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None,
        })
        if not diagnostic_active:
            numerical_gate(
                "input",
                {
                    "input_image": batch.image,
                    "input_patch_gaze_weight": batch.patch_gaze_weight,
                    "input_high_conf_patch_prior": batch.high_conf_patch_prior,
                    "input_valid_content_patch_mask": batch.valid_content_patch_mask,
                },
            )

    visible_patch_mask = (~masking_output.patch_mask) & resolved.valid_content_patch_mask.to(
        device=device, dtype=torch.bool
    )
    visible_spatial_mask = expand_patch_mask_to_spatial(
        visible_patch_mask,
        resolved.patch_grid,
        config.model.patch_size,
    ).to(device=device)

    diagnostic_boundary_gate(
        "pre_forward_input",
        {
            "input_image": batch.image,
            "input_patch_gaze_weight": batch.patch_gaze_weight,
            "input_patch_gaze_weight_mass": batch.patch_gaze_weight.sum(dim=1),
            "input_high_conf_patch_prior": batch.high_conf_patch_prior,
            "input_valid_content_patch_mask": batch.valid_content_patch_mask,
            "dynamic_gaze": dynamic_gaze,
            "dynamic_high_conf": dynamic_high_conf,
            "patch_mask": masking_output.patch_mask,
            "visible_patch_mask": visible_patch_mask,
            "visible_spatial_mask": visible_spatial_mask,
        },
        positive_names={"input_patch_gaze_weight_mass"},
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with _autocast_context(device, effective_amp):
        if diagnostic_active:
            clean_output = forward_student(
                model,
                batch.image,
                modality_ids=batch.modality_ids,
                diagnostic_tensors=diagnostic_tensors,
                diagnostic_internal_summaries=diagnostic_internal_module_summaries,
                diagnostic_batch_context={
                    "image_id": batch.image_ids,
                    "case_id": batch.case_ids,
                    "dataset_id": batch.dataset_ids,
                    "modality": batch.modalities,
                },
            )
        else:
            clean_output = forward_student(model, batch.image, modality_ids=batch.modality_ids)
        # The real backbone may reuse its pooled [B,2048] output storage on a
        # subsequent masked-view forward. Clone the clean anchor at the
        # dual-view boundary so its autograd graph cannot be mutated in place;
        # the clean-view value and V6.1 loss semantics are unchanged.
        clean_global = clean_output.global_image_feature.clone()
        # Semantic alignment also consumes the clean anchor. Rebuild the
        # frozen output record so every downstream consumer shares the
        # protected clone rather than the backbone-owned pooled buffer.
        clean_output = StudentForwardOutput(
            patch_tokens=clean_output.patch_tokens,
            global_image_feature=clean_global,
            patch_grid=clean_output.patch_grid,
            num_patches=clean_output.num_patches,
        )
        clean_forward_boundary_tensors: dict[str, Any] = {
            "clean_encoder_output": clean_output.patch_tokens,
            "clean_pooled_global_feature": clean_global,
        }
        if diagnostic_active:
            clean_forward_boundary_tensors.update(diagnostic_tensors)
        diagnostic_boundary_gate("clean_student_forward", clean_forward_boundary_tensors)
        clean_patches_detached = clean_output.patch_tokens.detach()

        masked_image = batch.image * visible_spatial_mask.to(dtype=batch.image.dtype)
        diagnostic_boundary_gate("masked_input", {"masked_image": masked_image})
        masked_forward_internal_diagnostics: dict[str, Any] | None = None
        if diagnostic_active:
            masked_forward_internal_diagnostics = {}
            masked_output = forward_student(
                model,
                masked_image,
                modality_ids=batch.modality_ids,
                diagnostic_tensors=masked_forward_internal_diagnostics,
            )
        else:
            masked_output = forward_student(model, masked_image, modality_ids=batch.modality_ids)
        context_patches = masked_output.patch_tokens
        masked_forward_boundary_tensors: dict[str, Any] = {
            "masked_encoder_output": context_patches,
            "context_patches": context_patches,
            "masked_pooled_global_feature": masked_output.global_image_feature,
        }
        if diagnostic_active:
            assert masked_forward_internal_diagnostics is not None
            masked_forward_boundary_tensors.update({
                f"masked_{name}": value
                for name, value in masked_forward_internal_diagnostics.items()
            })
        diagnostic_boundary_gate("masked_student_forward", masked_forward_boundary_tensors)
        validate_patch_token_grid_consistency(context_patches, resolved)

        predicted_patch_tokens = mask_regressor(context_patches)
        diagnostic_boundary_gate(
            "mask_regressor_forward",
            {"predicted_patch_tokens": predicted_patch_tokens},
        )
        local_branch = getattr(unwrap_ddp(model), "local_high_conf_branch", None)
        local_branch_result = run_local_high_conf_branch_step(
            local_branch=local_branch,
            clean_image=batch.image,
            masked_image=masked_image,
            high_conf_patch_scores=dynamic_high_conf.to(device=device),
            patch_gaze_scores=dynamic_gaze.to(device=device),
            patch_grid=resolved.patch_grid,
            patch_size=config.model.patch_size,
            loss_weight=config.model.local_high_conf_branch_loss_weight,
        )
        teacher_batch = _build_teacher_batch(
            config=config,
            batch=batch,
            device=device,
            clean_patches_detached=clean_patches_detached,
        )
        semantic_output, semantic_warnings = forward_semantic(
            runtime=semantic_runtime,
            config=config,
            batch=batch,
            student_output=clean_output,
            masking_output=masking_output,
            device=device,
            visible_patch_tokens=context_patches,
            valid_content_patch_mask=resolved.valid_content_patch_mask.to(device=device),
            diagnostic=diagnostic_active,
        )
        loss_result = compute_stage1_joint_losses(
            config=config,
            batch=batch,
            student_output=StudentForwardOutput(
                patch_tokens=predicted_patch_tokens,
                global_image_feature=clean_global,
            ),
            teacher_batch=teacher_batch,
            semantic_output=semantic_output,
            masking_output=masking_output,
            valid_content_patch_mask=resolved.valid_content_patch_mask.to(device=device),
            local_branch_loss=local_branch_result.local_loss,
            local_branch_metrics=local_branch_result.metrics,
        )

    if formal_runtime:
        failure_context.update({
            "q_vis": float(masking_output.q_vis.mean().item()),
            "omega_rec": float(loss_result.dynamic_loss_weights.get("reconstruction", 1.0)),
            "omega_sem": float(loss_result.dynamic_loss_weights.get("semantic_soft", 1.0)),
            "lr": float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None,
        })
        if diagnostic_active:
            selected_positions = semantic_output.semantic_batch_positions
            soft_mixed_embedding = 0.5 * (
                semantic_output.global_image_embedding[selected_positions]
                + semantic_output.visible_image_embedding[selected_positions]
            )
            soft_case_embedding = semantic_output.case_embedding[selected_positions]
            soft_target_pre = semantic_output.semantic_target_matrix
            soft_target_post = torch.maximum(
                soft_target_pre.to(dtype=soft_mixed_embedding.dtype),
                torch.eye(soft_target_pre.shape[0], device=device, dtype=soft_mixed_embedding.dtype),
            )
            soft_target_post = soft_target_post / soft_target_post.sum(dim=1, keepdim=True).clamp_min(1e-6)
            diagnostic_tensors.update(semantic_output.diagnostic_tensors or {})
            diagnostic_tensors.update({
                "clean_encoder_output": clean_output.patch_tokens,
                "clean_pooled_global_feature": clean_global,
                "global_visual_embedding_post_normalize": semantic_output.global_image_embedding,
                "text_case_embedding_post_normalize": semantic_output.case_embedding,
                "global_contrastive_temperature": 0.07,
                "global_logits": semantic_output.global_image_embedding.float() @ semantic_output.case_embedding.float().transpose(0, 1) / 0.07,
                "L_global_raw": loss_result.losses["global_align"],
                "visible_embedding": semantic_output.visible_image_embedding,
                "L_visible_raw": loss_result.losses["visible_align"],
                "soft_mixed_embedding": soft_mixed_embedding,
                "soft_target_pre_normalization": soft_target_pre,
                "soft_target_post_normalization": soft_target_post,
                "soft_contrastive_temperature": 0.07,
                "soft_logits": soft_mixed_embedding.float() @ soft_case_embedding.float().transpose(0, 1) / 0.07,
                "L_soft_raw": loss_result.losses["semantic_soft"],
                "reconstruction_prediction": F.layer_norm(predicted_patch_tokens.float(), (int(predicted_patch_tokens.shape[-1]),), eps=1e-6),
                "reconstruction_target": F.layer_norm(teacher_batch.teacher_latent.detach().float(), (int(teacher_batch.teacher_latent.shape[-1]),), eps=1e-6),
                "L_rec_raw": loss_result.losses["reconstruction_total"],
            })
            diagnostic_boundary_gate(
                "semantic_and_loss",
                {
                    "global_image_embedding": semantic_output.global_image_embedding,
                    "visible_image_embedding": semantic_output.visible_image_embedding,
                    "text_case_embedding": semantic_output.case_embedding,
                    "soft_target": semantic_output.semantic_target_matrix,
                    "losses": loss_result.losses,
                },
            )
        numerical_gate(
            "forward_or_loss",
            {
                "clean_encoder_output": clean_output.patch_tokens,
                "clean_pooled_global_feature": clean_global,
                "masked_representation": context_patches,
                "reconstruction_prediction": predicted_patch_tokens,
                "reconstruction_target": teacher_batch.teacher_latent,
                "global_image_embedding": semantic_output.global_image_embedding,
                "visible_image_embedding": semantic_output.visible_image_embedding,
                "text_case_embedding": semantic_output.case_embedding,
                "soft_target": semantic_output.semantic_target_matrix,
                "losses": loss_result.losses,
            },
        )

    if optimizer is not None:
        # Keep DDP's static graph valid across modality/target-validity
        # buckets.  Some formal branches are legitimately zero for a bucket;
        # this exact-zero anchor preserves the loss and every true gradient
        # while making the trainable parameter participation set invariant.
        ddp_zero_anchor = batch.image.new_zeros(())
        for owned_module in (model, semantic_runtime.branch, mask_regressor):
            for parameter in owned_module.parameters():
                if parameter.requires_grad:
                    ddp_zero_anchor = ddp_zero_anchor + parameter.sum() * 0.0
        backward_loss = loss_result.losses["total"] + ddp_zero_anchor
        active_scaler.scale(backward_loss).backward()
        scale_before_step = float(active_scaler.get_scale())
        active_scaler.unscale_(optimizer)
        gradient_values = module_gradient_values({
            "model": model,
            "semantic_branch": semantic_runtime.branch,
            "mask_regressor": mask_regressor,
        })
        if formal_runtime:
            local_nonfinite_grad = aggregate_has_invalid_value(gradient_values)
            local_scaler_found_inf = scaler_found_inf_after_unscale(active_scaler, optimizer)
            local_failure = first_invalid_value(gradient_values) if local_nonfinite_grad else None
            if local_failure is None and local_scaler_found_inf:
                local_failure = {"tensor_name": "grad_scaler", "reason": "found_inf"}
            if local_failure is not None:
                local_failure["scaler_found_inf"] = local_scaler_found_inf
                local_failure["source_context"] = dict(failure_context)
                if diagnostic_active:
                    local_failure["model_state_at_failure"] = model_state_at_failure({
                        "model": model,
                        "semantic_branch": semantic_runtime.branch,
                        "mask_regressor": mask_regressor,
                    })
            if diagnostic_active:
                failure_context.update(gradient_aggregate_stats(gradient_values))
                write_diagnostic_summary(
                    output_dir=config.data.output_dir,
                    context=failure_context,
                    tensors=diagnostic_tensors,
                )
            coordinated = coordinate_failure(local_failure, device)
            if coordinated is not None:
                rank_failures = coordinated.get("rank_failures", [])
                found_inf_all_rank = bool(rank_failures) and all(
                    isinstance(item.get("failure"), dict)
                    and item["failure"].get("scaler_found_inf") is True
                    for item in rank_failures
                )
                recoverable_overflow = should_recover_amp_overflow(
                    effective_amp=effective_amp,
                    scaler_enabled=active_scaler.is_enabled(),
                    local_found_inf=local_scaler_found_inf,
                    found_inf_all_rank=found_inf_all_rank,
                    local_nonfinite_grad=local_nonfinite_grad,
                    failure_reason=str(coordinated.get("reason", "")),
                    consecutive_recoveries=consecutive_amp_overflow_recoveries,
                )
                active_scaler.update()
                failure_context.update({
                    "failure_boundary": (
                        "grad_scaler_found_inf"
                        if coordinated.get("reason") == "found_inf"
                        else "unscaled_gradient"
                    ),
                    "old_scale": scale_before_step,
                    "new_scale": float(active_scaler.get_scale()),
                    "local_nonfinite_grad": local_nonfinite_grad,
                    "found_inf_any_rank": True,
                })
                if recoverable_overflow:
                    optimizer.zero_grad(set_to_none=True)
                    return FormalTrainStepOutput(
                        loss_result=loss_result,
                        masking_output=masking_output,
                        semantic_warnings=semantic_warnings,
                        semantic_output=semantic_output,
                        teacher_batch=teacher_batch,
                        clean_output=clean_output,
                        masked_context_tokens=context_patches,
                        amp_state={
                            "old_scale": scale_before_step,
                            "new_scale": float(active_scaler.get_scale()),
                            "local_nonfinite_grad": local_nonfinite_grad,
                            "found_inf_any_rank": True,
                            "optimizer_step_executed": False,
                            "formal_step_committed": False,
                            "amp_overflow_recovered": True,
                        },
                    )
                raise_coordinated_failure(
                    output_dir=config.data.output_dir,
                    context=failure_context,
                    failure=coordinated,
                )
        if config.train.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(semantic_runtime.branch.parameters())
                + list(mask_regressor.parameters()),
                max_norm=config.train.grad_clip_norm,
            )
        active_scaler.step(optimizer)
        active_scaler.update()
        scale_after_step = float(active_scaler.get_scale())
        optimizer_step_was_skipped = (
            active_scaler.is_enabled() and scale_after_step < scale_before_step
        )
        if formal_runtime:
            failure_context.update({
                "old_scale": scale_before_step,
                "new_scale": scale_after_step,
                "optimizer_step_executed": not optimizer_step_was_skipped,
            })
            numerical_gate(
                "optimizer_step",
                module_parameter_values({
                    "model": model,
                    "semantic_branch": semantic_runtime.branch,
                    "mask_regressor": mask_regressor,
                }),
            )
            numerical_gate(
                "grad_scaler_state",
                {"grad_scaler_scale": scale_after_step},
                positive_names={"grad_scaler_scale"},
            )
        if scheduler is not None and (formal_runtime or not optimizer_step_was_skipped):
            scheduler.step()
        if formal_runtime:
            failure_context.update({
                "old_scale": scale_before_step,
                "new_scale": scale_after_step,
                "optimizer_step_executed": not optimizer_step_was_skipped,
                "formal_step_committed": not optimizer_step_was_skipped,
            })
            if diagnostic_active:
                write_diagnostic_summary(
                    output_dir=config.data.output_dir,
                    context=failure_context,
                    tensors=diagnostic_tensors,
                )
        amp_state = {
            "old_scale": scale_before_step,
            "new_scale": scale_after_step,
            "local_nonfinite_grad": False,
            "found_inf_any_rank": False,
            "optimizer_step_executed": not optimizer_step_was_skipped,
            "formal_step_committed": not optimizer_step_was_skipped,
        }
    else:
        amp_state = {
            "old_scale": float(active_scaler.get_scale()),
            "new_scale": float(active_scaler.get_scale()),
            "local_nonfinite_grad": False,
            "found_inf_any_rank": False,
            "optimizer_step_executed": False,
            "formal_step_committed": False,
        }

    return FormalTrainStepOutput(
        loss_result=loss_result,
        masking_output=masking_output,
        semantic_warnings=semantic_warnings,
        semantic_output=semantic_output,
        teacher_batch=teacher_batch,
        clean_output=clean_output,
        masked_context_tokens=context_patches,
        amp_state=amp_state,
    )


__all__ = ["FormalTrainStepOutput", "formal_train_step"]
