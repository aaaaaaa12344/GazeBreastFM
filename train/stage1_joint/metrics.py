from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from breast_pretrain.train.reproducibility import compute_mask_sequence_checksum


def _gaze_coverage(
    attention_tokens: torch.Tensor,
    patch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mass = attention_tokens.clamp_min(0.0)
    total_mass = attention_mass.sum(dim=1).clamp_min(1e-6)
    masked_coverage = (
        attention_mass * patch_mask.to(dtype=attention_mass.dtype)
    ).sum(dim=1) / total_mass
    visible_coverage = (
        attention_mass * (~patch_mask).to(dtype=attention_mass.dtype)
    ).sum(dim=1) / total_mass
    return masked_coverage, visible_coverage


class Stage1JointMetricLogger:
    def __init__(self) -> None:
        self.total_samples = 0
        self.semantic_soft_valid_sample_count = 0
        self.semantic_soft_skipped_batch_count = 0
        self.losses = {
            "reconstruction_gaze_weighted": 0.0,
            "reconstruction_unweighted": 0.0,
            "reconstruction_total": 0.0,
            "local_loss": 0.0,
            "global_align": 0.0,
            "visible_align": 0.0,
            "semantic_soft": 0.0,
            "concept_cls": 0.0,
            "concept_consistency": 0.0,
            "graph_consistency": 0.0,
            "total": 0.0,
        }
        self.masked_gaze_coverage_total = 0.0
        self.visible_gaze_coverage_total = 0.0
        self.teacher_source_counter: Counter[str] = Counter()
        self.missing_teacher_latent_count = 0
        self.warnings: list[str] = []
        self.prior_schema_versions: set[str] = set()
        self.gaze_supervision_source_counter: Counter[str] = Counter()
        self.modality_counter: Counter[str] = Counter()
        self.modality_gaze_source_counters: dict[str, Counter[str]] = {}
        self.per_modality_loss_sums: dict[str, dict[str, float]] = {}
        self.per_modality_sample_counts: dict[str, int] = {}
        self.per_modality_concept_valid_counts: dict[str, dict[str, int]] = {}
        self.per_modality_concept_correct_counts: dict[str, dict[str, float]] = {}
        self.usable_prior_count = 0
        self.review_prior_count = 0
        self.no_valid_prior_count = 0
        self.high_conf_patch_ratio_total = 0.0
        self.gaze_weight_mean_total = 0.0
        self.gaze_weight_std_total = 0.0
        self.mask_policy_counter: Counter[str] = Counter()
        self.fallback_reason_counter: Counter[str] = Counter()
        self.adaptive_gaze_quota_total = 0.0
        self.adaptive_random_fraction_total = 0.0
        self.high_conf_mask_quota_actual_total = 0.0
        self.random_mask_fraction_actual_total = 0.0
        self.total_salient_count_total = 0
        self.masked_salient_count_total = 0
        self.visible_salient_count_total = 0
        self.visible_salient_sample_count = 0
        self.q_vis_total = 0.0
        self.q_vis_min: float | None = None
        self.visible_salient_floor_violation_count = 0
        self.dynamic_weight_sums: dict[str, float] = {}
        self.dynamic_weight_mins: dict[str, float] = {}
        self.dynamic_weight_maxs: dict[str, float] = {}
        self.loss_weight_audit_sums: dict[str, dict[str, float]] = {}
        self.concept_head_loss_sums: dict[str, float] = {}
        self.concept_head_correct_counts: dict[str, float] = {}
        self.concept_head_valid_label_counts: dict[str, int] = {}
        self.concept_head_missing_label_counts: dict[str, int] = {}
        self.graph_rule_term_count_total = 0
        self.graph_rule_skipped_count_total = 0
        self.graph_rule_skipped_reason_counter: Counter[str] = Counter()
        self.graph_rule_nonzero_steps = 0
        self.graph_pair_prior_sum = 0.0
        self.graph_pair_prior_steps = 0
        self.graph_encoder_consumed_steps = 0
        self.graph_prototype_head_count_total = 0
        self.graph_prototype_aligned_head_count_total = 0
        self.graph_prototype_skipped_missing_label_count_total = 0
        self.graph_prototype_skipped_missing_projector_count_total = 0
        self.graph_prototype_skipped_projection_dim_count_total = 0
        self.conflict_aware_enabled = False
        self.conflict_aware_gaze_fallback_count = 0
        self.local_branch_used_count = 0
        self.local_token_count_total = 0
        self.local_loss_total = 0.0
        self.local_high_conf_coverage_total = 0.0
        self.local_roi_valid_count_total = 0

    def _warning_count(self, prefix: str) -> int:
        return sum(1 for item in self.warnings if str(item).startswith(prefix))

    def update(
        self,
        current_batch_size: int,
        losses: dict[str, torch.Tensor],
        attention_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        teacher_sources: list[str],
        missing_teacher_flags: list[bool],
        warnings: list[str],
        semantic_valid_count: int,
        semantic_skipped_batch_count: int,
        prior_schema_versions: tuple[str, ...],
        modalities: list[str],
        gaze_supervision_sources: list[str],
        prior_statuses: list[str],
        high_conf_tokens: torch.Tensor,
        patch_weights: torch.Tensor,
        dynamic_loss_weights: dict[str, float],
        concept_head_losses: dict[str, torch.Tensor],
        concept_head_correct_counts: dict[str, float],
        concept_head_valid_label_counts: dict[str, int],
        concept_head_missing_label_counts: dict[str, int],
        mask_policy_used: tuple[str, ...],
        adaptive_gaze_quota: torch.Tensor,
        adaptive_random_fraction: torch.Tensor,
        fallback_reason: tuple[str, ...],
        high_conf_mask_quota_actual: torch.Tensor | None = None,
        random_mask_fraction_actual: torch.Tensor | None = None,
        graph_consistency_metrics: dict[str, Any] | None = None,
        conflict_aware_enabled: bool = False,
        local_branch_metrics: dict[str, Any] | None = None,
        loss_weight_audit: dict[str, dict[str, float]] | None = None,
        total_salient_count: torch.Tensor | None = None,
        masked_salient_count: torch.Tensor | None = None,
        visible_salient_count: torch.Tensor | None = None,
        q_vis: torch.Tensor | None = None,
        visible_salient_floor_violation_count: torch.Tensor | None = None,
    ) -> None:
        masked_coverage, visible_coverage = _gaze_coverage(attention_tokens, patch_mask)
        self.total_samples += int(current_batch_size)
        self.semantic_soft_valid_sample_count += int(semantic_valid_count)
        self.semantic_soft_skipped_batch_count += int(semantic_skipped_batch_count)
        for key in self.losses:
            loss_value = losses.get(key)
            if loss_value is not None:
                self.losses[key] += float(loss_value.item()) * current_batch_size
        self.masked_gaze_coverage_total += float(masked_coverage.sum().item())
        self.visible_gaze_coverage_total += float(visible_coverage.sum().item())
        self.teacher_source_counter.update(teacher_sources)
        self.missing_teacher_latent_count += int(sum(bool(item) for item in missing_teacher_flags))
        self.warnings.extend(warnings)
        self.prior_schema_versions.update(prior_schema_versions)
        self.gaze_supervision_source_counter.update(
            str(item).strip() or "unspecified" for item in gaze_supervision_sources
        )
        self.modality_counter.update(str(item).strip() or "unknown" for item in modalities)
        for modality, source in zip(modalities, gaze_supervision_sources):
            normalized_modality = str(modality).strip() or "unknown"
            self.modality_gaze_source_counters.setdefault(normalized_modality, Counter())
            self.modality_gaze_source_counters[normalized_modality][str(source).strip() or "unspecified"] += 1
        for source, status in zip(gaze_supervision_sources, prior_statuses):
            normalized_status = str(status).strip().lower()
            normalized_source = str(source).strip().lower()
            if normalized_status in {"usable_prior", "pass", "accepted"}:
                self.usable_prior_count += 1
            elif normalized_status in {"review_prior", "warning", "pass_with_warnings"}:
                self.review_prior_count += 1
            elif normalized_source in {"no_gaze", ""}:
                self.no_valid_prior_count += 1
            elif normalized_status:
                self.review_prior_count += 1
            else:
                self.usable_prior_count += int(normalized_source in {"observed_gaze", "diffeye_generated_gaze"})
                self.no_valid_prior_count += int(normalized_source not in {"observed_gaze", "diffeye_generated_gaze"})
        self.high_conf_patch_ratio_total += float((high_conf_tokens > 0.0).to(dtype=torch.float32).mean().item()) * current_batch_size
        self.gaze_weight_mean_total += float(patch_weights.mean().item()) * current_batch_size
        self.gaze_weight_std_total += float(patch_weights.std(unbiased=False).item()) * current_batch_size
        self.mask_policy_counter.update(str(item) for item in mask_policy_used)
        self.fallback_reason_counter.update(
            str(item) for item in fallback_reason if str(item).strip()
        )
        self.adaptive_gaze_quota_total += float(adaptive_gaze_quota.sum().item())
        self.adaptive_random_fraction_total += float(adaptive_random_fraction.sum().item())
        if high_conf_mask_quota_actual is not None:
            self.high_conf_mask_quota_actual_total += float(high_conf_mask_quota_actual.detach().float().sum().item())
        if random_mask_fraction_actual is not None:
            self.random_mask_fraction_actual_total += float(random_mask_fraction_actual.detach().float().sum().item())
        if (
            total_salient_count is not None
            and masked_salient_count is not None
            and visible_salient_count is not None
            and q_vis is not None
            and visible_salient_floor_violation_count is not None
        ):
            total = total_salient_count.detach().to(dtype=torch.long)
            self.total_salient_count_total += int(total.sum().item())
            self.masked_salient_count_total += int(masked_salient_count.detach().to(dtype=torch.long).sum().item())
            self.visible_salient_count_total += int(visible_salient_count.detach().to(dtype=torch.long).sum().item())
            self.visible_salient_floor_violation_count += int(
                visible_salient_floor_violation_count.detach().to(dtype=torch.long).sum().item()
            )
            applicable_q_vis = q_vis.detach().to(dtype=torch.float32)[total > 0]
            if applicable_q_vis.numel() > 0:
                self.visible_salient_sample_count += int(applicable_q_vis.numel())
                self.q_vis_total += float(applicable_q_vis.sum().item())
                current_min = float(applicable_q_vis.min().item())
                self.q_vis_min = current_min if self.q_vis_min is None else min(self.q_vis_min, current_min)
        for key, value in dynamic_loss_weights.items():
            numeric = float(value)
            self.dynamic_weight_sums[key] = self.dynamic_weight_sums.get(key, 0.0) + numeric * current_batch_size
            self.dynamic_weight_mins[key] = min(numeric, self.dynamic_weight_mins.get(key, numeric))
            self.dynamic_weight_maxs[key] = max(numeric, self.dynamic_weight_maxs.get(key, numeric))
        for term_name, term_payload in (loss_weight_audit or {}).items():
            bucket = self.loss_weight_audit_sums.setdefault(term_name, {})
            for field_name in ("raw_loss", "static_weight", "dynamic_weight", "effective_weight", "weighted_loss"):
                bucket[field_name] = bucket.get(field_name, 0.0) + float(term_payload.get(field_name, 0.0)) * current_batch_size
        for head_name, head_loss in concept_head_losses.items():
            valid_count = int(concept_head_valid_label_counts.get(head_name, 0))
            missing_count = int(concept_head_missing_label_counts.get(head_name, 0))
            self.concept_head_loss_sums[head_name] = self.concept_head_loss_sums.get(head_name, 0.0) + (
                float(head_loss.item()) * valid_count
            )
            self.concept_head_correct_counts[head_name] = self.concept_head_correct_counts.get(head_name, 0.0) + float(
                concept_head_correct_counts.get(head_name, 0.0)
            )
            self.concept_head_valid_label_counts[head_name] = self.concept_head_valid_label_counts.get(head_name, 0) + valid_count
            self.concept_head_missing_label_counts[head_name] = self.concept_head_missing_label_counts.get(head_name, 0) + missing_count

        self.conflict_aware_enabled = self.conflict_aware_enabled or bool(conflict_aware_enabled)
        if conflict_aware_enabled:
            for w in warnings:
                if "conflict_aware_fallback_base_weights" in str(w):
                    self.conflict_aware_gaze_fallback_count += 1

        if graph_consistency_metrics is not None:
            self.graph_rule_term_count_total += int(graph_consistency_metrics.get("graph_rule_term_count", 0))
            self.graph_rule_skipped_count_total += int(graph_consistency_metrics.get("graph_rule_skipped_count", 0))
            self.graph_rule_skipped_reason_counter.update(
                {
                    str(reason): int(count)
                    for reason, count in (
                        graph_consistency_metrics.get("graph_rule_skipped_reason_counts", {}) or {}
                    ).items()
                }
            )
            if (
                int(graph_consistency_metrics.get("graph_rule_term_count", 0)) > 0
                and float(losses["graph_consistency"].detach().item()) > 0.0
            ):
                self.graph_rule_nonzero_steps += 1
            pair_prior_mean = graph_consistency_metrics.get("graph_pair_prior_mean", 0.0)
            if pair_prior_mean:
                self.graph_pair_prior_sum += float(pair_prior_mean)
                self.graph_pair_prior_steps += 1
            if graph_consistency_metrics.get("graph_encoder_consumed"):
                self.graph_encoder_consumed_steps += 1
            self.graph_prototype_head_count_total += int(graph_consistency_metrics.get("graph_prototype_head_count", 0))
            self.graph_prototype_aligned_head_count_total += int(
                graph_consistency_metrics.get("graph_prototype_aligned_head_count", 0)
            )
            self.graph_prototype_skipped_missing_label_count_total += int(
                graph_consistency_metrics.get("graph_prototype_skipped_missing_label_count", 0)
            )
            self.graph_prototype_skipped_missing_projector_count_total += int(
                graph_consistency_metrics.get("graph_prototype_skipped_missing_projector_count", 0)
            )
            self.graph_prototype_skipped_projection_dim_count_total += int(
                graph_consistency_metrics.get("graph_prototype_skipped_projection_dim_count", 0)
            )

        if local_branch_metrics is not None:
            if bool(local_branch_metrics.get("local_branch_used", False)):
                self.local_branch_used_count += current_batch_size
            self.local_token_count_total += int(local_branch_metrics.get("local_token_count", 0)) * current_batch_size
            self.local_loss_total += float(local_branch_metrics.get("local_loss", 0.0)) * current_batch_size
            self.local_high_conf_coverage_total += (
                float(local_branch_metrics.get("local_high_conf_coverage", 0.0)) * current_batch_size
            )
            self.local_roi_valid_count_total += int(local_branch_metrics.get("local_roi_valid_count", 0))

        for idx, modality in enumerate(modalities):
            norm_mod = str(modality).strip() or "unknown"
            self.per_modality_sample_counts[norm_mod] = self.per_modality_sample_counts.get(norm_mod, 0) + 1
            self.per_modality_loss_sums.setdefault(norm_mod, {})
            for loss_key in ("reconstruction_total", "local_loss", "global_align", "visible_align", "semantic_soft", "concept_cls", "concept_consistency", "total"):
                loss_val = float(losses.get(loss_key, losses.get("total", 0.0)).item()) if loss_key in losses else 0.0
                self.per_modality_loss_sums[norm_mod][loss_key] = self.per_modality_loss_sums[norm_mod].get(loss_key, 0.0) + loss_val
            self.per_modality_concept_valid_counts.setdefault(norm_mod, {})
            self.per_modality_concept_correct_counts.setdefault(norm_mod, {})
            for head_name in concept_head_valid_label_counts:
                self.per_modality_concept_valid_counts[norm_mod][head_name] = self.per_modality_concept_valid_counts[norm_mod].get(head_name, 0) + int(concept_head_valid_label_counts.get(head_name, 0))
                self.per_modality_concept_correct_counts[norm_mod][head_name] = self.per_modality_concept_correct_counts[norm_mod].get(head_name, 0.0) + float(concept_head_correct_counts.get(head_name, 0.0))

    def _build_concept_head_metrics(self) -> dict[str, dict[str, float | int]]:
        concept_head_metrics: dict[str, dict[str, float | int]] = {}
        all_heads = set(self.concept_head_loss_sums) | set(self.concept_head_valid_label_counts) | set(self.concept_head_missing_label_counts)
        for head_name in sorted(all_heads):
            valid_count = int(self.concept_head_valid_label_counts.get(head_name, 0))
            loss_sum = float(self.concept_head_loss_sums.get(head_name, 0.0))
            correct_count = float(self.concept_head_correct_counts.get(head_name, 0.0))
            concept_head_metrics[head_name] = {
                "loss": (loss_sum / float(valid_count)) if valid_count > 0 else 0.0,
                "accuracy": (correct_count / float(valid_count)) if valid_count > 0 else 0.0,
                "valid_label_count": valid_count,
                "missing_label_count": int(self.concept_head_missing_label_counts.get(head_name, 0)),
                "observed_count": valid_count,
                "ignored_count": int(self.concept_head_missing_label_counts.get(head_name, 0)),
            }
        return concept_head_metrics

    def _dynamic_weight_summary(self, sample_normalizer: float) -> dict[str, object]:
        return {
            key: {
                "mean": self.dynamic_weight_sums.get(key, 0.0) / sample_normalizer,
                "min": self.dynamic_weight_mins.get(key, 1.0),
                "max": self.dynamic_weight_maxs.get(key, 1.0),
            }
            for key in sorted(self.dynamic_weight_sums)
        }

    def _loss_weight_audit_summary(self, sample_normalizer: float) -> dict[str, dict[str, float]]:
        return {
            term_name: {
                field_name: float(value) / sample_normalizer
                for field_name, value in sorted(field_sums.items())
            }
            for term_name, field_sums in sorted(self.loss_weight_audit_sums.items())
        }

    def build_summary_payload(
        self,
        *,
        config_path: str,
        output_dir: str,
        dataset_sample_count: int,
        teacher_latent_availability: dict[str, object],
        teacher_latent_source_summary: dict[str, int],
        mask_checksum: str,
        model_init_checksum: str,
        checkpoint_path: str | None,
        eval_payload: dict[str, object] | None,
        used_patch_masks: list[torch.Tensor],
        normalized_gaze_loss_mode: str,
        configured_mask_strategy: str,
        normalized_mask_strategy: str,
        mask_prior_mode: str,
        reconstruction_teacher_source: str,
        text_dim: int,
        align_dim: int,
        latent_dim: int,
        active_concept_heads: tuple[str, ...],
        pending_concept_heads: tuple[str, ...],
        require_teacher_latents: bool,
        loss_weights: dict[str, object],
        visual_encoder: dict[str, object],
        run_metadata: dict[str, object],
        checkpoint_policy_applied: dict[str, object],
        resume_info: dict[str, object],
        graph_activation_audit: dict[str, Any] | None = None,
        graph_activation_audit_path: str | None = None,
        graph_activation_audit_status: str | None = None,
        concept_head_activation_audit: dict[str, Any] | None = None,
        concept_head_activation_audit_path: str | None = None,
        concept_head_activation_audit_status: str | None = None,
    ) -> dict[str, object]:
        sample_normalizer = float(self.total_samples) if self.total_samples > 0 else 1.0
        final_mask_checksum = (
            compute_mask_sequence_checksum(used_patch_masks)
            if used_patch_masks
            else mask_checksum
        )
        reconstruction_runtime_mode = reconstruction_teacher_source
        if reconstruction_teacher_source == "self_masked_reconstruction":
            reconstruction_loss_definition = (
                "patch-weighted gaze-guided masked visual reconstruction "
                "(no-teacher mainline: image-patch normalised projection target)"
            )
        elif reconstruction_teacher_source == "teacher_latent_npy":
            reconstruction_loss_definition = (
                "patch-weighted gaze-guided masked visual reconstruction "
                "(legacy/ablation: teacher latent .npy target)"
            )
        else:
            reconstruction_loss_definition = (
                f"patch-weighted gaze-guided masked visual reconstruction "
                f"(unknown source: {reconstruction_teacher_source})"
            )

        graph_audit_metrics = graph_activation_audit or {}
        graph_rule_summary = graph_audit_metrics.get("rule_summary", {}) if isinstance(graph_audit_metrics, dict) else {}
        concept_head_audit = concept_head_activation_audit or {}
        concept_head_reports = (
            concept_head_audit.get("heads", {}) if isinstance(concept_head_audit, dict) else {}
        )
        payload = {
            "config_path": config_path,
            "output_dir": output_dir,
            "graph_activation_audit_path": graph_activation_audit_path,
            "graph_activation_audit_status": (
                graph_activation_audit_status
                or graph_audit_metrics.get("graph_activation_audit_status")
                or "not_generated"
            ),
            "concept_head_activation_audit_path": concept_head_activation_audit_path,
            "concept_head_activation_audit_status": (
                concept_head_activation_audit_status
                or concept_head_audit.get("concept_head_activation_audit_status")
                or "not_generated"
            ),
            "trial_active_concept_heads": sorted(
                head
                for head, report in concept_head_reports.items()
                if report.get("configured_active")
                and report.get("recommended_action") == "active_confirmed_supervised"
            ),
            "eligible_but_inactive_concept_heads": sorted(
                head
                for head, report in concept_head_reports.items()
                if not report.get("configured_active")
                and report.get("recommended_action") == "eligible_trial_active"
            ),
            "pending_no_confirmed_label_heads": sorted(
                head
                for head, report in concept_head_reports.items()
                if report.get("configured_pending") and int(report.get("confirmed", 0)) <= 0
            ),
            "summary_generated_by": "train_stage1_joint_v5.py",
            "trainer_entrypoint": "formal_stage1_joint_v5",
            "run_tier": run_metadata["run_tier"],
            "model_role": run_metadata["model_role"],
            "compliance_status": run_metadata["compliance_status"],
            "known_limitations": run_metadata["known_limitations"],
            "allowed_claims": run_metadata["allowed_claims"],
            "forbidden_claims": run_metadata["forbidden_claims"],
            "v5_mainline_note": (
                "V5 mainline: single-stage gaze-guided masked visual reconstruction + semantic alignment. "
                "Teacher latent supervision is optional/legacy/ablation only. "
                "BI-RADS graph / clinical concepts are semantic priors and regularization signals, "
                "not independent graph pretraining targets. "
                "DiffEye-derived gaze is a weak spatial attention prior, not physician ground truth."
            ),
            "dataset_sample_count": int(dataset_sample_count),
            "total_optimized_samples": int(self.total_samples),
            "semantic_soft_valid_sample_count": int(self.semantic_soft_valid_sample_count),
            "semantic_soft_skipped_batch_count": int(self.semantic_soft_skipped_batch_count),
            "visual_dim": int(latent_dim),
            **visual_encoder,
            "text_dim": int(text_dim),
            "align_dim": int(align_dim),
            "semantic_image_feature_source": "student_patch_tokens_and_global_feature",
            "reconstruction_teacher_source": reconstruction_teacher_source,
            "reconstruction_runtime_mode": reconstruction_runtime_mode,
            "reconstruction_loss_definition": reconstruction_loss_definition,
            "loss_reconstruction_gaze_weighted": self.losses["reconstruction_gaze_weighted"] / sample_normalizer,
            "loss_reconstruction_unweighted": self.losses["reconstruction_unweighted"] / sample_normalizer,
            "loss_reconstruction_total": self.losses["reconstruction_total"] / sample_normalizer,
            "loss_local": self.losses["local_loss"] / sample_normalizer,
            "loss_global_align": self.losses["global_align"] / sample_normalizer,
            "loss_visible_align": self.losses["visible_align"] / sample_normalizer,
            "loss_semantic_soft": self.losses["semantic_soft"] / sample_normalizer,
            "loss_concept_cls": self.losses["concept_cls"] / sample_normalizer,
            "loss_concept_consistency": self.losses["concept_consistency"] / sample_normalizer,
            "loss_graph_consistency": self.losses["graph_consistency"] / sample_normalizer,
            "loss_total": self.losses["total"] / sample_normalizer,
            "concept_prototype_scope": "configurable_multitask_clinical_heads",
            "concept_head_fields_active": list(active_concept_heads),
            "concept_head_fields_pending": list(pending_concept_heads),
            "concept_head_metrics": self._build_concept_head_metrics(),
            "concept_head_activation_decision": graph_audit_metrics.get("concept_head_activation_decision", {}),
            "concept_consistency_mode": "prior_masked_semantic_similarity_consistency",
            "concept_consistency_note": (
                "Uses BI-RADS prior availability to gate pairwise semantic consistency on image embeddings only; "
                "does not perform graph-image or region-node alignment."
            ),
            "local_branch_used": self.local_branch_used_count > 0,
            "local_branch_used_count": int(self.local_branch_used_count),
            "local_token_count": self.local_token_count_total / sample_normalizer,
            "local_loss": self.local_loss_total / sample_normalizer,
            "local_high_conf_coverage": self.local_high_conf_coverage_total / sample_normalizer,
            "local_roi_valid_count": int(self.local_roi_valid_count_total),
            "masked_gaze_coverage_mean": self.masked_gaze_coverage_total / sample_normalizer,
            "visible_gaze_coverage_mean": self.visible_gaze_coverage_total / sample_normalizer,
            "gaze_supervision_source_distribution": dict(self.gaze_supervision_source_counter),
            "modality_embedding_enabled": bool(loss_weights.get("modality_embedding_enabled", False)),
            "modality_vocab": list(loss_weights.get("modality_vocab", [])),
            "modality_distribution": dict(self.modality_counter),
            "per_modality_loss": {
                modality: {
                    loss_key: self.per_modality_loss_sums.get(modality, {}).get(loss_key, 0.0) / max(1, self.per_modality_sample_counts.get(modality, 0))
                    for loss_key in ("reconstruction_total", "local_loss", "global_align", "visible_align", "semantic_soft", "concept_cls", "concept_consistency", "total")
                }
                for modality in sorted(self.modality_counter)
            },
            "per_modality_concept_coverage": {
                modality: {
                    head_name: {
                        "valid_count": self.per_modality_concept_valid_counts.get(modality, {}).get(head_name, 0),
                        "correct": self.per_modality_concept_correct_counts.get(modality, {}).get(head_name, 0.0),
                    }
                    for head_name in sorted(set(self.per_modality_concept_valid_counts.get(modality, {})) | set(self.per_modality_concept_correct_counts.get(modality, {})))
                }
                for modality in sorted(self.modality_counter)
            },
            "per_modality_summary": {
                modality: {
                    "sample_count": int(count),
                    "gaze_source_distribution": dict(self.modality_gaze_source_counters.get(modality, Counter())),
                    "loss": dict(self.per_modality_loss_sums.get(modality, {})),
                    "concept_coverage": dict(self.per_modality_concept_valid_counts.get(modality, {})),
                }
                for modality, count in sorted(self.modality_counter.items())
            },
            "usable_prior_count": int(self.usable_prior_count),
            "review_prior_count": int(self.review_prior_count),
            "no_valid_prior_count": int(self.no_valid_prior_count),
            "high_conf_patch_ratio": self.high_conf_patch_ratio_total / sample_normalizer,
            "gaze_weight_mean": self.gaze_weight_mean_total / sample_normalizer,
            "gaze_weight_std": self.gaze_weight_std_total / sample_normalizer,
            "mask_policy_used": dict(self.mask_policy_counter),
            "adaptive_gaze_quota": self.adaptive_gaze_quota_total / sample_normalizer,
            "adaptive_random_fraction": self.adaptive_random_fraction_total / sample_normalizer,
            "high_conf_mask_quota_actual": self.high_conf_mask_quota_actual_total / sample_normalizer,
            "random_mask_fraction_actual": self.random_mask_fraction_actual_total / sample_normalizer,
            "total_salient_count": int(self.total_salient_count_total),
            "masked_salient_count": int(self.masked_salient_count_total),
            "visible_salient_count": int(self.visible_salient_count_total),
            "visible_salient_floor_violation_count": int(self.visible_salient_floor_violation_count),
            "q_vis_min": self.q_vis_min,
            "q_vis_mean": (
                self.q_vis_total / self.visible_salient_sample_count
                if self.visible_salient_sample_count > 0
                else None
            ),
            "fallback_reason": dict(self.fallback_reason_counter),
            "masked_gaze_coverage": self.masked_gaze_coverage_total / sample_normalizer,
            "visible_gaze_coverage": self.visible_gaze_coverage_total / sample_normalizer,
            "gaze_prior_qc_metrics": {
                "usable_prior_count": int(self.usable_prior_count),
                "review_prior_count": int(self.review_prior_count),
                "no_valid_prior_count": int(self.no_valid_prior_count),
                "high_conf_patch_ratio": self.high_conf_patch_ratio_total / sample_normalizer,
            },
            "dynamic_loss_weight_summary": self._dynamic_weight_summary(sample_normalizer),
            "loss_weight_audit_summary": self._loss_weight_audit_summary(sample_normalizer),
            "dynamic_loss_weighting_enabled": bool(self.conflict_aware_enabled),
            "dynamic_loss_weighting_gaze_fallback_count": int(self.conflict_aware_gaze_fallback_count),
            "reconstruction_weight_effective": (
                float(loss_weights.get("reconstruction_weight", 1.0))
                * self.dynamic_weight_sums.get("reconstruction", sample_normalizer) / sample_normalizer
            ),
            "global_align_weight_effective": (
                float(loss_weights.get("global_align_weight", 1.0))
                * self.dynamic_weight_sums.get("global_align", sample_normalizer) / sample_normalizer
            ),
            "visible_align_weight_effective": (
                float(loss_weights.get("visible_align_weight", 1.0))
                * self.dynamic_weight_sums.get("visible_align", sample_normalizer) / sample_normalizer
            ),
            "semantic_soft_weight_effective": (
                float(loss_weights.get("semantic_soft_weight", 1.0))
                * self.dynamic_weight_sums.get("semantic_soft", sample_normalizer) / sample_normalizer
            ),
            "concept_weight_effective": (
                float(loss_weights.get("concept_loss_weight", 1.0))
                * self.dynamic_weight_sums.get("concept_loss", sample_normalizer) / sample_normalizer
            ),
            "concept_consistency_weight_effective": (
                float(loss_weights.get("concept_consistency_weight", 1.0))
                * self.dynamic_weight_sums.get("concept_consistency", sample_normalizer) / sample_normalizer
            ),
            "graph_consistency_weight_effective": float(loss_weights.get("graph_consistency_weight", 0.0)),
            "require_teacher_latents": bool(require_teacher_latents),
            "gaze_loss_mode": normalized_gaze_loss_mode,
            "configured_mask_strategy": configured_mask_strategy,
            "mask_strategy": normalized_mask_strategy,
            "mask_prior_mode": mask_prior_mode,
            "mask_checksum": final_mask_checksum,
            "model_init_checksum": model_init_checksum,
            "teacher_latent_availability": teacher_latent_availability,
            "teacher_latent_source_summary": teacher_latent_source_summary,
            "missing_effective_report_count": self._warning_count("missing_effective_report:"),
            "prompt_cache_miss_count": self._warning_count("prompt_embedding_cache_miss:"),
            "prior_schema_versions": sorted(self.prior_schema_versions),
            "checkpoint_policy_applied": checkpoint_policy_applied,
            "resume_info": resume_info,
            "checkpoint_path": checkpoint_path,
            "last_checkpoint_path": checkpoint_path,
            "eval_output_path": (
                str(eval_payload.get("output_path"))
                if isinstance(eval_payload, dict) and eval_payload.get("output_path") is not None
                else None
            ),
            "final_eval": eval_payload,
            "graph_node_status_counts": graph_audit_metrics.get("graph_node_status_counts", {}),
            "active_graph_node_count": graph_audit_metrics.get("active_graph_node_count", 0),
            "graph_node_activation_by_modality": graph_audit_metrics.get("graph_node_activation_by_modality", {}),
            "graph_consistency_metrics": {
                "graph_rule_term_count": self.graph_rule_term_count_total,
                "graph_rule_skipped_count": self.graph_rule_skipped_count_total,
                "graph_rule_skipped_reason_counts": dict(self.graph_rule_skipped_reason_counter),
                "graph_rule_nonzero_steps": int(self.graph_rule_nonzero_steps),
                "graph_pair_prior_mean": self.graph_pair_prior_sum / max(self.graph_pair_prior_steps, 1),
                "graph_encoder_consumed_steps": self.graph_encoder_consumed_steps,
                "graph_prototype_head_count": self.graph_prototype_head_count_total,
                "graph_prototype_aligned_head_count": self.graph_prototype_aligned_head_count_total,
                "graph_prototype_skipped_missing_label_count": self.graph_prototype_skipped_missing_label_count_total,
                "graph_prototype_skipped_missing_projector_count": self.graph_prototype_skipped_missing_projector_count_total,
                "graph_prototype_skipped_projection_dim_count": self.graph_prototype_skipped_projection_dim_count_total,
                "audit_graph_rule_term_count": int(graph_rule_summary.get("graph_rule_term_count", 0)),
                "audit_graph_rule_skipped_count": int(graph_rule_summary.get("graph_rule_skipped_count", 0)),
                "audit_graph_rule_skipped_reason_counts": graph_rule_summary.get("graph_rule_skipped_reason_counts", {}),
            },
            "warnings": self.warnings,
        }
        payload.update(loss_weights)
        return payload
