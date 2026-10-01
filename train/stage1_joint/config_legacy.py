from __future__ import annotations

"""Legacy config loader extracted from config.py to keep that file under the size threshold."""

from pathlib import Path
from typing import Any

from breast_pretrain.train.stage1_joint.types import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LossConfig,
    MaskingConfig,
    ModelConfig,
    ReproducibilityConfig,
    SemanticConfig,
    Stage1JointTrainerConfig,
    TrainConfig,
)

from .config import (
    _resolve_path,
    _resolve_optional_path,
    _as_bool,
    _as_optional_string,
    _as_optional_positive_int,
    _as_tuple_of_strings,
    _parse_vision_encoder_name,
    _build_reference_paths,
    _build_run_metadata,
    _parse_conflict_aware_loss,
    _parse_concept_head_weights,
    _concept_head_output_dims,
)


def load_legacy_config(
    raw_config: dict[str, Any],
    config_path: Path,
) -> Stage1JointTrainerConfig:
    project_root = _resolve_path(raw_config["project_root"], config_path.parent)
    train_smoke = raw_config.get("train_smoke") or {}
    stage1_joint = raw_config.get("stage1_joint") or {}
    if not isinstance(train_smoke, dict) or not isinstance(stage1_joint, dict):
        raise ValueError("Legacy config must contain mapping blocks train_smoke and stage1_joint.")

    active_heads = ("view", "laterality")
    concept_head_weights = _parse_concept_head_weights(
        stage1_joint.get("concept_head_weights"),
        field_name="stage1_joint.concept_head_weights",
        active_heads=active_heads,
        default_all_zero=False,
    )
    concept_consistency_head_weights = _parse_concept_head_weights(
        stage1_joint.get("concept_consistency_head_weights"),
        field_name="stage1_joint.concept_consistency_head_weights",
        active_heads=active_heads,
        default_all_zero=True,
    )

    return Stage1JointTrainerConfig(
        config_path=config_path,
        metadata=_build_run_metadata(raw_config),
        data=DataConfig(
            project_root=project_root,
            image_manifest_path=_resolve_path(raw_config["image_manifest_path"], project_root),
            attention_map_dir=_resolve_optional_path(raw_config.get("attention_map_dir"), project_root),
            teacher_latent_dir=_resolve_optional_path(raw_config.get("teacher_latent_dir"), project_root),
            text_prompt_path=_resolve_optional_path(raw_config.get("text_prompt_path"), project_root),
            image_size=int(raw_config["image_size"]),
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
        ),
        references=_build_reference_paths(raw_config, project_root),
        model=ModelConfig(
            patch_size=int(train_smoke["patch_size"]),
            latent_dim=int(train_smoke["latent_dim"]),
            text_dim=int(stage1_joint["text_dim"]),
            align_dim=int(stage1_joint["align_dim"]),
            vision_encoder_name=_parse_vision_encoder_name(
                train_smoke.get("vision_encoder_name")
            ),
            pretrained_model_path=_as_optional_string(
                train_smoke.get("pretrained_model_path")
            ),
            pretrained_weight_path=(
                str(_resolve_optional_path(
                    train_smoke.get("pretrained_weight_path", train_smoke.get("pretrained_model_path")),
                    project_root,
                ))
                if _resolve_optional_path(
                    train_smoke.get("pretrained_weight_path", train_smoke.get("pretrained_model_path")),
                    project_root,
                )
                is not None
                else None
            ),
            allow_missing_pretrained_fallback=_as_bool(
                train_smoke.get("allow_missing_pretrained_fallback", False),
                "train_smoke.allow_missing_pretrained_fallback",
            ),
            freeze_backbone=_as_bool(
                train_smoke.get("freeze_backbone", False),
                "train_smoke.freeze_backbone",
            ),
            output_patch_dim=_as_optional_positive_int(
                train_smoke.get("output_patch_dim", train_smoke["latent_dim"]),
                "train_smoke.output_patch_dim",
            ),
        ),
        masking=MaskingConfig(
            mask_ratio=float(train_smoke["mask_ratio"]),
            gaze_loss_mode=str(train_smoke.get("gaze_loss_mode", "soft_attention_plus_high_conf")),
            gaze_weight_alpha=float(train_smoke.get("gaze_weight_alpha", 1.0)),
            mask_strategy=str(train_smoke.get("mask_strategy", "random")),
            gaze_mask_sampling_alpha=float(train_smoke.get("gaze_mask_sampling_alpha", 0.7)),
            high_conf_mask_quota=float(train_smoke.get("high_conf_mask_quota", 0.6)),
            min_random_mask_fraction=float(train_smoke.get("min_random_mask_fraction", 0.3)),
            mask_sampling_temperature=float(train_smoke.get("mask_sampling_temperature", 1.0)),
            gaze_mask_eps=float(train_smoke.get("gaze_mask_eps", 1e-6)),
        ),
        semantic=SemanticConfig(
            text_backend=str(stage1_joint.get("text_backend", "")).strip().lower(),
            prompt_embedding_path=_resolve_path(stage1_joint["prompt_embedding_path"], project_root),
            semantic_soft_label_path=_resolve_path(stage1_joint["semantic_soft_label_path"], project_root),
            semantic_manifest_path=_resolve_path(stage1_joint["semantic_manifest_path"], project_root),
            semantic_soft_label_format="dense",
            semantic_soft_label_topk_path=None,
            semantic_soft_label_symmetrization="max",
            birads_prior_manifest_path=_resolve_path(
                stage1_joint["birads_prior_manifest_path"],
                project_root,
            ),
            high_conf_weight_alpha=float(stage1_joint.get("high_conf_weight_alpha", 1.0)),
            reconstruction_teacher_source=str(
                stage1_joint.get("reconstruction_teacher_source", "self_masked_reconstruction")
            ).strip()
            or "self_masked_reconstruction",
            active_concept_heads=active_heads,
            pending_concept_heads=("density", "finding", "birads"),
            concept_head_output_dims=_concept_head_output_dims(active_heads),
        ),
        losses=LossConfig(
            reconstruction_weight=float(stage1_joint.get("reconstruction_weight", 1.0)),
            global_align_weight=float(stage1_joint.get("global_align_weight", 1.0)),
            visible_align_weight=float(stage1_joint.get("visible_align_weight", 1.0)),
            semantic_soft_weight=float(stage1_joint.get("semantic_soft_weight", 1.0)),
            concept_loss_weight=float(stage1_joint.get("concept_loss_weight", 1.0)),
            concept_consistency_weight=float(
                stage1_joint.get("concept_consistency_weight", 1.0)
            ),
            graph_consistency_weight=float(stage1_joint.get("graph_consistency_weight", 0.0)),
            concept_head_weights=concept_head_weights,
            concept_consistency_head_weights=concept_consistency_head_weights,
            **_parse_conflict_aware_loss(stage1_joint),
        ),
        train=TrainConfig(
            optimizer="adamw",
            learning_rate=float(train_smoke["learning_rate"]),
            weight_decay=0.0,
            max_epochs=1,
            max_steps=int(train_smoke["max_steps"]),
            grad_clip_norm=None,
            log_every_n_steps=1,
            resume_from_checkpoint=None,
        ),
        eval=EvalConfig(
            enabled=True,
            every_n_steps=None,
            every_n_epochs=None,
            attention_top_fraction=0.2,
        ),
        checkpoint=CheckpointConfig(
            save_every_n_steps=None,
            save_every_n_epochs=None,
            save_last=True,
            save_best=False,
            monitor="loss_total",
            mode="min",
            resume_from=None,
        ),
        reproducibility=ReproducibilityConfig(
            seed=int(stage1_joint.get("seed", train_smoke.get("seed", 42))),
            deterministic_ablation=_as_bool(
                train_smoke.get("deterministic_ablation", True),
                "train_smoke.deterministic_ablation",
            ),
            reuse_initial_model=_as_bool(
                train_smoke.get("reuse_initial_model", True),
                "train_smoke.reuse_initial_model",
            ),
            reuse_patch_mask=_as_bool(
                train_smoke.get("reuse_patch_mask", True),
                "train_smoke.reuse_patch_mask",
            ),
        ),
        clinical_graph=None,
        graph_encoder=None,
        teacher=raw_config.get("teacher") if isinstance(raw_config.get("teacher"), dict) else None,
    )
