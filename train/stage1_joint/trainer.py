from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from breast_pretrain.data.bucketed_stage1_dataloader import Stage1EpochBucketCoverageTracker
from breast_pretrain.data.checkpoint_transition_authority import validate_path_only_diff_receipt, validate_transition_contract
from breast_pretrain.clinical_graph_sidecar.activation_audit import (
    build_graph_activation_audit_from_resolved_config,
)
from breast_pretrain.clinical_graph_sidecar.concept_head_activation_audit import (
    build_concept_head_activation_audit_from_resolved_config,
)
from breast_pretrain.models import visual_encoder_summary
from breast_pretrain.train.reproducibility import (
    clone_state_dict_to_cpu,
    compute_state_dict_checksum,
    set_global_seed,
)
from breast_pretrain.train.distributed import resolve_distributed_runtime
from breast_pretrain.train.stage1_joint.checkpointing import (
    _to_serializable,
    validate_authorized_resume_checksum_transition,
    load_checkpoint,
    maybe_save_policy_checkpoints,
    maybe_warn_save_best_not_implemented,
    resolve_rank_local_sampler_state,
    resolve_resume_checkpoint_path,
    save_last_checkpoint,
    write_summary_json,
)
from breast_pretrain.train.stage1_joint.config import (
    load_stage1_joint_trainer_config,
    override_stage1_joint_trainer_config,
)
from breast_pretrain.train.stage1_joint.config_integrity import (
    assert_formal_init_config_checksum,
    sha256_file,
)
from breast_pretrain.train.stage1_joint.config_runtime import (
    resolve_runtime_intervals,
)
from breast_pretrain.train.stage1_joint.config_validation import validate_formal_stage1_config
from breast_pretrain.train.stage1_joint.dataset_batch import (
    build_stage1_batching_summary,
    build_stage1_joint_dataloader,
    build_stage1_joint_dataset,
    default_summary_name,
    prepare_stage1_joint_batch,
    resolve_device,
)
from breast_pretrain.train.stage1_joint.dynamic_patch_utils import (
    resolve_patch_grid_from_batch,
)
from breast_pretrain.train.stage1_joint.eval_hook import run_minimal_eval_hook, should_run_eval
from breast_pretrain.train.stage1_joint.gaze_masking import (
    build_masking_runtime_state,
)
from breast_pretrain.train.stage1_joint.formal_step import formal_train_step
from breast_pretrain.train.stage1_joint.local_branch_attach import (
    attach_local_high_conf_branch,
)
from breast_pretrain.train.stage1_joint.metrics import Stage1JointMetricLogger
from breast_pretrain.train.stage1_joint.optimization import (
    build_optimizer_with_differential_lr,
)
from breast_pretrain.train.stage1_joint.semantic_forward import (
    build_semantic_runtime,
)
from breast_pretrain.train.stage1_joint.student_forward import (
    build_student_encoder,
    set_bn_policy,
)
from breast_pretrain.train.stage1_joint.mask_regressor import MaskRegressor
from breast_pretrain.train.stage1_joint.ddp_components import wrap_independent_trainable_components
from breast_pretrain.train.stage1_joint.high_conf_sidecar import HighConfSidecarLoader
from breast_pretrain.train.stage1_joint.formal_v6_entry_gate import assert_v6_formal_entry
from breast_pretrain.train.stage1_joint.stage1_sampler_contract import (
    LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
    resume_mismatch_keys,
    resume_required_missing_keys,
    validate_legacy_historical_sampler_authority,
)
from breast_pretrain.train.stage1_joint.teacher_latents import (
    RECONSTRUCTION_SOURCE_SELF,
    RECONSTRUCTION_SOURCE_TEACHER_NPY,
    build_no_teacher_latent_batch,
    build_teacher_latent_source_summary,
    load_teacher_latent_batch,
    validate_dataset_teacher_latents,
    validate_training_teacher_latents,
)
def _is_formal_config(config: object) -> bool:
    tier = str(getattr(config.metadata, "run_tier", "")).lower()
    return tier in {"formal_production", "production_ready_candidate"}

def _require_v6_for_formal_runtime(config_path: str | Path) -> None:
    """Block direct trainer invocation from bypassing the formal V6 entry gate."""
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Stage 1 config must contain a mapping: {path}")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    run_tier = str(raw.get("run_tier", metadata.get("run_tier", ""))).strip()
    entry_contract = str(raw.get("entry_contract_version", "")).strip().lower()
    if run_tier in {"formal_production", "production_ready_candidate"} or entry_contract == "v6":
        assert_v6_formal_entry(path)


def _load_projection_metadata(config: object) -> dict[str, object] | None:
    manifest_dir = Path(config.data.image_manifest_path).parent
    candidates = (
        manifest_dir / "projection_metadata.json",
        manifest_dir / "stage1_gaze_projection_metadata.json",
        manifest_dir / "patch_gaze_weights_dynamic" / "projection_metadata.json",
    )
    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    # Dataset Entry V2 carries the immutable geometry/mask contract per row.
    # Projection metadata remains an optional consistency authority; formal
    # callers fail closed on the frozen Dataset Entry fields themselves.
    return None


def _manifest_row_count(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _build_optimizer(
    config: object, model: torch.nn.Module, semantic_branch: torch.nn.Module
) -> tuple[torch.optim.Optimizer, dict]:
    if hasattr(config.train, "differential_lr") and config.train.differential_lr is not None:
        diff_lr = config.train.differential_lr
        if getattr(diff_lr, "enabled", False):
            return build_optimizer_with_differential_lr(
                model=model,
                semantic_branch=semantic_branch,
                backbone_lr=float(diff_lr.backbone_lr),
                head_lr=float(diff_lr.head_lr),
                weight_decay=float(config.train.weight_decay),
            )
    if str(config.train.optimizer).strip().lower() != "adamw":
        raise ValueError("Phase 1.5 formal trainer only supports optimizer=adamw.")
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(semantic_branch.parameters()),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    return opt, {}


def _current_epoch(step: int, num_batches: int) -> int:
    return 1 + ((step - 1) // max(1, num_batches))


def _advance_batch_stream(batch_stream: object, batch_offset: int) -> None:
    for _ in range(max(0, batch_offset)):
        next(batch_stream)


def _restore_masking_generator_state(masking_state: object, state: object | None) -> None:
    generator = getattr(masking_state, "mask_generator", None)
    if generator is not None and state is not None:
        generator.set_state(state)


def _build_checkpoint_snapshot(metrics: Stage1JointMetricLogger, loss_total: float) -> dict[str, object]:
    return {
        "loss_total": float(loss_total),
        "semantic_soft_valid_sample_count": int(metrics.semantic_soft_valid_sample_count),
        "total_optimized_samples": int(metrics.total_samples),
    }


def _file_checksum(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_existing_run_artifact(
    configured_path: Path | None,
    fallback_path: Path,
) -> Path | None:
    if configured_path is not None and configured_path.is_file():
        return configured_path
    if fallback_path.is_file():
        return fallback_path
    return configured_path


def _build_run_checksums(
    *,
    config: object,
    init_state_path: str | Path | None,
    high_conf_prior_npz: str | Path | None,
    high_conf_prior_manifest: str | Path | None,
    formal_init_payload: dict[str, object] | None,
    world_size: int,
) -> dict[str, object]:
    return {
        "checkpoint_schema_version": "formal_stage1_resume_v2",
        "resolved_training_config_sha256": _file_checksum(config.config_path),
        "manifest_checksum": _file_checksum(config.data.image_manifest_path),
        "transform_metadata_checksum": _file_checksum(Path(config.data.image_manifest_path).parent / "projection_metadata.json"),
        "sidecar_npz_checksum": _file_checksum(high_conf_prior_npz),
        "sidecar_manifest_checksum": _file_checksum(high_conf_prior_manifest),
        "init_state_checksum": _file_checksum(init_state_path),
        "formal_init_checksum": (
            formal_init_payload.get("formal_init_checksum")
            if formal_init_payload is not None
            else None
        ),
        "runtime_code_sha256": os.environ.get("HSM_FORMAL_RUNTIME_CODE_SHA256", "").strip() or None,
        "authorization_sha256": os.environ.get("HSM_FORMAL_AUTHORIZATION_SHA256", "").strip() or None,
        "gaze_runtime_sqlite_sha256": _file_checksum(os.environ.get("HSM_GAZE_RUNTIME_INDEX_PATH")),
        "world_size": int(world_size),
        "batch_policy": str(config.data.batch_policy),
        "batch_size_by_modality": dict(config.data.batch_size_by_modality or {}),
        "sampler_contract_version": str(config.data.sampler_contract_version),
        "checkpoint_every_n_steps": config.checkpoint.save_every_n_steps,
    }
def _validate_resume_checksums(
    *,
    resume_payload: dict[str, object] | None,
    current_checksums: dict[str, object],
    allow_legacy_missing_sampler_version: bool = False,
) -> None:
    if resume_payload is None:
        return
    saved = resume_payload.get("run_checksums") or {}
    if not isinstance(saved, dict) or not saved:
        raise ValueError("Resume checkpoint is missing run_checksums.")
    required = {
        "checkpoint_schema_version", "resolved_training_config_sha256", "manifest_checksum",
        "runtime_code_sha256", "authorization_sha256", "gaze_runtime_sqlite_sha256",
        "world_size", "batch_policy", "batch_size_by_modality", "sampler_contract_version",
        "checkpoint_every_n_steps",
    }
    missing = sorted(resume_required_missing_keys(
        saved, current_checksums, required,
        allow_legacy_missing_version=allow_legacy_missing_sampler_version,
    ))
    if missing:
        raise ValueError("Resume checkpoint is missing required authority fields: " + ", ".join(missing))
    mismatch_keys = resume_mismatch_keys(
        saved, current_checksums,
        allow_legacy_missing_version=allow_legacy_missing_sampler_version,
    )
    if not mismatch_keys:
        return
    _validate_resume_authority_successor(
        resume_payload=resume_payload, saved=saved, current_checksums=current_checksums,
    )


def _validate_resume_authority_successor(
    *, resume_payload: dict[str, object], saved: dict[str, object], current_checksums: dict[str, object],
) -> None:
    authorization_path = Path(os.environ.get("HSM_FORMAL_AUTHORIZATION_PATH", "")).expanduser()
    if not authorization_path.is_absolute() or not authorization_path.is_file():
        raise ValueError("Resume authority successor requires an absolute existing authorization path.")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    actual_authorization_sha = sha256_file(authorization_path)
    if actual_authorization_sha != current_checksums.get("authorization_sha256"):
        raise ValueError("Resume authority successor authorization file SHA mismatch.")
    authorities = authorization.get("bound_authorities")
    if not isinstance(authorities, dict):
        raise ValueError("Resume authority successor authorization has no bound authorities.")
    validation_only = authorities.get("authorization_scope") == "BOUNDED_REPAIR_VALIDATION_ONLY"
    policy = {
        "approved_numerical_safety_successor": True,
        "formal_method_drift": "NOT_YET_AUTHORIZED_CANDIDATE" if validation_only else False,
        "mammo_encoder_fp32_safety_required": True,
        "mammo_encoder_precision_policy": "fp32_safety_island",
        "required_environment": {"HSM_STAGE1_MAMMO_ENCODER_FP32_SAFETY": "1"},
    }
    if authorization.get("authorized") is not True or any(authorities.get(k) != v for k, v in policy.items()):
        raise ValueError("Resume authority successor numerical safety policy is invalid.")
    if os.environ.get("HSM_STAGE1_MAMMO_ENCODER_FP32_SAFETY") != "1":
        raise ValueError("Resume authority successor requires active Mammo FP32 safety.")
    transition = authorities.get("checkpoint_resume_authority_transition")
    if not isinstance(transition, dict):
        raise ValueError("Resume authority successor transition contract is invalid.")
    path_only, _ = validate_transition_contract(transition, authorities, validation_only=validation_only)
    if path_only: validate_path_only_diff_receipt(transition, authorities)
    canonical_saved = json.dumps(saved, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    bindings = {
        "parent_checkpoint_sha256": resume_payload.get("checkpoint_sha256"),
        "parent_global_step": resume_payload.get("step"),
        "parent_runtime_code_sha256": saved.get("runtime_code_sha256"),
        "parent_authorization_sha256": saved.get("authorization_sha256"),
        "parent_resolved_training_config_sha256": saved.get("resolved_training_config_sha256"),
        "parent_run_checksums_sha256": hashlib.sha256(canonical_saved).hexdigest(),
        "successor_runtime_code_sha256": current_checksums.get("runtime_code_sha256"),
    }
    if any(transition.get(key) != value for key, value in bindings.items()):
        raise ValueError("Resume authority successor checkpoint-parent binding mismatch.")
    validate_authorized_resume_checksum_transition(saved, current_checksums, transition)


def run_stage1_joint_trainer(
    config_path: str | Path,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
    *,
    backbone_weight_path: str | None = None,
    text_dim: int | None = None,
    init_state_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    high_conf_prior_npz: str | Path | None = None,
    high_conf_prior_manifest: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    _require_v6_for_formal_runtime(config_path)
    config = override_stage1_joint_trainer_config(
        config=load_stage1_joint_trainer_config(config_path), output_dir=output_dir,
        max_steps=max_steps, max_samples=max_samples,
        text_dim=text_dim, pretrained_weight_path=backbone_weight_path,
        resume_from=resume_from,
    )
    checkpoint_interval_override = os.environ.get("HSM_STAGE1_CHECKPOINT_EVERY_N_STEPS", "").strip()
    if checkpoint_interval_override:
        interval = int(checkpoint_interval_override)
        if interval <= 0:
            raise ValueError("HSM_STAGE1_CHECKPOINT_EVERY_N_STEPS must be positive.")
        config = replace(config, checkpoint=replace(config.checkpoint, save_every_n_steps=interval))
    # Resolve text_dim from prompt embeddings if not in config
    if config.model.text_dim <= 0 and config.semantic.prompt_embedding_path is not None:
        from breast_pretrain.train.stage1_joint.config_runtime import resolve_text_dim_from_prompt_embeddings
        resolved_text_dim = resolve_text_dim_from_prompt_embeddings(str(config.semantic.prompt_embedding_path))
        config = replace(config, model=replace(config.model, text_dim=resolved_text_dim))
    if _is_formal_config(config):
        issues = validate_formal_stage1_config(config, backbone_weight_path=Path(backbone_weight_path) if backbone_weight_path else None)
        if issues:
            raise ValueError("Formal Stage 1 config validation failed:\n  " + "\n  ".join(issues))
        if init_state_path is None:
            raise ValueError("formal-production trainer requires --init-state formal_init_state.pt.")
    formal_init_payload = None
    if init_state_path is not None:
        init_path = Path(init_state_path).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"formal_init_state.pt not found: {init_path}")
        formal_init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        if _is_formal_config(config):
            assert_formal_init_config_checksum(
                payload=formal_init_payload,
                current_config_checksum=sha256_file(config_path),
            )
    distributed_runtime = resolve_distributed_runtime(config.data.device)
    historical_sampler_authority = None
    if _is_formal_config(config) and config.data.sampler_contract_version == LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1:
        historical_sampler_authority = validate_legacy_historical_sampler_authority(
            config_path=config.config_path,
            manifest_path=config.data.image_manifest_path,
            world_size=int(distributed_runtime["world_size"]),
            resume_checkpoint_path=(config.train.resume_from_checkpoint or config.checkpoint.resume_from),
        )
    device = resolve_device(str(distributed_runtime["device"]))
    set_global_seed(seed=config.reproducibility.seed, deterministic_ablation=config.reproducibility.deterministic_ablation)

    config.data.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_stage1_joint_dataset(config, split="train")
    val_dataset = None
    teacher_latent_availability = validate_dataset_teacher_latents(dataset=dataset, require_teacher_latents=config.data.require_teacher_latents)
    dataloader = build_stage1_joint_dataloader(dataset, config, epoch=1)
    bucket_coverage_tracker = Stage1EpochBucketCoverageTracker(
        batch_policy=config.data.batch_policy,
        planned_coverage=build_stage1_batching_summary(dataset, config),
    )
    model, model_init_checksum = build_student_encoder(config, device)
    local_branch_build_info = attach_local_high_conf_branch(config, model)
    visual_summary = visual_encoder_summary(
        model,
        patch_size=config.model.patch_size,
        image_size=config.data.image_size,
        vision_encoder_name=config.model.vision_encoder_name,
    )

    # Load formal init state dicts if provided
    if formal_init_payload is not None:
        model.load_state_dict(formal_init_payload["model_state_dict"])
        model_init_checksum = formal_init_payload.get("model_init_checksum", model_init_checksum)
    elif config.model.local_high_conf_branch_enabled:
        model_init_checksum = compute_state_dict_checksum(clone_state_dict_to_cpu(model.state_dict()))

    semantic_runtime = build_semantic_runtime(config, device)
    if formal_init_payload is not None:
        semantic_runtime.branch.load_state_dict(
            formal_init_payload["semantic_branch_state_dict"]
        )

    # Build MaskRegressor
    visual_dim = config.model.output_patch_dim or config.model.latent_dim
    mask_regressor = MaskRegressor(dim=visual_dim).to(device)
    if formal_init_payload is not None:
        if "mask_regressor_state_dict" not in formal_init_payload:
            raise ValueError("formal_init_state.pt must include mask_regressor_state_dict.")
        mask_regressor.load_state_dict(formal_init_payload["mask_regressor_state_dict"])

    optimizer, opt_build_info = _build_optimizer(config, model, semantic_runtime.branch)
    opt_build_info.update(local_branch_build_info)

    mask_regressor_params = [p for p in mask_regressor.parameters() if p.requires_grad]
    if mask_regressor_params:
        mask_regressor_lr = config.train.learning_rate
        if config.train.differential_lr is not None and config.train.differential_lr.enabled:
            mask_regressor_lr = float(config.train.differential_lr.head_lr)
        optimizer.add_param_group(
            {
                "params": mask_regressor_params,
                "lr": mask_regressor_lr,
                "weight_decay": float(config.train.weight_decay),
            }
        )
        opt_build_info["mask_regressor_trainable_tensors"] = len(mask_regressor_params)

    masking_state = build_masking_runtime_state(config, num_patches=0)  # per-step resolution

    # Load high-confidence sidecar
    high_conf_sidecar = None
    if high_conf_prior_npz is not None and high_conf_prior_manifest is not None:
        high_conf_sidecar = HighConfSidecarLoader(
            npz_path=high_conf_prior_npz,
            manifest_path=high_conf_prior_manifest,
            expected_count=_manifest_row_count(config.data.image_manifest_path),
            expected_training_manifest_path=config.data.image_manifest_path,
            block_on_checksum_mismatch=True,
        )
        print(f"HighConfSidecarLoader: {high_conf_sidecar.total_loaded} priors loaded, "
              f"{high_conf_sidecar.zero_prior_count} zero-priors")

    projection_metadata = _load_projection_metadata(config)
    run_checksums = _build_run_checksums(
        config=config,
        init_state_path=init_state_path,
        high_conf_prior_npz=high_conf_prior_npz,
        high_conf_prior_manifest=high_conf_prior_manifest,
        formal_init_payload=formal_init_payload,
        world_size=int(distributed_runtime["world_size"]),
    )
    metrics = Stage1JointMetricLogger()
    visual_dim = config.model.output_patch_dim or config.model.latent_dim

    # BN policy for dual forward
    bn_policy = getattr(config.model, "batch_norm_policy", "freeze_running_stats")
    train_bn_affine = getattr(config.model, "train_batch_norm_affine", True)
    set_bn_policy(model, bn_policy, train_bn_affine)

    # AMP scaler
    use_amp = getattr(config.train, "use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=1.0)

    # Scheduler
    scheduler = None
    if hasattr(config.train, "scheduler") and config.train.scheduler is not None:
        from breast_pretrain.train.scheduler import build_warmup_cosine_scheduler
        intervals = resolve_runtime_intervals(
            dataloader_length=len(dataloader),
            epochs=config.train.max_epochs,
            warmup_ratio=float(getattr(config.train.scheduler, "warmup_ratio", 0.05)),
        )
        if max_steps is not None:
            intervals = replace(
                intervals,
                max_steps=int(config.train.max_steps),
                warmup_steps=max(1, int(int(config.train.max_steps) * float(getattr(config.train.scheduler, "warmup_ratio", 0.05)))),
            )
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            warmup_steps=intervals.warmup_steps,
            total_steps=intervals.max_steps,
            min_lr_ratio=float(getattr(config.train.scheduler, "min_lr_ratio", 0.01)),
        )
        if max_steps is None:
            config = replace(config, train=replace(config.train, max_steps=intervals.max_steps))
    else:
        intervals = resolve_runtime_intervals(
            dataloader_length=len(dataloader),
            epochs=config.train.max_epochs,
        )
        if max_steps is not None:
            intervals = replace(intervals, max_steps=int(config.train.max_steps))

    # Override config intervals with resolved values
    checkpoint_interval = intervals.checkpoint_interval
    eval_interval = intervals.eval_interval
    resolved_config_path = config.data.output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(_to_serializable(config), sort_keys=False),
        encoding="utf-8",
    )

    resume_payload = None
    resume_path = resolve_resume_checkpoint_path(config)
    if resume_path is not None:
        resume_payload = load_checkpoint(
            checkpoint_path=resume_path, model=model,
            semantic_branch=semantic_runtime.branch, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            mask_regressor=mask_regressor,
        )
        _restore_masking_generator_state(
            masking_state,
            resume_payload.get("mask_generator_state"),
        )
        _validate_resume_checksums(
            resume_payload=resume_payload,
            current_checksums=run_checksums,
            allow_legacy_missing_sampler_version=historical_sampler_authority is not None,
        )
    if bool(distributed_runtime["ddp_ready"]):
        model, semantic_runtime.branch, mask_regressor = wrap_independent_trainable_components(
            model=model,
            semantic_branch=semantic_runtime.branch,
            mask_regressor=mask_regressor,
            device=device,
            local_rank=int(distributed_runtime["local_rank"]),
            find_unused_parameters=False,
            # Formal batches are modality/image-size buckets.  A mammography,
            # MRI, or ultrasound bucket can legally have no valid target for
            # another modality-specific P0-B head, so the semantic reducer
            # must discover those unused parameters on that iteration.
            semantic_find_unused_parameters=True,
        )
    start_step = int(resume_payload["step"]) if resume_payload is not None else 0
    start_attempt_step = int(resume_payload.get("attempt_step", start_step)) if resume_payload is not None else 0
    start_epoch = int(resume_payload["epoch"]) if resume_payload is not None else 0
    if start_step >= config.train.max_steps:
        raise ValueError(
            f"resume checkpoint step {start_step} already reached/exceeded max_steps={config.train.max_steps}."
        )
    if resume_payload is not None and int(resume_payload.get("world_size", -1)) != int(distributed_runtime["world_size"]):
        raise ValueError("RESUME_REFUSED: checkpoint WORLD_SIZE differs from the formal launch WORLD_SIZE.")
    resume_sampler_state = (
        resolve_rank_local_sampler_state(
            resume_payload,
            dataloader_length=len(dataloader),
            rank=int(distributed_runtime["rank"]),
        )
        if resume_payload is not None else {}
    )
    resume_batch_offset = int(resume_sampler_state.get("next_batch_index", 0))
    resume_epoch = int(resume_sampler_state.get("epoch", start_epoch or 1))

    checkpoint_warning = maybe_warn_save_best_not_implemented(config.checkpoint.save_best)
    if checkpoint_warning is not None: metrics.warnings.append(checkpoint_warning)

    print(f"Loaded formal Stage 1 trainer config: {config.config_path}")
    print(f"Resolved output_dir: {config.data.output_dir}")
    print(
        "Formal Stage 1 trainer summary: "
        + json.dumps(
            {
                "image_manifest_path": str(config.data.image_manifest_path),
                "patch_size": config.model.patch_size,
                "latent_dim": config.model.latent_dim,
                "vision_encoder_name": config.model.vision_encoder_name,
                "visual_encoder_backend": visual_summary["visual_encoder_backend"],
                "pretrained_weight_path": visual_summary["pretrained_weight_path"],
                "encoder_trainable": visual_summary["encoder_trainable"],
                "output_patch_dim": visual_dim,
                "text_dim": config.model.text_dim,
                "align_dim": config.model.align_dim,
                "max_steps": config.train.max_steps,
                "seed": config.reproducibility.seed,
                "optimizer": config.train.optimizer,
                "reconstruction_teacher_source": config.semantic.reconstruction_teacher_source,
                "reconstruction_runtime_mode": config.semantic.reconstruction_teacher_source,
                "require_teacher_latents": config.data.require_teacher_latents,
                "resume_from": str(resume_path) if resume_path is not None else None,
                "start_step": start_step,
                "start_attempt_step": start_attempt_step,
                "start_epoch": start_epoch,
            },
            ensure_ascii=True,
        )
    )
    if resume_payload is not None:
        print(
            "FORMAL_STAGE1_RESUME " + json.dumps(
                {
                    "RESUME_MODE": True,
                    "PARENT_LAUNCH_ID": os.environ.get("HSM_FORMAL_PARENT_LAUNCH_ID"),
                    "CHECKPOINT_PATH": str(resume_path),
                    "CHECKPOINT_SHA256": resume_payload.get("checkpoint_sha256"),
                    "RESUMED_GLOBAL_STEP": start_step,
                    "NEXT_GLOBAL_STEP": start_step + 1,
                    "SAMPLER_POSITION": {"epoch": resume_epoch, "next_batch_index": resume_batch_offset},
                    "RNG_RESTORED": True,
                    "OPTIMIZER_RESTORED": True,
                    "SCALER_RESTORED": True,
                },
                ensure_ascii=True,
            )
        )

    checkpoint_dir = config.data.output_dir / "checkpoints"
    periodic_checkpoint_paths: list[str] = []
    used_patch_masks: list[torch.Tensor] = []
    final_eval_result = None
    final_epoch = start_epoch or 1
    last_loss_total = 0.0
    executed_steps = 0
    next_sampler_state: dict[str, object] | None = None
    consecutive_amp_overflow_recoveries = 0

    global_step = start_step
    attempt_step = start_attempt_step
    epoch = resume_epoch if resume_payload is not None else (global_step // max(1, len(dataloader))) + 1
    while global_step < config.train.max_steps:
        batch_offset = resume_batch_offset if epoch == resume_epoch else 0
        epoch_loader = build_stage1_joint_dataloader(
            dataset, config, epoch=epoch, start_batch_index=batch_offset,
        )
        epoch_sampler = getattr(epoch_loader, "batch_sampler", None)
        if hasattr(epoch_sampler, "set_epoch"):
            epoch_sampler.set_epoch(max(0, epoch - 1))
        final_epoch = epoch
        for relative_batch_index, raw_batch in enumerate(epoch_loader):
            batch_index = batch_offset + relative_batch_index
            if global_step >= config.train.max_steps:
                break
            attempt_step += 1
            is_epoch_end = (
                relative_batch_index == len(epoch_loader) - 1
            ) or (global_step + 1 == config.train.max_steps)
            batch = prepare_stage1_joint_batch(raw_batch, device=device, config=config)
            bucket_coverage_tracker.update(epoch=epoch, modalities=batch.modalities, image=batch.image)

            step_output = formal_train_step(
                config=config,
                batch=batch,
                model=model,
                semantic_runtime=semantic_runtime,
                mask_regressor=mask_regressor,
                masking_state=masking_state,
                device=device,
                step_index=attempt_step,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                projection_metadata=projection_metadata,
                high_conf_sidecar=high_conf_sidecar,
                use_amp=use_amp,
                last_committed_global_step=global_step,
                epoch=epoch,
                sampler_position={"epoch": epoch, "next_batch_index": batch_index},
                consecutive_amp_overflow_recoveries=consecutive_amp_overflow_recoveries,
            )
            next_batch_index = batch_index + 1
            next_epoch = epoch
            if next_batch_index >= len(dataloader):
                next_batch_index = 0
                next_epoch = epoch + 1
            next_sampler_state = {
                "epoch": next_epoch,
                "next_batch_index": next_batch_index,
                "next_global_step": global_step + 1,
                "batches_per_rank_epoch": len(dataloader),
                "sampler_contract_version": str(config.data.sampler_contract_version),
            }
            if not step_output.amp_state["optimizer_step_executed"]:
                if step_output.amp_state.get("amp_overflow_recovered"):
                    consecutive_amp_overflow_recoveries += 1
                    if int(distributed_runtime["rank"]) == 0:
                        print("AMP_GRADSCALER_RECOVERY_V1 " + json.dumps({
                            "attempt_step": attempt_step,
                            "last_committed_global_step": global_step,
                            "old_scale": step_output.amp_state["old_scale"],
                            "new_scale": step_output.amp_state["new_scale"],
                            "optimizer_update_at_overflow": False,
                            "formal_step_committed": False,
                            "batch_retry": False,
                        }, sort_keys=True))
                    continue
                raise RuntimeError("Formal train step returned without a committed optimizer update.")
            consecutive_amp_overflow_recoveries = 0
            global_step += 1
            loss_result = step_output.loss_result
            masking_output = step_output.masking_output
            teacher_batch = step_output.teacher_batch
            semantic_warnings = step_output.semantic_warnings
            semantic_output = step_output.semantic_output
            used_patch_masks.append(masking_output.patch_mask.detach().cpu().clone())
            executed_steps += 1
            last_loss_total = float(loss_result.losses["total"].item())
            next_sampler_state["next_global_step"] = global_step + 1

            metrics.update(
                current_batch_size=int(batch.image.shape[0]), losses=loss_result.losses,
                attention_tokens=masking_output.attention_tokens, patch_mask=masking_output.patch_mask,
                teacher_sources=teacher_batch.teacher_sources, missing_teacher_flags=teacher_batch.missing_teacher_flags,
                warnings=batch.prompt_warnings + masking_output.warnings + semantic_warnings + loss_result.warnings,
                semantic_valid_count=loss_result.semantic_valid_count,
                semantic_skipped_batch_count=loss_result.semantic_skipped_batch_count,
                prior_schema_versions=semantic_output.prior_schema_versions,
                modalities=batch.modalities,
                gaze_supervision_sources=batch.gaze_supervision_sources,
                prior_statuses=batch.prior_statuses,
                high_conf_tokens=masking_output.high_conf_tokens,
                patch_weights=masking_output.patch_weights,
                dynamic_loss_weights=loss_result.dynamic_loss_weights,
                loss_weight_audit=loss_result.loss_weight_audit,
                concept_head_losses=loss_result.concept_head_losses,
                concept_head_correct_counts=loss_result.concept_head_correct_counts,
                concept_head_valid_label_counts=loss_result.concept_head_valid_label_counts,
                concept_head_missing_label_counts=loss_result.concept_head_missing_label_counts,
                mask_policy_used=masking_output.mask_policy_used,
                adaptive_gaze_quota=masking_output.adaptive_gaze_quota,
                adaptive_random_fraction=masking_output.adaptive_random_fraction,
                fallback_reason=masking_output.fallback_reason,
                high_conf_mask_quota_actual=masking_output.high_conf_mask_quota_actual,
                random_mask_fraction_actual=masking_output.random_mask_fraction_actual,
                total_salient_count=masking_output.total_salient_count,
                masked_salient_count=masking_output.masked_salient_count,
                visible_salient_count=masking_output.visible_salient_count,
                q_vis=masking_output.q_vis,
                visible_salient_floor_violation_count=masking_output.visible_salient_floor_violation_count,
                graph_consistency_metrics=loss_result.graph_consistency_metrics,
                conflict_aware_enabled=config.losses.conflict_aware_enabled,
                local_branch_metrics=loss_result.local_branch_metrics,
            )

            periodic_checkpoint_paths.extend(maybe_save_policy_checkpoints(
                checkpoint_dir=checkpoint_dir, model=model, semantic_branch=semantic_runtime.branch,
                optimizer=optimizer, step=global_step, epoch=epoch, config=config,
                summary_snapshot=_build_checkpoint_snapshot(metrics, last_loss_total),
                is_epoch_end=is_epoch_end,
                scheduler=scheduler,
                scaler=scaler,
                mask_regressor=mask_regressor,
                masking_state=masking_state,
                dataloader_length=len(dataloader),
                run_checksums=run_checksums,
                sampler_state=next_sampler_state,
                attempt_step=attempt_step,
                successful_optimizer_steps=global_step,
            ))
            if global_step % config.train.log_every_n_steps == 0 or global_step == config.train.max_steps:
                if int(distributed_runtime["rank"]) == 0:
                    print(
                        "Stage 1 joint formal trainer step: "
                        + json.dumps(
                            {
                                "split": "train", "step": global_step, "attempt_step": attempt_step, "epoch": epoch,
                                "modality": batch.modalities[0] if batch.modalities else None,
                                "loss_total": last_loss_total,
                                "loss_global": float(loss_result.losses["global_align"].item()),
                                "loss_reconstruction": float(loss_result.losses["reconstruction_total"].item()),
                                "loss_visible": float(loss_result.losses["visible_align"].item()),
                                "loss_soft": float(loss_result.losses["semantic_soft"].item()),
                                "loss_concept": float(loss_result.losses["concept_cls"].item()),
                                "loss_cc": float(loss_result.losses["concept_consistency"].item()),
                                "loss_graph": float(loss_result.losses["graph_consistency"].item()),
                                "omega_rec": float(loss_result.dynamic_loss_weights.get("reconstruction", 1.0)),
                                "omega_sem": float(loss_result.dynamic_loss_weights.get("semantic_soft", 1.0)),
                                "q_vis": float(masking_output.q_vis.mean().item()),
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "grad_scale": float(scaler.get_scale()),
                            }, ensure_ascii=True,
                        )
                    )
            if global_step % eval_interval == 0 or global_step == config.train.max_steps:
                if val_dataset is None:
                    val_dataset = build_stage1_joint_dataset(config, split="val")
                final_eval_result = run_minimal_eval_hook(
                    model=model, dataset=val_dataset, config=config, device=device, step=global_step,
                    epoch=epoch, output_dir=config.data.output_dir,
                    normalized_mask_strategy=masking_state.normalized_mask_strategy,
                    semantic_runtime=semantic_runtime,
                    mask_regressor=mask_regressor,
                    projection_metadata=projection_metadata,
                    high_conf_sidecar=high_conf_sidecar,
                )
        epoch += 1

    if config.eval.enabled and final_eval_result is None:
        if val_dataset is None:
            val_dataset = build_stage1_joint_dataset(config, split="val")
        final_eval_result = run_minimal_eval_hook(
            model=model, dataset=val_dataset, config=config, device=device,
            step=global_step, epoch=final_epoch,
            output_dir=config.data.output_dir,
            normalized_mask_strategy=masking_state.normalized_mask_strategy,
            semantic_runtime=semantic_runtime,
            mask_regressor=mask_regressor,
            projection_metadata=projection_metadata,
            high_conf_sidecar=high_conf_sidecar,
        )

    teacher_latent_source_summary = build_teacher_latent_source_summary(teacher_sources=metrics.teacher_source_counter, missing_teacher_latent_count=metrics.missing_teacher_latent_count)
    validate_training_teacher_latents(config, teacher_latent_source_summary)
    checkpoint_path = None
    if config.checkpoint.save_last:
        checkpoint_path = save_last_checkpoint(
            checkpoint_dir=checkpoint_dir, model=model, semantic_branch=semantic_runtime.branch,
            optimizer=optimizer, step=global_step, epoch=final_epoch,
            config=config, summary_snapshot=_build_checkpoint_snapshot(metrics, last_loss_total),
            scheduler=scheduler,
            scaler=scaler,
            mask_regressor=mask_regressor,
            masking_state=masking_state,
            dataloader_length=len(dataloader),
            run_checksums=run_checksums,
            sampler_state=next_sampler_state,
            attempt_step=attempt_step,
            successful_optimizer_steps=global_step,
        )
    checkpoint_path = _resolve_existing_run_artifact(
        checkpoint_path,
        checkpoint_dir / "last.pt",
    )
    eval_output_path = _resolve_existing_run_artifact(
        final_eval_result.output_path if final_eval_result is not None else None,
        config.data.output_dir / "minimal_eval.json",
    )
    graph_activation_audit = build_graph_activation_audit_from_resolved_config(
        resolved_config_path,
        require_sidecar=False,
    )
    graph_activation_audit_status = "not_generated"
    graph_activation_audit_path = None
    if isinstance(graph_activation_audit, dict):
        graph_activation_audit_status = str(
            graph_activation_audit.get("graph_activation_audit_status") or "generated_without_status"
        )
        graph_activation_audit_path = config.data.output_dir / "graph_activation_audit.json"
        graph_activation_audit_path.write_text(
            json.dumps(graph_activation_audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    concept_head_activation_audit = build_concept_head_activation_audit_from_resolved_config(
        resolved_config_path,
    )
    concept_head_activation_audit_status = str(
        concept_head_activation_audit.get("concept_head_activation_audit_status")
        or "generated_without_status"
    )
    concept_head_activation_audit_path = config.data.output_dir / "concept_head_activation_audit.json"
    concept_head_activation_audit_path.write_text(
        json.dumps(concept_head_activation_audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_payload = metrics.build_summary_payload(
        config_path=str(config.config_path),
        output_dir=str(config.data.output_dir),
        dataset_sample_count=len(dataset),
        teacher_latent_availability=teacher_latent_availability,
        teacher_latent_source_summary=teacher_latent_source_summary,
        mask_checksum=masking_state.mask_checksum,
        model_init_checksum=model_init_checksum,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        eval_payload=(
            {
                **final_eval_result.payload,
                "output_path": str(eval_output_path or final_eval_result.output_path),
            }
            if final_eval_result is not None
            else None
        ),
        used_patch_masks=used_patch_masks,
        normalized_gaze_loss_mode=masking_state.normalized_gaze_loss_mode,
        configured_mask_strategy=masking_state.configured_mask_strategy,
        normalized_mask_strategy=masking_state.normalized_mask_strategy,
        mask_prior_mode=masking_state.mask_prior_mode,
        reconstruction_teacher_source=config.semantic.reconstruction_teacher_source,
        text_dim=config.model.text_dim,
        align_dim=config.model.align_dim,
        latent_dim=visual_dim,
        active_concept_heads=config.semantic.active_concept_heads,
        pending_concept_heads=config.semantic.pending_concept_heads,
        require_teacher_latents=config.data.require_teacher_latents,
        loss_weights={
            "reconstruction_weight": config.losses.reconstruction_weight,
            "global_align_weight": config.losses.global_align_weight,
            "visible_align_weight": config.losses.visible_align_weight,
            "semantic_soft_weight": config.losses.semantic_soft_weight,
            "concept_loss_weight": config.losses.concept_loss_weight,
            "concept_consistency_weight": config.losses.concept_consistency_weight,
            "graph_consistency_weight": config.losses.graph_consistency_weight,
            "local_high_conf_branch_loss_weight": config.model.local_high_conf_branch_loss_weight,
            "concept_head_weights": config.losses.concept_head_weights,
            "concept_consistency_head_weights": config.losses.concept_consistency_head_weights,
            "modality_embedding_enabled": config.model.modality_embedding,
            "modality_vocab": list(config.model.modality_vocab),
            "modality_embedding_strategy": config.model.modality_embedding_strategy,
            "conflict_aware": {
                "enabled": config.losses.conflict_aware_enabled,
                "semantic_visible_coverage_target": config.losses.conflict_aware_semantic_visible_coverage_target,
                "reconstruction_masked_gaze_target": config.losses.conflict_aware_reconstruction_masked_gaze_target,
                "min_weight": config.losses.conflict_aware_min_weight,
                "max_weight": config.losses.conflict_aware_max_weight,
                "warmup_steps": config.losses.conflict_aware_warmup_steps,
                "allow_dynamic_graph_consistency_weighting": config.losses.allow_dynamic_graph_consistency_weighting,
            },
            "graph_encoder": (
                {
                    "enabled": bool(config.graph_encoder.enabled),
                    "implementation": config.graph_encoder.implementation,
                    "source_model_family": config.graph_encoder.source_model_family,
                    "mode": config.graph_encoder.mode,
                    "fusion_target": list(config.graph_encoder.fusion_target),
                    "output_consumed_by_semantic_branch": True,
                    "forbid_image_region_graph_node_alignment": (
                        config.graph_encoder.forbid_image_region_graph_node_alignment
                    ),
                }
                if config.graph_encoder is not None
                else {"enabled": False}
            ),
        },
        visual_encoder=visual_summary,
        run_metadata={
            "run_tier": config.metadata.run_tier,
            "model_role": config.metadata.model_role,
            "compliance_status": config.metadata.compliance_status,
            "known_limitations": list(config.metadata.known_limitations),
            "allowed_claims": list(config.metadata.allowed_claims),
            "forbidden_claims": list(config.metadata.forbidden_claims),
        },
        checkpoint_policy_applied={
            "resume_from": str(resume_path) if resume_path is not None else None,
            "save_last": bool(config.checkpoint.save_last),
            "save_every_n_steps": config.checkpoint.save_every_n_steps,
            "save_every_n_epochs": config.checkpoint.save_every_n_epochs,
            "save_best": bool(config.checkpoint.save_best),
            "save_best_status": (
                "not_implemented_phase_1_5" if config.checkpoint.save_best else "disabled"
            ),
            "periodic_checkpoint_paths": periodic_checkpoint_paths,
        },
        resume_info={
            "resumed": bool(resume_payload is not None),
            "resume_checkpoint_path": str(resume_path) if resume_path is not None else None,
            "resume_start_step": int(start_step),
            "resume_start_epoch": int(start_epoch),
            "resume_batch_offset": int(resume_batch_offset),
            "steps_executed_this_run": int(executed_steps),
            "target_max_steps": int(config.train.max_steps),
        },
        graph_activation_audit=graph_activation_audit,
        graph_activation_audit_path=str(graph_activation_audit_path) if graph_activation_audit_path is not None else None,
        graph_activation_audit_status=graph_activation_audit_status,
        concept_head_activation_audit=concept_head_activation_audit,
        concept_head_activation_audit_path=str(concept_head_activation_audit_path),
        concept_head_activation_audit_status=concept_head_activation_audit_status,
    )
    summary_payload["run_checksums"] = run_checksums
    summary_payload["resolved_config_path"] = str(resolved_config_path)
    summary_payload["batching_summary"] = bucket_coverage_tracker.to_summary()
    summary_payload["batch_norm_policy"] = {
        "policy": bn_policy,
        "train_affine": train_bn_affine,
        "dual_forward_safe": True,
    }
    summary_payload["training_config"] = {
        "use_amp": use_amp,
        "grad_clip_norm": config.train.grad_clip_norm,
        "differential_lr": bool(getattr(config.train, "differential_lr", None) is not None),
        "scheduler": "warmup_cosine" if scheduler is not None else "none",
        "resolved_intervals": {
            "batches_per_epoch": intervals.batches_per_epoch,
            "max_steps": intervals.max_steps,
            "warmup_steps": intervals.warmup_steps,
            "checkpoint_interval": intervals.checkpoint_interval,
            "eval_interval": intervals.eval_interval,
        },
    }
    summary_payload["optimizer_build_info"] = opt_build_info
    summary_path = write_summary_json(default_summary_name(config.data.output_dir), summary_payload)
    return {
        **summary_payload,
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "eval_output_path": str(eval_output_path) if eval_output_path is not None else None,
    }
