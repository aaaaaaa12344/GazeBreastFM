from __future__ import annotations

import json
from pathlib import Path

import torch

from breast_pretrain.train.eval_masked_latent import evaluate_masked_latent_reconstruction
from breast_pretrain.train.stage1_joint.dataset_batch import (
    build_stage1_joint_dataloader,
    prepare_stage1_joint_batch,
)
from breast_pretrain.train.stage1_joint.formal_step import formal_train_step
from breast_pretrain.train.stage1_joint.gaze_masking import build_masking_runtime_state
from breast_pretrain.train.stage1_joint.student_forward import set_bn_policy
from breast_pretrain.train.stage1_joint.types import EvalHookResult, Stage1JointTrainerConfig


def should_run_eval(
    config: Stage1JointTrainerConfig,
    step: int,
    epoch: int,
    is_epoch_end: bool,
) -> bool:
    if not config.eval.enabled:
        return False
    if config.eval.every_n_steps is not None and step % config.eval.every_n_steps == 0:
        return True
    if (
        config.eval.every_n_epochs is not None
        and is_epoch_end
        and epoch % config.eval.every_n_epochs == 0
    ):
        return True
    return False


def run_minimal_eval_hook(
    *,
    model: torch.nn.Module,
    dataset: object,
    config: Stage1JointTrainerConfig,
    device: torch.device,
    step: int,
    epoch: int,
    output_dir: Path,
    normalized_mask_strategy: str,
    semantic_runtime: object | None = None,
    mask_regressor: torch.nn.Module | None = None,
    projection_metadata: dict[str, object] | None = None,
    high_conf_sidecar: object | None = None,
) -> EvalHookResult:
    eval_output_dir = output_dir / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    if semantic_runtime is not None and mask_regressor is not None:
        eval_masking_state = build_masking_runtime_state(config, num_patches=0)
        was_training = model.training
        model.eval()
        mask_regressor.eval()
        semantic_runtime.branch.eval()
        dataloader = build_stage1_joint_dataloader(dataset, config)
        raw_batch = next(iter(dataloader))
        batch = prepare_stage1_joint_batch(raw_batch, device=device, config=config)
        with torch.no_grad():
            step_output = formal_train_step(
                config=config,
                batch=batch,
                model=model,
                semantic_runtime=semantic_runtime,
                mask_regressor=mask_regressor,
                masking_state=eval_masking_state,
                device=device,
                step_index=step,
                projection_metadata=projection_metadata,
                high_conf_sidecar=high_conf_sidecar,
                use_amp=config.train.use_amp,
            )
        if was_training:
            model.train()
            set_bn_policy(
                model,
                getattr(config.model, "batch_norm_policy", "freeze_running_stats"),
                getattr(config.model, "train_batch_norm_affine", True),
            )
            mask_regressor.train()
            semantic_runtime.branch.train()
        payload = {
            "formal_train_monitor": True,
            "loss_total": float(step_output.loss_result.losses["total"].detach().cpu().item()),
            "loss_reconstruction": float(
                step_output.loss_result.losses["reconstruction_total"].detach().cpu().item()
            ),
            "loss_semantic_soft": float(
                step_output.loss_result.losses["semantic_soft"].detach().cpu().item()
            ),
            "batch_size": int(batch.image.shape[0]),
            "modalities": list(batch.modalities),
            "dynamic_grid_supported": True,
        }
    else:
        payload = evaluate_masked_latent_reconstruction(
            model=model,
            dataset=dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            device=device,
            patch_size=config.model.patch_size,
            latent_dim=config.model.output_patch_dim or config.model.latent_dim,
            mask_ratio=config.masking.mask_ratio,
            eval_seed=config.reproducibility.seed,
            attention_top_fraction=config.eval.attention_top_fraction,
            mask_strategy=normalized_mask_strategy,
            gaze_mask_sampling_alpha=config.masking.gaze_mask_sampling_alpha,
            high_conf_mask_quota=config.masking.high_conf_mask_quota,
            min_random_mask_fraction=config.masking.min_random_mask_fraction,
            mask_sampling_temperature=config.masking.mask_sampling_temperature,
            gaze_mask_eps=config.masking.gaze_mask_eps,
        )
    payload = {
        "eval_type": "minimal_masked_latent_reconstruction",
        "step": int(step),
        "epoch": int(epoch),
        **payload,
    }
    output_path = eval_output_dir / f"step_{step:06d}_minimal_eval.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return EvalHookResult(payload=payload, output_path=output_path)
