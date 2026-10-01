from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from breast_pretrain.train.stage1_joint.formal_runtime_safety import (
    checkpoint_numerical_validation,
)


CHECKPOINT_SCHEMA_VERSION = "formal_stage1_resume_v2"


def config_sha_transition_allowed(
    saved_sha: object, current_sha: object, transition: object,
) -> bool:
    if saved_sha == current_sha:
        return True
    return bool(
        isinstance(transition, dict)
        and transition.get("transition_type") == "PATH_ONLY_CONFIG_REBIND_V1"
        and transition.get("parent_resolved_training_config_sha256") == saved_sha
        and transition.get("successor_resolved_training_config_sha256") == current_sha
    )


def validate_authorized_resume_checksum_transition(
    saved: dict[str, object], current: dict[str, object], transition: object,
) -> None:
    mismatches = {key for key, value in current.items() if saved.get(key) != value}
    ordinary = {"authorization_sha256", "runtime_code_sha256"}
    if not mismatches:
        return
    if not isinstance(transition, dict) or transition.get("transition_type") != "PATH_ONLY_CONFIG_REBIND_V1":
        if mismatches != ordinary:
            raise ValueError(f"Unauthorized resume checksum mismatches: {sorted(mismatches)}")
        return
    if not config_sha_transition_allowed(
        saved.get("resolved_training_config_sha256"),
        current.get("resolved_training_config_sha256"), transition,
    ):
        raise ValueError("Path-only resume config checksum transition is invalid.")
    allowed = ordinary | {"resolved_training_config_sha256"}
    if saved.get("init_state_checksum") != current.get("init_state_checksum"):
        allowed.add("init_state_checksum")
        receipt_path = Path(str(transition.get("formal_init_state_rebind_receipt_path", ""))).expanduser()
        if not receipt_path.is_absolute() or not receipt_path.is_file() or _sha256_file(receipt_path) != transition.get("formal_init_state_rebind_receipt_sha256"):
            raise ValueError("Formal init-state rebind receipt SHA/path mismatch.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required = {
            "schema_version": "formal_init_state_path_rebind_v1", "status": "PASS",
            "parent_init_state_sha256": saved.get("init_state_checksum"),
            "successor_init_state_sha256": current.get("init_state_checksum"),
            "tensor_state_changed": False, "formal_method_drift": False,
            "all_state_checksums_equal": True, "formal_init_checksum_equal": True,
        }
        transition_required = {
            "parent_init_state_sha256": saved.get("init_state_checksum"),
            "successor_init_state_sha256": current.get("init_state_checksum"),
            "init_state_transition_class": "METADATA_ONLY_PATH_CONFIG_REBIND",
            "tensor_state_changed": False, "formal_init_checksum_changed": False,
            "allowed_mismatch_keys": ["authorization_sha256", "runtime_code_sha256", "resolved_training_config_sha256", "init_state_checksum"],
        }
        checksum_pairs = (
            ("model_init_checksum_parent", "model_init_checksum_successor"),
            ("semantic_init_checksum_parent", "semantic_init_checksum_successor"),
            ("mask_regressor_init_checksum_parent", "mask_regressor_init_checksum_successor"),
            ("formal_init_checksum_parent", "formal_init_checksum_successor"),
        )
        if any(receipt.get(key) != value for key, value in required.items()) or any(transition.get(key) != value for key, value in transition_required.items()) or any(receipt.get(parent) != receipt.get(successor) for parent, successor in checksum_pairs):
            raise ValueError("Formal init-state artifact SHA transition is invalid.")
    if mismatches != allowed:
        raise ValueError(f"Unauthorized resume checksum mismatches: {sorted(mismatches)}")


def _checkpoint_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _distributed_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_formal_config(config: object) -> bool:
    metadata = config.get("metadata", {}) if isinstance(config, dict) else getattr(config, "metadata", None)
    tier = metadata.get("run_tier", "") if isinstance(metadata, dict) else getattr(metadata, "run_tier", "")
    return str(tier).strip() in {"formal_production", "production_ready_candidate"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _capture_rank_runtime_state(masking_state: object | None) -> dict[str, Any]:
    return {
        "rank": _distributed_rank()[0],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mask_generator_state": (
            masking_state.mask_generator.get_state()
            if masking_state is not None and getattr(masking_state, "mask_generator", None) is not None
            else None
        ),
    }


def _gather_rank_runtime_states(masking_state: object | None) -> list[dict[str, Any]]:
    local = _capture_rank_runtime_state(masking_state)
    _, world_size = _distributed_rank()
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("DDP checkpoint did not collect every rank runtime state.")
    return [dict(item) for item in gathered if item is not None]


def _gather_rank_sampler_states(sampler_state: dict[str, object] | None) -> list[dict[str, Any]]:
    rank, world_size = _distributed_rank()
    local = {"rank": rank, **dict(sampler_state or {})}
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("DDP checkpoint did not collect every rank sampler state.")
    return [dict(item) for item in gathered if item is not None]


def _restore_rank_runtime_state(states: object) -> dict[str, Any]:
    if not isinstance(states, list):
        raise ValueError("Checkpoint is missing rank_rng_states.")
    rank, world_size = _distributed_rank()
    if len(states) != world_size:
        raise ValueError(
            f"Checkpoint rank_rng_states world-size mismatch: checkpoint={len(states)} current={world_size}."
        )
    selected = next((state for state in states if isinstance(state, dict) and int(state.get("rank", -1)) == rank), None)
    if selected is None:
        raise ValueError(f"Checkpoint has no runtime state for DDP rank {rank}.")
    if selected.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(selected["torch_cpu_rng_state"])
    if selected.get("python_rng_state") is not None:
        random.setstate(selected["python_rng_state"])
    if selected.get("numpy_rng_state") is not None:
        np.random.set_state(selected["numpy_rng_state"])
    if selected.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(selected["cuda_rng_state_all"])
    return selected


def _write_atomic_checkpoint(path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checksum = _sha256_file(temporary)
    os.replace(temporary, path)
    return path, checksum


def _prune_rolling_checkpoints(checkpoint_dir: Path, retain: int = 2) -> None:
    rolling = []
    for candidate in checkpoint_dir.glob("checkpoint_step_*.pt"):
        receipt_path = candidate.with_name(candidate.name + ".receipt.json")
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if receipt.get("status") == "PASS" and receipt.get("numerical_validation", {}).get("passed") is True:
            rolling.append(candidate)
    rolling.sort(
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for path in rolling[retain:]:
        receipt = path.with_name(path.name + ".receipt.json")
        path.unlink()
        if receipt.is_file():
            receipt.unlink()


def save_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    semantic_branch: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: object,
    summary_snapshot: dict[str, object],
    scheduler_state_dict: dict[str, object] | None = None,
    scaler_state_dict: dict[str, object] | None = None,
    mask_regressor_state_dict: dict[str, object] | None = None,
    masking_state: object | None = None,
    sampler_state: dict[str, object] | None = None,
    run_checksums: dict[str, object] | None = None,
    rolling: bool = False,
    attempt_step: int | None = None,
    successful_optimizer_steps: int | None = None,
) -> Path | None:
    """Write one rank-0-only checkpoint after every rank reaches a step boundary."""
    path = Path(checkpoint_path)
    rank, world_size = _distributed_rank()
    rank_states = _gather_rank_runtime_states(masking_state)
    rank_sampler_states = _gather_rank_sampler_states(sampler_state)
    result: Path | None = None
    checkpoint_valid = True
    if rank == 0:
        payload: dict[str, Any] = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": int(step),
            "attempt_step": int(attempt_step if attempt_step is not None else step),
            "successful_optimizer_steps": int(successful_optimizer_steps if successful_optimizer_steps is not None else step),
            "epoch": int(epoch),
            "config": _to_serializable(config),
            "summary_snapshot": _to_serializable(summary_snapshot),
            "model_state_dict": _checkpoint_module(model).state_dict(),
            "semantic_branch_state_dict": _checkpoint_module(semantic_branch).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state_dict,
            "scaler_state_dict": scaler_state_dict,
            "mask_regressor_state_dict": mask_regressor_state_dict,
            "sampler_state": _to_serializable(sampler_state or {}),
            "sampler_positions_by_rank": _to_serializable(rank_sampler_states),
            "run_checksums": _to_serializable(run_checksums or {}),
            "rank_rng_states": rank_states,
            "python_rng_state": rank_states[0]["python_rng_state"],
            "numpy_rng_state": rank_states[0]["numpy_rng_state"],
            "rng_state": rank_states[0]["torch_cpu_rng_state"],
            "cuda_rng_state_all": rank_states[0]["cuda_rng_state_all"],
            "mask_generator_state": rank_states[0]["mask_generator_state"],
        }
        validation = checkpoint_numerical_validation(
            payload,
            require_scaler=_is_formal_config(config),
        )
        receipt_path = path.with_name(path.name + ".receipt.json")
        if not validation["passed"]:
            _atomic_json_write(
                receipt_path,
                {
                    "schema_version": "formal_stage1_checkpoint_receipt_v2",
                    "status": validation["status"],
                    "checkpoint_path": str(path),
                    "checkpoint_written": False,
                    "global_step": int(step),
                    "attempt_step": int(attempt_step if attempt_step is not None else step),
                    "epoch": int(epoch),
                    "sampler_state": _to_serializable(sampler_state or {}),
                    "numerical_validation": validation,
                },
            )
            checkpoint_valid = False
        if checkpoint_valid:
            result, checksum = _write_atomic_checkpoint(path, payload)
            disk_payload = torch.load(path, map_location="cpu", weights_only=False)
            disk_validation = checkpoint_numerical_validation(
                disk_payload,
                require_scaler=_is_formal_config(config),
            )
            metadata_matches = (
                int(disk_payload.get("step", -1)) == int(step)
                and int(disk_payload.get("epoch", -1)) == int(epoch)
                and disk_payload.get("sampler_state") == payload.get("sampler_state")
                and disk_payload.get("sampler_positions_by_rank") == payload.get("sampler_positions_by_rank")
            )
            if not disk_validation["passed"] or not metadata_matches:
                _atomic_json_write(
                    receipt_path,
                    {
                        "schema_version": "formal_stage1_checkpoint_receipt_v2",
                        "status": disk_validation["status"],
                        "checkpoint_path": str(path),
                        "checkpoint_written": True,
                        "checkpoint_sha256": checksum,
                        "global_step": int(step),
                        "epoch": int(epoch),
                        "numerical_validation": disk_validation,
                        "metadata_matches": metadata_matches,
                    },
                )
                checkpoint_valid = False
                result = None
        if checkpoint_valid:
            receipt = {
            "schema_version": "formal_stage1_checkpoint_receipt_v2",
            "status": "PASS",
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_path": str(path),
            "checkpoint_sha256": checksum,
            "global_step": int(step),
            "attempt_step": int(attempt_step if attempt_step is not None else step),
            "successful_optimizer_steps": int(successful_optimizer_steps if successful_optimizer_steps is not None else step),
            "epoch": int(epoch),
            "world_size": int(world_size),
            "sampler_state": _to_serializable(sampler_state or {}),
            "sampler_positions_by_rank": _to_serializable(rank_sampler_states),
            "run_checksums": _to_serializable(run_checksums or {}),
            "numerical_validation": disk_validation,
        }
            _atomic_json_write(receipt_path, receipt)
            _atomic_json_write(
                path.parent / "latest_valid_checkpoint.json",
                {
                    "schema_version": "formal_stage1_latest_valid_checkpoint_v1",
                    "status": "PASS",
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": checksum,
                    "receipt_path": str(receipt_path),
                    "global_step": int(step),
                },
            )
            if rolling:
                _prune_rolling_checkpoints(path.parent)
    if world_size > 1:
        dist.barrier()
    return result


def _sampler_state(
    step: int,
    epoch: int,
    dataloader_length: int | None,
    *,
    next_batch_index: int | None = None,
    next_epoch: int | None = None,
) -> dict[str, int]:
    batches = max(1, int(dataloader_length or 1))
    return {
        "epoch": int(next_epoch if next_epoch is not None else epoch),
        "next_batch_index": int(next_batch_index if next_batch_index is not None else int(step) % batches),
        "next_global_step": int(step) + 1,
        "batches_per_rank_epoch": batches,
    }


def resolve_rank_local_sampler_state(
    payload: dict[str, object], *, dataloader_length: int, rank: int
) -> dict[str, int]:
    """Select a saved rank cursor or reconstruct legacy rank-0-only checkpoints."""
    positions = payload.get("sampler_positions_by_rank")
    if isinstance(positions, list):
        selected = next(
            (item for item in positions if isinstance(item, dict) and int(item.get("rank", -1)) == rank),
            None,
        )
        if selected is None and 0 <= int(rank) < len(positions) and isinstance(positions[int(rank)], dict):
            selected = positions[int(rank)]
        if selected is None:
            raise ValueError(f"Checkpoint has no sampler position for DDP rank {rank}.")
        return {
            key: (str(value) if key == "sampler_contract_version" else int(value))
            for key, value in selected.items()
            if key != "rank"
        }

    batches = int(dataloader_length)
    if batches <= 0:
        raise ValueError("Rank-local dataloader length must be positive.")
    consumed_batches = int(payload.get("step", 0))
    completed_epochs, next_batch_index = divmod(consumed_batches, batches)
    return {
        "epoch": completed_epochs + 1,
        "next_batch_index": next_batch_index,
        "next_global_step": consumed_batches + 1,
        "batches_per_rank_epoch": batches,
    }


def save_last_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    semantic_branch: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: object,
    summary_snapshot: dict[str, object],
    scheduler: object | None = None,
    scaler: object | None = None,
    mask_regressor: torch.nn.Module | None = None,
    masking_state: object | None = None,
    dataloader_length: int | None = None,
    run_checksums: dict[str, object] | None = None,
    sampler_state: dict[str, object] | None = None,
    attempt_step: int | None = None,
    successful_optimizer_steps: int | None = None,
) -> Path | None:
    return save_checkpoint(
        checkpoint_path=checkpoint_dir / "last.pt",
        model=model, semantic_branch=semantic_branch, optimizer=optimizer,
        step=step, epoch=epoch, config=config, summary_snapshot=summary_snapshot,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        scaler_state_dict=scaler.state_dict() if scaler is not None else None,
        mask_regressor_state_dict=_checkpoint_module(mask_regressor).state_dict() if mask_regressor is not None else None,
        masking_state=masking_state,
        sampler_state=sampler_state or _sampler_state(step, epoch, dataloader_length),
        run_checksums=run_checksums,
        attempt_step=attempt_step,
        successful_optimizer_steps=successful_optimizer_steps,
    )


def _verify_checkpoint_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.with_name(path.name + ".receipt.json")
    if not receipt_path.is_file():
        raise ValueError(f"Checkpoint is NEVER_RESUMABLE without a receipt: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError(f"Checkpoint receipt is not PASS: {receipt_path}")
    if receipt.get("checkpoint_sha256") != _sha256_file(path):
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    return receipt


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    semantic_branch: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    mask_regressor: torch.nn.Module | None = None,
) -> dict[str, object]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    receipt = _verify_checkpoint_receipt(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {path}")
    validation = checkpoint_numerical_validation(
        payload,
        require_scaler=_is_formal_config(payload.get("config", {})),
    )
    if not validation["passed"]:
        refusal = (
            "RESUME_REFUSED_STATE_CONTRACT"
            if validation["status"] == "INVALID_STATE_CONTRACT"
            else "RESUME_REFUSED_NUMERICAL_INVALID"
        )
        raise ValueError(f"{refusal}: {path}: {validation['failures']}")
    model.load_state_dict(payload["model_state_dict"])
    semantic_branch.load_state_dict(payload["semantic_branch_state_dict"])
    if optimizer is not None:
        if payload.get("optimizer_state_dict") is None:
            raise ValueError(f"Checkpoint is missing optimizer_state_dict: {path}")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        if payload.get("scheduler_state_dict") is None:
            raise ValueError(f"Checkpoint is missing scheduler_state_dict: {path}")
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None:
        if payload.get("scaler_state_dict") is None:
            raise ValueError(f"Checkpoint is missing scaler_state_dict: {path}")
        scaler.load_state_dict(payload["scaler_state_dict"])
    if mask_regressor is not None:
        if payload.get("mask_regressor_state_dict") is None:
            raise ValueError(f"Checkpoint is missing mask_regressor_state_dict: {path}")
        mask_regressor.load_state_dict(payload["mask_regressor_state_dict"])
    local_state = _restore_rank_runtime_state(payload.get("rank_rng_states"))
    sampler_state = payload.get("sampler_state")
    if not isinstance(sampler_state, dict) or "next_batch_index" not in sampler_state:
        raise ValueError(f"Checkpoint is missing exact sampler_state: {path}")
    return {
        "checkpoint_path": str(path), "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
        "step": int(payload.get("step", 0)), "attempt_step": int(payload.get("attempt_step", payload.get("step", 0))), "epoch": int(payload.get("epoch", 0)),
        "summary_snapshot": payload.get("summary_snapshot", {}), "config": payload.get("config"),
        "scheduler_state_dict": payload.get("scheduler_state_dict"),
        "scaler_state_dict": payload.get("scaler_state_dict"),
        "mask_generator_state": local_state.get("mask_generator_state"),
        "sampler_state": sampler_state,
        "sampler_positions_by_rank": payload.get("sampler_positions_by_rank"),
        "run_checksums": payload.get("run_checksums", {}),
        "world_size": int(receipt["world_size"]),
    }


def maybe_save_policy_checkpoints(
    checkpoint_dir: Path, model: torch.nn.Module, semantic_branch: torch.nn.Module,
    optimizer: torch.optim.Optimizer, step: int, epoch: int, config: object,
    summary_snapshot: dict[str, object], is_epoch_end: bool, scheduler: object | None = None,
    scaler: object | None = None, mask_regressor: torch.nn.Module | None = None,
    masking_state: object | None = None, dataloader_length: int | None = None,
    run_checksums: dict[str, object] | None = None,
    sampler_state: dict[str, object] | None = None,
    attempt_step: int | None = None,
    successful_optimizer_steps: int | None = None,
) -> list[str]:
    checkpoint_config = config.checkpoint
    save_step = checkpoint_config.save_every_n_steps is not None and step % checkpoint_config.save_every_n_steps == 0
    save_epoch = checkpoint_config.save_every_n_epochs is not None and is_epoch_end and epoch % checkpoint_config.save_every_n_epochs == 0
    if not save_step and not save_epoch:
        return []
    name = f"checkpoint_step_{step:06d}.pt" if save_step else f"checkpoint_epoch_{epoch:03d}.pt"
    path = save_checkpoint(
        checkpoint_path=checkpoint_dir / name, model=model, semantic_branch=semantic_branch,
        optimizer=optimizer, step=step, epoch=epoch, config=config, summary_snapshot=summary_snapshot,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        scaler_state_dict=scaler.state_dict() if scaler is not None else None,
        mask_regressor_state_dict=_checkpoint_module(mask_regressor).state_dict() if mask_regressor is not None else None,
        masking_state=masking_state,
        sampler_state=sampler_state or _sampler_state(step, epoch, dataloader_length),
        run_checksums=run_checksums, rolling=save_step,
        attempt_step=attempt_step,
        successful_optimizer_steps=successful_optimizer_steps,
    )
    return [str(path)] if path is not None else []


def maybe_warn_save_best_not_implemented(save_best: bool) -> str | None:
    if not save_best:
        return None
    warning_message = "checkpoint_save_best_not_implemented:phase_1_5"
    warnings.warn(warning_message, stacklevel=2)
    return warning_message


def resolve_resume_checkpoint_path(config: object) -> Path | None:
    resolved = getattr(config.checkpoint, "resume_from", None) or getattr(config.train, "resume_from_checkpoint", None)
    if resolved is None:
        return None
    candidate = Path(resolved).expanduser()
    if str(candidate) != "latest":
        return candidate.resolve()
    root_value = os.environ.get("HSM_STAGE1_LATEST_CHECKPOINT_ROOT", "").strip()
    root = Path(root_value) if root_value else Path(config.data.output_dir) / "checkpoints"
    pointer = root / "latest_valid_checkpoint.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No latest valid checkpoint pointer: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or not payload.get("checkpoint_path"):
        raise ValueError(f"Latest checkpoint pointer is not resumable: {pointer}")
    return Path(str(payload["checkpoint_path"])).expanduser().resolve()


def write_summary_json(summary_path: Path, summary_payload: dict[str, object]) -> Path:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(summary_path, summary_payload)
    return summary_path
