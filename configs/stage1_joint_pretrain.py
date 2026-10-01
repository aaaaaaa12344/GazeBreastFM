from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.train.stage1_joint import (
    load_stage1_joint_trainer_config,
    override_stage1_joint_trainer_config,
)
from breast_pretrain.train.stage1_joint.types import Stage1JointTrainerConfig


@dataclass(frozen=True)
class AuditConfig:
    warn_only: bool = False


@dataclass(frozen=True)
class ArtifactConfig:
    summary_filename: str = "stage1_joint_train_summary.json"
    input_audit_filename: str = "input_audit_summary.json"
    resolved_config_filename: str = "resolved_config.yaml"
    train_log_filename: str = "train_log.jsonl"
    metric_history_filename: str = "metric_history.csv"
    loss_breakdown_filename: str = "loss_breakdown.csv"
    checkpoint_last_filename: str = "checkpoint_last.pt"
    checkpoint_best_filename: str = "checkpoint_best.pt"


@dataclass(frozen=True)
class Stage1JointPretrainBundle:
    config_path: Path
    trainer: Stage1JointTrainerConfig
    audit: AuditConfig
    artifacts: ArtifactConfig
    raw_config: dict[str, Any]


def _load_raw_yaml(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return payload


def _as_bool(raw_value: Any, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean-like value, got {raw_value!r}")


def load_stage1_joint_pretrain_bundle(
    config_path: str | Path,
    *,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
) -> Stage1JointPretrainBundle:
    resolved_config_path = Path(config_path).expanduser().resolve()
    raw_config = _load_raw_yaml(resolved_config_path)
    trainer = override_stage1_joint_trainer_config(
        config=load_stage1_joint_trainer_config(resolved_config_path),
        output_dir=output_dir,
        max_steps=max_steps,
        max_samples=max_samples,
    )
    audit_block = raw_config.get("audit") if isinstance(raw_config.get("audit"), dict) else {}
    artifacts_block = (
        raw_config.get("artifacts") if isinstance(raw_config.get("artifacts"), dict) else {}
    )
    return Stage1JointPretrainBundle(
        config_path=resolved_config_path,
        trainer=trainer,
        audit=AuditConfig(
            warn_only=_as_bool(audit_block.get("warn_only"), default=False),
        ),
        artifacts=ArtifactConfig(
            summary_filename=str(
                artifacts_block.get("summary_filename", "stage1_joint_train_summary.json")
            ),
            input_audit_filename=str(
                artifacts_block.get("input_audit_filename", "input_audit_summary.json")
            ),
            resolved_config_filename=str(
                artifacts_block.get("resolved_config_filename", "resolved_config.yaml")
            ),
            train_log_filename=str(
                artifacts_block.get("train_log_filename", "train_log.jsonl")
            ),
            metric_history_filename=str(
                artifacts_block.get("metric_history_filename", "metric_history.csv")
            ),
            loss_breakdown_filename=str(
                artifacts_block.get("loss_breakdown_filename", "loss_breakdown.csv")
            ),
            checkpoint_last_filename=str(
                artifacts_block.get("checkpoint_last_filename", "checkpoint_last.pt")
            ),
            checkpoint_best_filename=str(
                artifacts_block.get("checkpoint_best_filename", "checkpoint_best.pt")
            ),
        ),
        raw_config=raw_config,
    )


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


def serialize_stage1_joint_pretrain_bundle(
    bundle: Stage1JointPretrainBundle,
) -> dict[str, Any]:
    return {
        "config_path": str(bundle.config_path),
        "trainer": _to_serializable(bundle.trainer),
        "audit": _to_serializable(bundle.audit),
        "artifacts": _to_serializable(bundle.artifacts),
    }
