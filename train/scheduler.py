from __future__ import annotations

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a scheduler with linear warmup followed by cosine decay.

    During warmup (0 → warmup_steps): LR ramps linearly from 0 to base_lr.
    After warmup (warmup_steps → total_steps): LR follows cosine decay to min_lr.

    The base_lr is taken from the first param group in the optimizer.
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: object,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Adapt the frozen Stage 1 scheduler config to the existing scheduler.

    This mirrors the scheduler branch already used by ``stage1_joint.trainer``
    without changing the warmup-cosine formula or any optimizer behavior.
    """
    train = getattr(config, "train", None)
    if train is None:
        raise ValueError("build_scheduler requires config.train.")
    scheduler_config = getattr(train, "scheduler", None)
    if scheduler_config is None:
        return None

    scheduler_type = str(getattr(scheduler_config, "type", "cosine") or "cosine").strip().lower()
    if scheduler_type not in {"cosine", "warmup_cosine"}:
        raise ValueError(
            "Unsupported scheduler.type "
            f"{getattr(scheduler_config, 'type', None)!r}; expected 'cosine' or 'warmup_cosine'."
        )

    total_steps = getattr(train, "max_steps", None)
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("config.train.max_steps must be a positive integer.")

    try:
        warmup_ratio = float(getattr(scheduler_config, "warmup_ratio", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduler.warmup_ratio must be a float in [0.0, 1.0].") from exc
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("scheduler.warmup_ratio must be in [0.0, 1.0].")

    try:
        min_lr_ratio = float(getattr(scheduler_config, "min_lr_ratio", 0.01))
    except (TypeError, ValueError) as exc:
        raise ValueError("scheduler.min_lr_ratio must be a float in [0.0, 1.0].") from exc
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("scheduler.min_lr_ratio must be in [0.0, 1.0].")

    return build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=max(1, int(total_steps * warmup_ratio)),
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )


__all__ = ["build_scheduler", "build_warmup_cosine_scheduler"]
