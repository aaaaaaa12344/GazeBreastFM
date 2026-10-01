from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeResolution:
    """Runtime-resolved values computed from the actual dataloader."""

    batches_per_epoch: int
    max_steps: int
    warmup_steps: int
    checkpoint_interval: int
    eval_interval: int
    epochs: int = 100


def resolve_runtime_intervals(
    dataloader_length: int,
    epochs: int = 100,
    warmup_ratio: float = 0.05,
    checkpoint_every_n_epochs: int = 5,
    eval_every_n_epochs: int = 5,
) -> RuntimeResolution:
    """Resolve all step-based intervals from the actual dataloader length.

    All intervals are computed from `len(dataloader) * epochs` to ensure
    consistent epoch boundaries regardless of bucket composition.
    """
    batches_per_epoch = max(1, dataloader_length)
    max_steps = batches_per_epoch * epochs
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    checkpoint_interval = max(1, batches_per_epoch * checkpoint_every_n_epochs)
    eval_interval = max(1, batches_per_epoch * eval_every_n_epochs)

    return RuntimeResolution(
        batches_per_epoch=batches_per_epoch,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        checkpoint_interval=checkpoint_interval,
        eval_interval=eval_interval,
        epochs=epochs,
    )


def resolve_text_dim_from_prompt_embeddings(prompt_embedding_path: str) -> int:
    """Read text_dim from the stage1_prompt_embeddings.json file."""
    import json
    from pathlib import Path

    path = Path(prompt_embedding_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt embedding file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        embeddings = data.get("embeddings") or data.get("prompt_embeddings")
        if embeddings is not None:
            if isinstance(embeddings, dict):
                first_val = next(iter(embeddings.values()), None)
            elif isinstance(embeddings, list) and len(embeddings) > 0:
                first_val = embeddings[0]
            else:
                first_val = embeddings
            if isinstance(first_val, list):
                return len(first_val)
        # Try top-level keys
        for key in ("dim", "text_dim", "embedding_dim"):
            if key in data:
                return int(data[key])

    raise ValueError(
        f"Cannot determine text_dim from prompt embeddings file: {path}. "
        f"Expected format: {{\"embeddings\": {{id: [vector]}}}} or {{\"dim\": int}}"
    )


def build_bn_policy_summary(policy: str, train_affine: bool) -> dict[str, object]:
    """Build a summary of the BatchNorm policy for run metadata."""
    return {
        "policy": policy,
        "train_affine": train_affine,
        "running_stats_frozen": policy == "freeze_running_stats",
        "dual_forward_safe": True,
    }


__all__ = [
    "RuntimeResolution",
    "build_bn_policy_summary",
    "resolve_runtime_intervals",
    "resolve_text_dim_from_prompt_embeddings",
]
