from __future__ import annotations

import os

import torch
import torch.distributed as dist


def resolve_distributed_runtime(device: str) -> dict[str, object]:
    normalized = str(device).strip().lower()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid distributed environment: rank={rank}, world_size={world_size}."
        )
    if world_size == 1:
        return {
            "backend": "none",
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": normalized or "cpu",
            "ddp_ready": False,
            "process_group_initialized": False,
        }
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable for WORLD_SIZE > 1.")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA distributed runtime requested but CUDA is unavailable.")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds CUDA device count {torch.cuda.device_count()}."
            )
        torch.cuda.set_device(local_rank)
        runtime_device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        runtime_device = normalized or "cpu"
        backend = "gloo"
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        initialized_here = True
    return {
        "backend": backend,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": runtime_device,
        "ddp_ready": True,
        "process_group_initialized": True,
        "initialized_by_runtime": initialized_here,
    }

