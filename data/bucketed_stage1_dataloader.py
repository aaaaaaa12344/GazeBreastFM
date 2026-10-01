from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from breast_pretrain.data.collators.joint_pretrain_collator import joint_pretrain_collate_fn
from breast_pretrain.train.stage1_joint.stage1_sampler_contract import (
    FORMAL_BUCKETED_SAMPLER_CONTRACT_V2,
    LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
    validate_formal_bucket_batches,
)


BATCH_POLICY_BUCKET_BY_MODALITY_AND_IMAGE_SIZE = "bucket_by_modality_and_image_size"


def normalize_stage1_modality(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"mammo", "mammography"}:
        return "mammography"
    if value in {"us", "ultrasound"}:
        return "ultrasound"
    return value or "unknown"


def image_size_key(size: tuple[int, int]) -> str:
    return f"{int(size[0])}x{int(size[1])}"


def _record_target_size(dataset: Any, record: dict[str, Any]) -> tuple[int, int]:
    if hasattr(dataset, "_target_size_for_record"):
        size = dataset._target_size_for_record(record)
        return int(size[0]), int(size[1])
    return tuple(int(item) for item in dataset.image_size)


def _build_bucket_index(dataset: Any) -> dict[tuple[str, tuple[int, int]], list[int]]:
    buckets: dict[tuple[str, tuple[int, int]], list[int]] = defaultdict(list)
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("Bucketed Stage 1 dataloader requires a manifest-backed dataset with records.")
    for index, record in enumerate(records):
        modality = normalize_stage1_modality(record.get("modality"))
        size = _record_target_size(dataset, record)
        buckets[(modality, size)].append(index)
    return dict(buckets)


class Stage1BucketedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Any,
        *,
        batch_size_by_modality: dict[str, int],
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_batch_index: int = 0,
        avoid_singleton_tail: bool = True,
        contract_version: str = LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
        strict_formal: bool = False,
    ) -> None:
        self.dataset = dataset
        self.bucket_indices = _build_bucket_index(dataset)
        self.batch_size_by_modality = {
            normalize_stage1_modality(key): int(value)
            for key, value in batch_size_by_modality.items()
        }
        for modality, batch_size in self.batch_size_by_modality.items():
            if batch_size <= 0:
                raise ValueError(f"batch_size_by_modality.{modality} must be positive.")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_batch_index = int(start_batch_index)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid bucket sampler rank/world_size: {self.rank}/{self.world_size}")
        if self.start_batch_index < 0:
            raise ValueError("start_batch_index must be non-negative.")
        self.avoid_singleton_tail = bool(avoid_singleton_tail)
        self.contract_version = str(contract_version).strip() or LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1
        self.strict_formal = bool(strict_formal)

    def _global_bucket_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for (modality, _size), indices in self.bucket_indices.items():
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)
            batch_size = self.batch_size_by_modality.get(modality)
            if batch_size is None:
                raise ValueError(f"Missing batch_size_by_modality entry for modality: {modality}")
            bucket_batches: list[list[int]] = []
            for start in range(0, len(bucket_indices), batch_size):
                chunk = bucket_indices[start:start + batch_size]
                bucket_batches.append(chunk)

            # Singleton avoidance: redistribute tail if last batch has only 1 sample
            if self.avoid_singleton_tail and len(bucket_batches) >= 2:
                last = bucket_batches[-1]
                if len(last) == 1:
                    # Redistribute singleton into the preceding batch
                    bucket_batches[-2].extend(last)
                    bucket_batches.pop()

            batches.extend(bucket_batches)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _describe_batch(self, batch: list[int]) -> dict[str, Any]:
        record = self.dataset.records[batch[0]]
        modality = normalize_stage1_modality(record.get("modality"))
        return {
            "modality": modality,
            "image_size": list(_record_target_size(self.dataset, record)),
        }

    def _bucket_batches(self) -> list[list[int]]:
        global_batches = self._global_bucket_batches()
        if self.contract_version == FORMAL_BUCKETED_SAMPLER_CONTRACT_V2:
            validate_formal_bucket_batches(
                global_batches,
                batch_size_by_modality=self.batch_size_by_modality,
                world_size=self.world_size,
                describe_batch=self._describe_batch,
            )
        return global_batches[self.rank :: self.world_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._bucket_batches()
        if self.start_batch_index > len(batches):
            raise ValueError(
                f"start_batch_index={self.start_batch_index} exceeds epoch batches={len(batches)}."
            )
        yield from batches[self.start_batch_index:]

    def __len__(self) -> int:
        batches = self._bucket_batches()
        if self.start_batch_index > len(batches):
            raise ValueError(
                f"start_batch_index={self.start_batch_index} exceeds epoch batches={len(batches)}."
            )
        return len(batches) - self.start_batch_index


def build_bucketed_stage1_dataloader(
    dataset: Any,
    *,
    batch_size_by_modality: dict[str, int],
    shuffle: bool,
    num_workers: int,
    seed: int,
    epoch: int = 1,
    rank: int = 0,
    world_size: int = 1,
    start_batch_index: int = 0,
    avoid_singleton_tail: bool = True,
    contract_version: str = LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
    strict_formal: bool = False,
) -> DataLoader:
    sampler = Stage1BucketedBatchSampler(
        dataset,
        batch_size_by_modality=batch_size_by_modality,
        shuffle=shuffle,
        seed=int(seed),
        rank=int(rank),
        world_size=int(world_size),
        start_batch_index=int(start_batch_index),
        avoid_singleton_tail=avoid_singleton_tail,
        contract_version=contract_version,
        strict_formal=strict_formal,
    )
    sampler.set_epoch(max(0, int(epoch) - 1))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=joint_pretrain_collate_fn,
    )


def build_bucket_coverage_summary(
    dataset: Any,
    *,
    batch_policy: str,
    batch_size_by_modality: dict[str, int],
    contract_version: str = LEGACY_HISTORICAL_SAMPLER_CONTRACT_V1,
) -> dict[str, object]:
    buckets = _build_bucket_index(dataset)
    modality_counts: Counter[str] = Counter()
    image_size_counts: Counter[str] = Counter()
    bucket_records: list[dict[str, object]] = []
    for (modality, size), indices in sorted(buckets.items()):
        count = len(indices)
        modality_counts[modality] += count
        image_size_counts[image_size_key(size)] += count
        batch_size = int(batch_size_by_modality.get(modality, 0))
        planned_batches = Stage1BucketedBatchSampler(
            dataset,
            batch_size_by_modality=batch_size_by_modality,
            shuffle=False,
            seed=0,
        )._bucket_batches()
        planned_count = sum(
            1
            for batch in planned_batches
            if batch and normalize_stage1_modality(dataset.records[batch[0]].get("modality")) == modality
            and _record_target_size(dataset, dataset.records[batch[0]]) == size
        )
        bucket_records.append(
            {
                "modality": modality,
                "image_size": [int(size[0]), int(size[1])],
                "sample_count": count,
                "batch_size": batch_size,
                "planned_batch_count": planned_count if batch_size > 0 else 0,
                "sampler_contract_version": contract_version,
            }
        )
    return {
        "batch_policy": batch_policy,
        "batch_size_by_modality": dict(batch_size_by_modality),
        "bucket_count": len(bucket_records),
        "buckets": bucket_records,
        "modality_counts": dict(modality_counts),
        "image_size_counts": dict(image_size_counts),
        "sampler_contract_version": contract_version,
    }


@dataclass
class Stage1EpochBucketCoverageTracker:
    batch_policy: str
    planned_coverage: dict[str, object]
    epoch_records: dict[int, dict[str, Counter[str] | int]] = field(default_factory=dict)

    def update(self, *, epoch: int, modalities: list[str], image: torch.Tensor) -> None:
        if image.ndim != 4:
            raise ValueError(f"Expected batch image tensor [B, C, H, W], got {tuple(image.shape)}")
        record = self.epoch_records.setdefault(
            int(epoch),
            {
                "batch_count": 0,
                "sample_count": 0,
                "bucket_counts": Counter(),
                "modality_counts": Counter(),
                "image_size_counts": Counter(),
            },
        )
        batch_size = int(image.shape[0])
        size = (int(image.shape[-2]), int(image.shape[-1]))
        normalized_modalities = [normalize_stage1_modality(item) for item in modalities]
        modality_key = normalized_modalities[0] if normalized_modalities else "unknown"
        bucket_key = f"{modality_key}:{image_size_key(size)}"
        record["batch_count"] = int(record["batch_count"]) + 1
        record["sample_count"] = int(record["sample_count"]) + batch_size
        record["bucket_counts"].update([bucket_key])
        record["modality_counts"].update(normalized_modalities)
        record["image_size_counts"].update([image_size_key(size)] * batch_size)

    def to_summary(self) -> dict[str, object]:
        epoch_coverage = []
        for epoch, record in sorted(self.epoch_records.items()):
            epoch_coverage.append(
                {
                    "epoch": int(epoch),
                    "batch_count": int(record["batch_count"]),
                    "sample_count": int(record["sample_count"]),
                    "bucket_counts": dict(record["bucket_counts"]),
                    "modality_counts": dict(record["modality_counts"]),
                    "image_size_counts": dict(record["image_size_counts"]),
                }
            )
        return {
            **self.planned_coverage,
            "batch_policy": self.batch_policy,
            "epoch_bucket_coverage": epoch_coverage,
        }


__all__ = [
    "BATCH_POLICY_BUCKET_BY_MODALITY_AND_IMAGE_SIZE",
    "Stage1BucketedBatchSampler",
    "Stage1EpochBucketCoverageTracker",
    "build_bucket_coverage_summary",
    "build_bucketed_stage1_dataloader",
    "normalize_stage1_modality",
]
