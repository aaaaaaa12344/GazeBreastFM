"""Formal P0-B prototype projection and L_cc computation primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PrototypeRuntimeIndex:
    vectors: dict[str, torch.Tensor]

    @classmethod
    def load(cls, path: str | Path, *, schema, embedding_authority_hash: str | None = None, target_authority_hash: str | None = None) -> "PrototypeRuntimeIndex":
        vectors: dict[str, torch.Tensor] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                concept_id = str(row.get("concept_id", ""))
                spec = schema.concepts.get(concept_id)
                if spec is None or not spec.supervision_modes.get("prototype", False):
                    raise ValueError(f"Prototype asset has disallowed concept at line {line_number}")
                if row.get("split") != "train":
                    raise ValueError(f"Prototype asset violates train/minimum support at line {line_number}")
                if not bool(row.get("available", False)):
                    continue
                if int(row.get("support_case_count", 0)) < 2:
                    raise ValueError(f"Available prototype has insufficient support at line {line_number}")
                if row.get("concept_schema_version") != schema.version or row.get("concept_target_authority_hash") != target_authority_hash:
                    raise ValueError(f"Prototype asset schema/target lineage mismatch at line {line_number}")
                if embedding_authority_hash is not None and row.get("embedding_authority_hash") != embedding_authority_hash:
                    raise ValueError(f"Prototype asset embedding lineage mismatch at line {line_number}")
                vector = np.asarray(row.get("prototype_vector"), dtype=np.float32)
                if vector.ndim != 1 or not np.isfinite(vector).all() or not np.linalg.norm(vector):
                    raise ValueError(f"Invalid prototype vector at line {line_number}")
                expected = hashlib.sha256(vector.tobytes()).hexdigest()
                if row.get("prototype_sha256") != expected:
                    raise ValueError(f"Prototype SHA256 mismatch at line {line_number}")
                vectors[f"{concept_id}::{row.get('canonical_value')}"] = torch.from_numpy(vector)
        return cls(vectors)


def compute_p0b_pair_consistency_loss(
    concept_feature: torch.Tensor,
    comparable_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """MSE to legal pair agreement; diagonal and unsupported pairs are excluded.

    Each mapping is (values, valid_mask). Scalar values use [B]; multilabel
    values/masks use [B,K], where agreement is only over jointly valid values.
    """
    if concept_feature.ndim != 2:
        raise ValueError("concept_feature must be [B,D]")
    batch = concept_feature.shape[0]
    if batch < 2:
        return concept_feature.new_zeros(())
    target_sum = concept_feature.new_zeros((batch, batch))
    target_count = concept_feature.new_zeros((batch, batch))
    for values, valid in comparable_targets.values():
        values, valid = values.to(concept_feature.device), valid.to(concept_feature.device, dtype=torch.bool)
        if values.ndim == 1:
            pair_valid = valid[:, None] & valid[None, :]
            agreement = (values[:, None] == values[None, :]).to(concept_feature.dtype)
        elif values.ndim == 2:
            joint = valid[:, None, :] & valid[None, :, :]
            pair_valid = joint.any(dim=-1)
            agreement = ((values[:, None, :] == values[None, :, :]).to(concept_feature.dtype) * joint).sum(dim=-1) / joint.sum(dim=-1).clamp_min(1)
        else:
            raise ValueError("P0-B consistency values must be [B] or [B,K]")
        target_sum += agreement * pair_valid
        target_count += pair_valid
    pair_mask = target_count > 0
    pair_mask.fill_diagonal_(False)
    if not bool(pair_mask.any()):
        return concept_feature.new_zeros(())
    target = target_sum / target_count.clamp_min(1)
    similarity = 0.5 * (F.normalize(concept_feature, dim=-1) @ F.normalize(concept_feature, dim=-1).T + 1.0)
    return ((similarity - target).pow(2) * pair_mask).sum() / pair_mask.sum().clamp_min(1)


def compute_p0b_prototype_loss(
    concept_feature: torch.Tensor,
    projected_prototypes: dict[str, torch.Tensor],
    positive_support: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Average positive-value losses within each case/concept before global mean."""
    terms: list[torch.Tensor] = []
    for concept_id, supports in positive_support.items():
        prototype = projected_prototypes.get(concept_id)
        if prototype is None:
            continue
        support = supports.to(concept_feature.device, dtype=torch.bool)
        if support.ndim != 2 or prototype.ndim != 2:
            raise ValueError("Prototype support/projection must be [B,K] and [K,D]")
        if prototype.shape != (support.shape[1], concept_feature.shape[1]):
            raise ValueError(f"Prototype dimension mismatch for {concept_id}")
        cosine = F.normalize(concept_feature, dim=-1) @ F.normalize(prototype, dim=-1).T
        per_case = ((1.0 - cosine) * support).sum(dim=1) / support.sum(dim=1).clamp_min(1)
        terms.append(per_case[support.any(dim=1)])
    nonempty = [term for term in terms if term.numel()]
    return torch.cat(nonempty).mean() if nonempty else concept_feature.new_zeros(())


def combine_p0b_cc_loss(pair_loss: torch.Tensor, prototype_loss: torch.Tensor, *, pair_available: bool, prototype_available: bool) -> torch.Tensor:
    terms = [term for term, available in ((pair_loss, pair_available), (prototype_loss, prototype_available)) if available]
    return torch.stack(terms).mean() if terms else pair_loss.new_zeros(())
