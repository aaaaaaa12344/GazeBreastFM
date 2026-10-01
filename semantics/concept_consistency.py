from __future__ import annotations

import torch
from torch.nn import functional as F

from breast_pretrain.semantics.birads_prior import BiradsPriorRecord
from breast_pretrain.text.clinical_concepts import (
    benign_malignant_to_index,
    birads_to_index,
    density_to_index,
    finding_labels_to_multi_hot,
    laterality_to_index,
    normalize_benign_malignant_label,
    normalize_birads,
    normalize_cancer_label,
    normalize_density,
    normalize_finding_labels,
    normalize_laterality,
    normalize_view,
    view_to_index,
)


def compute_concept_consistency_loss(
    image_embeddings: torch.Tensor,
    semantic_targets: torch.Tensor,
    prior_records: list[BiradsPriorRecord],
) -> torch.Tensor:
    """Prior-masked semantic similarity consistency on image embeddings only.

    This is intentionally not graph reasoning: BI-RADS priors only decide which
    pairwise semantic targets are trusted enough to regularize.
    """
    if image_embeddings.ndim != 2:
        raise ValueError(
            f"image_embeddings must have shape [batch, dim], got {tuple(image_embeddings.shape)}"
        )
    if semantic_targets.ndim != 2:
        raise ValueError(
            f"semantic_targets must have shape [batch, batch], got {tuple(semantic_targets.shape)}"
        )
    if semantic_targets.shape[0] != semantic_targets.shape[1]:
        raise ValueError("semantic_targets must be square.")
    if int(image_embeddings.shape[0]) != int(semantic_targets.shape[0]):
        raise ValueError(
            "image_embeddings batch size must match semantic_targets shape: "
            f"{tuple(image_embeddings.shape)} vs {tuple(semantic_targets.shape)}"
        )

    if image_embeddings.shape[0] < 2:
        return image_embeddings.new_zeros(())

    known_mask = torch.tensor(
        [record.has_any_known_concept() for record in prior_records],
        dtype=torch.bool,
        device=image_embeddings.device,
    )
    pair_mask = known_mask.unsqueeze(1) & known_mask.unsqueeze(0)
    if not bool(pair_mask.any().item()):
        return image_embeddings.new_zeros(())

    normalized = F.normalize(image_embeddings, dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity = 0.5 * (similarity + 1.0)
    target = semantic_targets.to(device=image_embeddings.device, dtype=similarity.dtype).clamp(0.0, 1.0)
    masked_error = (similarity - target).pow(2) * pair_mask.to(dtype=similarity.dtype)
    return masked_error.sum() / pair_mask.to(dtype=similarity.dtype).sum().clamp_min(1.0)


def compute_prior_head_consistency_loss(
    concept_logits: dict[str, torch.Tensor],
    prior_records: list[BiradsPriorRecord],
    head_weights: dict[str, float],
) -> torch.Tensor:
    if not concept_logits or not prior_records:
        reference_tensor = next(iter(concept_logits.values()), None)
        if reference_tensor is None:
            return torch.zeros((), dtype=torch.float32)
        return reference_tensor.new_zeros(())

    reference_tensor = next(iter(concept_logits.values()))
    weighted_terms: list[torch.Tensor] = []
    applied_weights: list[float] = []

    for head_name, logits in concept_logits.items():
        head_weight = float(head_weights.get(head_name, 0.0))
        if head_weight <= 0.0:
            continue

        if head_name == "finding":
            targets: list[list[float]] = []
            valid_mask: list[bool] = []
            for record in prior_records:
                finding_labels = normalize_finding_labels(record.finding)
                valid_mask.append(bool(finding_labels))
                targets.append(finding_labels_to_multi_hot(finding_labels) if finding_labels else [0.0] * logits.shape[1])
            valid_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=logits.device)
            if not bool(valid_tensor.any().item()):
                continue
            target_tensor = torch.tensor(targets, dtype=logits.dtype, device=logits.device)[valid_tensor]
            weighted_terms.append(F.binary_cross_entropy_with_logits(logits[valid_tensor], target_tensor))
            applied_weights.append(head_weight)
            continue

        raw_targets: list[int] = []
        valid_mask = []
        for record in prior_records:
            if head_name == "view":
                label, _ = normalize_view(record.view)
                valid_mask.append(bool(label))
                raw_targets.append(view_to_index(label) if label else 0)
            elif head_name == "laterality":
                label = normalize_laterality(record.laterality)
                valid_mask.append(bool(label))
                raw_targets.append(laterality_to_index(label) if label else 0)
            elif head_name == "density":
                label = normalize_density(record.density)
                valid_mask.append(bool(label))
                raw_targets.append(density_to_index(label) if label else 0)
            elif head_name == "birads":
                label = normalize_birads(record.birads)
                valid_mask.append(bool(label))
                raw_targets.append(birads_to_index(label) if label else 0)
            elif head_name == "cancer_label":
                label = normalize_cancer_label("")
                valid_mask.append(bool(label))
                raw_targets.append(int(label) if label else 0)
            elif head_name == "benign_malignant_label":
                label = normalize_benign_malignant_label("")
                valid_mask.append(bool(label))
                raw_targets.append(benign_malignant_to_index(label) if label else 0)
            else:
                valid_mask.append(False)
                raw_targets.append(0)

        valid_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=logits.device)
        if not bool(valid_tensor.any().item()):
            continue
        target_tensor = torch.tensor(raw_targets, dtype=torch.long, device=logits.device)[valid_tensor]
        weighted_terms.append(F.cross_entropy(logits[valid_tensor], target_tensor))
        applied_weights.append(head_weight)

    if not weighted_terms:
        return reference_tensor.new_zeros(())
    total_weight = sum(applied_weights)
    stacked_terms = torch.stack(
        [weight * term for weight, term in zip(applied_weights, weighted_terms)]
    )
    return stacked_terms.sum() / max(total_weight, 1e-6)
