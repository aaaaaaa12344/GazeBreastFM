from __future__ import annotations

import torch
from torch.nn import functional as F

from breast_pretrain.clinical_graph_encoder.hetero_graph_encoder import GraphEncoderOutput
from breast_pretrain.text.clinical_concepts import (
    BIRADS_LABELS,
    DENSITY_LABELS,
    FINDING_LABELS,
    LATERALITY_LABELS,
    MRI_SEQUENCE_LABELS,
    MRI_TREATMENT_RESPONSE_LABELS,
    VIEW_LABELS,
)


_CLASS_LABELS = {
    "view": VIEW_LABELS,
    "laterality": LATERALITY_LABELS,
    "density": DENSITY_LABELS,
    "birads": BIRADS_LABELS,
    "finding": FINDING_LABELS,
    "mri_sequence": MRI_SEQUENCE_LABELS,
    "mri_treatment_response": MRI_TREATMENT_RESPONSE_LABELS,
}


def _prototype_for_target(
    graph_output: GraphEncoderOutput,
    head_name: str,
    target: torch.Tensor,
) -> torch.Tensor | None:
    prototypes = graph_output.graph_concept_prototypes.get(head_name)
    if prototypes is None:
        return None
    if head_name == "cancer_label":
        return prototypes[0] if float(target.item()) > 0.5 else None
    if target.ndim == 0:
        index = int(target.item())
        if 0 <= index < int(prototypes.shape[0]):
            return prototypes[index]
        return None
    active = torch.nonzero(target > 0.5, as_tuple=False).flatten()
    if active.numel() <= 0:
        return None
    active = active.to(device=prototypes.device)
    return prototypes[active].mean(dim=0)


def build_graph_sample_embeddings(
    graph_output: GraphEncoderOutput,
    concept_targets: dict[str, torch.Tensor],
    concept_valid_masks: dict[str, torch.Tensor],
    semantic_batch_positions: list[int],
) -> torch.Tensor | None:
    if graph_output.graph_pooled_embeddings is not None:
        pooled = graph_output.graph_pooled_embeddings
        if pooled.ndim != 2:
            raise ValueError("Observed graph pooled embeddings must have shape [B, H].")
        if not semantic_batch_positions:
            return None
        positions = torch.tensor(
            semantic_batch_positions,
            device=pooled.device,
            dtype=torch.long,
        )
        if int(positions.max().item()) >= int(pooled.shape[0]):
            raise ValueError("Semantic batch positions exceed observed graph batch size.")
        return pooled[positions]
    sample_embeddings = []
    for batch_position in semantic_batch_positions:
        pieces = []
        for head_name in sorted(graph_output.graph_concept_prototypes):
            targets = concept_targets.get(head_name)
            masks = concept_valid_masks.get(head_name)
            if targets is None or masks is None or not bool(masks[batch_position].item()):
                continue
            prototype = _prototype_for_target(
                graph_output,
                head_name=head_name,
                target=targets[batch_position].detach().to(device=graph_output.graph_node_embeddings.device),
            )
            if prototype is not None:
                pieces.append(prototype)
        if pieces:
            sample_embeddings.append(torch.stack(pieces, dim=0).mean(dim=0))
        else:
            sample_embeddings.append(torch.zeros_like(graph_output.graph_node_embeddings[0]))
    if not sample_embeddings:
        return None
    return F.normalize(torch.stack(sample_embeddings, dim=0), dim=-1)


def _semantic_position_observed_mask(
    graph_output: GraphEncoderOutput,
    semantic_batch_positions: list[int],
) -> torch.Tensor | None:
    observed_mask = graph_output.node_observed_mask
    if observed_mask is None:
        return None
    if observed_mask.ndim != 2:
        raise ValueError("Observed graph node mask must have shape [B, N].")
    if not semantic_batch_positions:
        return None
    positions = torch.tensor(
        semantic_batch_positions,
        device=observed_mask.device,
        dtype=torch.long,
    )
    if int(positions.max().item()) >= int(observed_mask.shape[0]):
        raise ValueError("Semantic batch positions exceed observed graph batch size.")
    return observed_mask[positions].any(dim=1)


def blend_graph_semantic_prior(
    semantic_targets: torch.Tensor,
    graph_output: GraphEncoderOutput,
    concept_targets: dict[str, torch.Tensor],
    concept_valid_masks: dict[str, torch.Tensor],
    semantic_batch_positions: list[int],
    *,
    weight: float,
) -> torch.Tensor:
    if semantic_targets.numel() == 0 or weight <= 0.0:
        return semantic_targets
    sample_embeddings = build_graph_sample_embeddings(
        graph_output=graph_output,
        concept_targets=concept_targets,
        concept_valid_masks=concept_valid_masks,
        semantic_batch_positions=semantic_batch_positions,
    )
    if sample_embeddings is None:
        return semantic_targets
    graph_prior = torch.matmul(sample_embeddings, sample_embeddings.transpose(0, 1))
    graph_prior = ((graph_prior + 1.0) * 0.5).clamp(0.0, 1.0).to(
        device=semantic_targets.device,
        dtype=semantic_targets.dtype,
    )
    alpha = min(max(float(weight), 0.0), 1.0)
    blended = ((1.0 - alpha) * semantic_targets + alpha * graph_prior).clamp(0.0, 1.0)
    observed_rows = _semantic_position_observed_mask(graph_output, semantic_batch_positions)
    if observed_rows is None:
        blended.fill_diagonal_(1.0)
        return blended
    pair_mask = observed_rows.unsqueeze(1) & observed_rows.unsqueeze(0)
    graph_prior = graph_prior * pair_mask.to(dtype=graph_prior.dtype)
    blended = ((1.0 - alpha) * semantic_targets + alpha * graph_prior).clamp(0.0, 1.0)
    return torch.where(pair_mask.to(device=blended.device), blended, semantic_targets)
