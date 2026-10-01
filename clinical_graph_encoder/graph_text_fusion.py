from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from breast_pretrain.clinical_graph_encoder.hetero_graph_encoder import GraphEncoderOutput


class GraphTextFusion(nn.Module):
    def __init__(self, graph_dim: int, text_dim: int, fusion_weight: float = 0.10) -> None:
        super().__init__()
        self.graph_to_text = nn.Linear(graph_dim, text_dim)
        self.fusion_weight = float(fusion_weight)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_output: GraphEncoderOutput,
    ) -> torch.Tensor:
        if self.fusion_weight <= 0.0:
            return text_embeddings
        if graph_output.graph_pooled_embeddings is not None:
            graph_context = graph_output.graph_pooled_embeddings.to(
                device=text_embeddings.device,
                dtype=text_embeddings.dtype,
            )
            if graph_context.ndim != 2 or int(graph_context.shape[0]) != int(text_embeddings.shape[0]):
                raise ValueError("Observed graph pooled embeddings must have shape [B, H] aligned with text embeddings.")
            observed_rows = (
                graph_output.node_observed_mask.any(dim=1, keepdim=True)
                if graph_output.node_observed_mask is not None
                else torch.ones((graph_context.shape[0], 1), dtype=torch.bool, device=graph_context.device)
            )
            enhanced = text_embeddings + self.fusion_weight * self.graph_to_text(graph_context) * observed_rows.to(
                dtype=text_embeddings.dtype
            )
            normalized = F.normalize(enhanced, dim=-1)
            return torch.where(observed_rows, normalized, text_embeddings)
        prototype_pool = []
        for value in graph_output.graph_concept_prototypes.values():
            prototype_pool.append(value.mean(dim=0))
        if not prototype_pool:
            return text_embeddings
        graph_context = torch.stack(prototype_pool, dim=0).mean(dim=0)
        graph_context = self.graph_to_text(graph_context).view(1, -1)
        enhanced = text_embeddings + self.fusion_weight * graph_context.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        return F.normalize(enhanced, dim=-1)
