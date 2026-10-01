from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from breast_pretrain.clinical_graph_encoder.config import GraphEncoderConfig
from breast_pretrain.clinical_graph_encoder.graph_tensor_builder import (
    ClinicalGraphTensor,
    ObservedClinicalGraphTensor,
)
from breast_pretrain.text.clinical_concepts import (
    BENIGN_MALIGNANT_LABELS,
    BIRADS_LABELS,
    DENSITY_LABELS,
    FINDING_LABELS,
    LATERALITY_LABELS,
    MRI_SEQUENCE_LABELS,
    MRI_TREATMENT_RESPONSE_LABELS,
    VIEW_LABELS,
)


@dataclass(frozen=True)
class GraphEncoderOutput:
    graph_node_embeddings: torch.Tensor
    graph_concept_prototypes: dict[str, torch.Tensor]
    graph_semantic_prior_matrix: torch.Tensor
    graph_enhanced_text_embeddings: torch.Tensor | None = None
    node_ids: tuple[str, ...] = ()
    node_observed_mask: torch.Tensor | None = None
    graph_pooled_embeddings: torch.Tensor | None = None


def _node_lookup(tensor: ClinicalGraphTensor) -> dict[str, int]:
    return {node_id: index for index, node_id in enumerate(tensor.node_ids)}


def _prototype_indices(tensor: ClinicalGraphTensor) -> dict[str, list[int]]:
    lookup = _node_lookup(tensor)

    def idx(node_id: str, fallback: str | None = None) -> int:
        if node_id in lookup:
            return lookup[node_id]
        if fallback and fallback in lookup:
            return lookup[fallback]
        raise KeyError(f"Clinical graph node missing for concept prototype: {node_id}")

    return {
        "view": [idx("mammography.view.cc"), idx("mammography.view.mlo")],
        "laterality": [idx(f"laterality.{value}") for value in LATERALITY_LABELS],
        "density": [idx(f"mammography.density.{value.lower()}") for value in DENSITY_LABELS],
        "birads": [
            idx(f"mammography.assessment.birads_{label[0]}", "mammography.assessment.birads_4")
            for label in BIRADS_LABELS
        ],
        "finding": [
            idx(
                {
                    "no_finding": "finding.no_finding",
                    "mass": "mammography.finding.mass",
                    "calcification": "mammography.finding.calcification",
                    "asymmetry": "mammography.finding.asymmetry",
                    "architectural_distortion": "mammography.finding.architectural_distortion",
                    "distortion": "mammography.finding.architectural_distortion",
                    "other": "lesion",
                }[label]
            )
            for label in FINDING_LABELS
        ],
        "cancer_label": [idx("benign_malignant.malignant")],
        "benign_malignant_label": [
            idx(f"benign_malignant.{label}") for label in BENIGN_MALIGNANT_LABELS
        ],
        "mri_sequence": [
            idx(f"mri.sequence.{label}") for label in MRI_SEQUENCE_LABELS
        ],
        "mri_treatment_response": [
            idx(f"mri.treatment_response.{label}") for label in MRI_TREATMENT_RESPONSE_LABELS
        ],
    }


class HeterogeneousMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relation_types: int, dropout: float) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relation_types, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_embeddings if node_mask is None else node_embeddings * node_mask.unsqueeze(-1).to(node_embeddings.dtype)
        source, target = edge_index[0], edge_index[1]
        relation = self.relation_embedding(edge_type)
        if node_embeddings.ndim == 2:
            messages = node_embeddings[source] + relation
            reverse_messages = node_embeddings[target] + relation
            if edge_mask is not None:
                edge_weight = edge_mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype).view(-1, 1)
                messages = messages * edge_weight
                reverse_messages = reverse_messages * edge_weight
            aggregate = torch.zeros_like(node_embeddings)
            degree = torch.zeros((node_embeddings.shape[0], 1), device=node_embeddings.device, dtype=node_embeddings.dtype)
            aggregate.index_add_(0, target, messages)
            aggregate.index_add_(0, source, reverse_messages)
            if edge_mask is None:
                degree.index_add_(0, target, torch.ones_like(degree[target]))
                degree.index_add_(0, source, torch.ones_like(degree[source]))
            else:
                degree.index_add_(0, target, edge_weight)
                degree.index_add_(0, source, edge_weight)
        elif node_embeddings.ndim == 3:
            messages = node_embeddings[:, source, :] + relation.unsqueeze(0)
            reverse_messages = node_embeddings[:, target, :] + relation.unsqueeze(0)
            if edge_mask is None:
                edge_weight = torch.ones(
                    (node_embeddings.shape[0], edge_index.shape[1], 1),
                    device=node_embeddings.device,
                    dtype=node_embeddings.dtype,
                )
            else:
                if edge_mask.ndim != 2 or tuple(edge_mask.shape) != (node_embeddings.shape[0], edge_index.shape[1]):
                    raise ValueError("Observed graph edge_mask must have shape [B, E].")
                edge_weight = edge_mask.to(device=node_embeddings.device, dtype=node_embeddings.dtype).unsqueeze(-1)
            messages = messages * edge_weight
            reverse_messages = reverse_messages * edge_weight
            aggregate = torch.zeros_like(node_embeddings)
            degree = torch.zeros(
                (node_embeddings.shape[0], node_embeddings.shape[1], 1),
                device=node_embeddings.device,
                dtype=node_embeddings.dtype,
            )
            aggregate.index_add_(1, target, messages)
            aggregate.index_add_(1, source, reverse_messages)
            degree.index_add_(1, target, edge_weight)
            degree.index_add_(1, source, edge_weight)
        else:
            raise ValueError("Graph node embeddings must have shape [N, H] or [B, N, H].")
        aggregate = aggregate / degree.clamp_min(1.0)
        updated = self.update(torch.cat([node_embeddings, aggregate], dim=-1))
        result = self.norm(node_embeddings + updated)
        if node_mask is not None:
            if tuple(node_mask.shape) != tuple(result.shape[:-1]):
                raise ValueError("Observed graph node_mask must match graph node embedding leading dimensions.")
            result = result * node_mask.to(device=result.device, dtype=result.dtype).unsqueeze(-1)
        return result


class BreastHeterogeneousGraphEncoder(nn.Module):
    def __init__(self, config: GraphEncoderConfig, graph_tensor: ClinicalGraphTensor) -> None:
        super().__init__()
        self.config = config
        self.graph_tensor = graph_tensor
        self.node_feature_projection = nn.Linear(config.node_feature_dim, config.hidden_dim)
        self.node_value_projection = nn.Linear(1, config.hidden_dim, bias=False)
        self.node_type_embedding = nn.Embedding(len(graph_tensor.node_type_vocab), config.hidden_dim)
        self.modality_scope_embedding = nn.Embedding(len(graph_tensor.modality_scope_vocab), config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                HeterogeneousMessagePassingLayer(
                    hidden_dim=config.hidden_dim,
                    num_relation_types=len(graph_tensor.relation_types),
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self._prototype_indices = (
            {} if graph_tensor.schema_version == "tri_modal_clinical_graph_v2" else _prototype_indices(graph_tensor)
        )

    def forward(
        self,
        graph_tensor: ClinicalGraphTensor | ObservedClinicalGraphTensor | None = None,
    ) -> GraphEncoderOutput:
        input_tensor = graph_tensor or self.graph_tensor
        observed_graph = input_tensor if isinstance(input_tensor, ObservedClinicalGraphTensor) else None
        tensor = observed_graph.base_graph if observed_graph is not None else input_tensor
        if tensor.schema_version == "tri_modal_clinical_graph_v2" and observed_graph is None:
            raise ValueError("Clinical Graph V2 encoder requires a per-sample observed subgraph tensor.")
        device = self.node_feature_projection.weight.device
        node_features = tensor.node_features.to(device=device)
        edge_index = tensor.edge_index.to(device=device)
        edge_type = tensor.edge_type.to(device=device)
        node_type = tensor.node_type.to(device=device)
        modality_scope = tensor.modality_scope.to(device=device)

        node_values = None
        node_mask = None
        edge_mask = None
        if observed_graph is not None:
            node_values = observed_graph.node_values.to(device=device, dtype=torch.float32)
            node_mask = observed_graph.observed_mask.to(device=device, dtype=torch.bool)
            edge_mask = observed_graph.edge_mask.to(device=device, dtype=torch.bool)
            node_features = node_features.unsqueeze(0).expand(node_values.shape[0], -1, -1)

        embeddings = (
            self.node_feature_projection(node_features)
            + self.node_type_embedding(node_type)
            + self.modality_scope_embedding(modality_scope)
        )
        if node_values is not None:
            embeddings = embeddings + self.node_value_projection(node_values.unsqueeze(-1))
            embeddings = embeddings * node_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
        embeddings = F.gelu(embeddings)
        for layer in self.layers:
            embeddings = layer(
                embeddings,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_mask=edge_mask,
                node_mask=node_mask,
            )
        graph_node_embeddings = F.normalize(self.output_norm(embeddings), dim=-1)
        if node_mask is not None:
            graph_node_embeddings = graph_node_embeddings * node_mask.to(
                dtype=graph_node_embeddings.dtype
            ).unsqueeze(-1)
            prior = torch.matmul(graph_node_embeddings, graph_node_embeddings.transpose(1, 2))
            pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
            graph_semantic_prior_matrix = ((prior + 1.0) * 0.5).clamp(0.0, 1.0) * pair_mask.to(
                dtype=prior.dtype
            )
            observed_counts = node_mask.sum(dim=1, keepdim=True)
            pooled = graph_node_embeddings.sum(dim=1) / observed_counts.clamp_min(1).to(
                dtype=graph_node_embeddings.dtype
            )
            graph_pooled_embeddings = F.normalize(pooled, dim=-1) * (observed_counts > 0).to(
                dtype=pooled.dtype
            )
            return GraphEncoderOutput(
                graph_node_embeddings=graph_node_embeddings,
                graph_concept_prototypes={},
                graph_semantic_prior_matrix=graph_semantic_prior_matrix,
                node_ids=tensor.node_ids,
                node_observed_mask=node_mask,
                graph_pooled_embeddings=graph_pooled_embeddings,
            )
        prototypes = {
            head_name: graph_node_embeddings[
                torch.tensor(indices, device=device, dtype=torch.long)
            ]
            for head_name, indices in self._prototype_indices.items()
        }
        prior = torch.matmul(graph_node_embeddings, graph_node_embeddings.transpose(0, 1))
        graph_semantic_prior_matrix = ((prior + 1.0) * 0.5).clamp(0.0, 1.0)
        return GraphEncoderOutput(
            graph_node_embeddings=graph_node_embeddings,
            graph_concept_prototypes=prototypes,
            graph_semantic_prior_matrix=graph_semantic_prior_matrix,
            node_ids=tensor.node_ids,
        )


TriModalClinicalGraphEncoder = BreastHeterogeneousGraphEncoder
