from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from breast_pretrain.clinical_graph_encoder.config import GraphEncoderConfig
from breast_pretrain.clinical_graph_encoder.graph_tensor_builder import (
    DEFAULT_GRAPH_DIR,
    ClinicalGraphTensor,
    materialize_graph_tensor,
    materialize_observed_subgraph_tensor,
)
from breast_pretrain.clinical_graph_encoder.graph_text_fusion import GraphTextFusion
from breast_pretrain.clinical_graph_encoder.hetero_graph_encoder import (
    BreastHeterogeneousGraphEncoder,
    GraphEncoderOutput,
)


@dataclass
class Stage1GraphEncoderRuntime:
    graph_tensor: ClinicalGraphTensor
    encoder: BreastHeterogeneousGraphEncoder
    text_fusion: GraphTextFusion
    alignment_graph_text_fusion_enabled: bool

    def modules(self) -> tuple[torch.nn.Module, ...]:
        return (self.encoder, self.text_fusion)


def build_graph_encoder_runtime(
    config: object,
    device: torch.device,
) -> Stage1GraphEncoderRuntime | None:
    graph_config = getattr(config, "graph_encoder", None)
    if graph_config is None or not getattr(graph_config, "enabled", False):
        return None
    graph_config.validate()
    clinical_graph = getattr(config, "clinical_graph", None)
    if (
        clinical_graph is not None
        and str(getattr(clinical_graph, "version", "")).strip().lower()
        in {"clinical_graph_v2", "tri_modal_clinical_graph_v2"}
        and (
            graph_config.nodes_path != getattr(clinical_graph, "nodes_path", None)
            or graph_config.edges_path != getattr(clinical_graph, "edges_path", None)
        )
    ):
        raise ValueError("Clinical Graph V2 graph_encoder paths must match clinical_graph nodes_path and edges_path.")
    graph_tensor = materialize_graph_tensor(
        graph_config.nodes_path,
        graph_config.edges_path,
        node_feature_dim=graph_config.node_feature_dim,
    )
    encoder = BreastHeterogeneousGraphEncoder(graph_config, graph_tensor).to(device)
    text_fusion = GraphTextFusion(
        graph_dim=graph_config.hidden_dim,
        text_dim=config.model.text_dim,
        fusion_weight=graph_config.text_fusion_weight,
    ).to(device)
    return Stage1GraphEncoderRuntime(
        graph_tensor=graph_tensor,
        encoder=encoder,
        text_fusion=text_fusion,
        alignment_graph_text_fusion_enabled=graph_config.alignment_graph_text_fusion_enabled,
    )


def forward_graph_encoder(
    runtime: Stage1GraphEncoderRuntime | None,
    text_embeddings: torch.Tensor,
    prototype_head_names: tuple[str, ...] | None = None,
    *,
    clinical_graph_v2_node_values: torch.Tensor | None = None,
    clinical_graph_v2_observed_mask: torch.Tensor | None = None,
    clinical_graph_v2_node_index: torch.Tensor | None = None,
) -> tuple[GraphEncoderOutput | None, torch.Tensor, list[str]]:
    if runtime is None:
        return None, text_embeddings, []
    warnings: list[str] = []
    has_observed_subgraph = any(
        value is not None
        for value in (
            clinical_graph_v2_node_values,
            clinical_graph_v2_observed_mask,
            clinical_graph_v2_node_index,
        )
    )
    if runtime.graph_tensor.schema_version == "tri_modal_clinical_graph_v2":
        if not all(
            value is not None
            for value in (
                clinical_graph_v2_node_values,
                clinical_graph_v2_observed_mask,
                clinical_graph_v2_node_index,
            )
        ):
            raise ValueError("Clinical Graph V2 forward requires per-sample values, observed mask, and node index.")
        semantic_role_mask = runtime.graph_tensor.diagnostic_semantic_eligibility_mask.to(
            device=clinical_graph_v2_observed_mask.device,
        )
        semantic_observed_mask = clinical_graph_v2_observed_mask & semantic_role_mask.unsqueeze(0)
        semantic_node_values = clinical_graph_v2_node_values * semantic_observed_mask.to(
            dtype=clinical_graph_v2_node_values.dtype,
        )
        excluded_evidence = clinical_graph_v2_observed_mask & ~semantic_role_mask.unsqueeze(0)
        if excluded_evidence.any():
            warnings.append("graph_encoder_forward:engineering_or_provenance_evidence_excluded")
        if clinical_graph_v2_observed_mask.any() and not semantic_observed_mask.any():
            warnings.append("graph_encoder_forward:no_diagnostic_canonical_evidence")
        observed_graph = materialize_observed_subgraph_tensor(
            runtime.graph_tensor,
            node_values=semantic_node_values,
            observed_mask=semantic_observed_mask,
            node_index=clinical_graph_v2_node_index,
        )
        graph_output = runtime.encoder(observed_graph)
    else:
        if has_observed_subgraph:
            raise ValueError("Clinical Graph V2 observed-subgraph tensors cannot be consumed by a V1 graph runtime.")
        graph_output = runtime.encoder(runtime.graph_tensor)
    if runtime.alignment_graph_text_fusion_enabled:
        alignment_text_embeddings = runtime.text_fusion(text_embeddings, graph_output)
        warnings.append("graph_encoder_forward:semantic_text_fusion_applied")
    else:
        # Formal V6 keeps Effective Report embeddings unchanged for global,
        # visible, and semantic-soft image-report alignment. The graph output
        # remains available below for semantic priors, prototypes, and rules.
        alignment_text_embeddings = text_embeddings
        warnings.append("graph_encoder_forward:alignment_graph_text_fusion_disabled")
    graph_concept_prototypes = graph_output.graph_concept_prototypes
    if prototype_head_names is not None:
        prototype_heads = set(prototype_head_names)
        graph_concept_prototypes = {
            head_name: prototypes
            for head_name, prototypes in graph_concept_prototypes.items()
            if head_name in prototype_heads
        }
    graph_output = GraphEncoderOutput(
        graph_node_embeddings=graph_output.graph_node_embeddings,
        graph_concept_prototypes=graph_concept_prototypes,
        graph_semantic_prior_matrix=graph_output.graph_semantic_prior_matrix,
        graph_enhanced_text_embeddings=alignment_text_embeddings,
        node_ids=runtime.graph_tensor.node_ids,
        node_observed_mask=graph_output.node_observed_mask,
        graph_pooled_embeddings=graph_output.graph_pooled_embeddings,
    )
    return graph_output, alignment_text_embeddings, warnings


def _smoke_config(graph_dir: Path) -> object:
    nodes_path = graph_dir / "breast_multimodal_nodes_v2.csv"
    edges_path = graph_dir / "breast_multimodal_edges_v2.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        nodes_path = graph_dir / "breast_multimodal_nodes_v1.csv"
        edges_path = graph_dir / "breast_multimodal_edges_v1.csv"

    class _Model:
        text_dim = 5

    class _Config:
        model = _Model()
        graph_encoder = GraphEncoderConfig(
            enabled=True,
            nodes_path=nodes_path,
            edges_path=edges_path,
            node_feature_dim=32,
            hidden_dim=16,
            num_layers=1,
        )

    return _Config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke the Stage 1 clinical graph encoder forward path.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu")
    runtime = build_graph_encoder_runtime(_smoke_config(args.graph_dir), device=device)
    text_embeddings = torch.randn((2, 5), dtype=torch.float32)
    forward_kwargs = {}
    if runtime is not None and runtime.graph_tensor.schema_version == "tri_modal_clinical_graph_v2":
        node_count = len(runtime.graph_tensor.node_ids)
        forward_kwargs = {
            "clinical_graph_v2_node_values": torch.ones((2, node_count), dtype=torch.float32),
            "clinical_graph_v2_observed_mask": torch.ones((2, node_count), dtype=torch.bool),
            "clinical_graph_v2_node_index": torch.arange(node_count, dtype=torch.long),
        }
    graph_output, enhanced_text, warnings = forward_graph_encoder(runtime, text_embeddings, **forward_kwargs)
    assert graph_output is not None
    print(json.dumps({
        "ok": True,
        "graph_node_embeddings": list(graph_output.graph_node_embeddings.shape),
        "graph_semantic_prior_matrix": list(graph_output.graph_semantic_prior_matrix.shape),
        "graph_enhanced_text_embeddings": list(enhanced_text.shape),
        "warnings": warnings,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
