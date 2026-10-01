from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from breast_pretrain.clinical_graph_encoder.schema_assets import load_clinical_vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "configs" / "clinical_graph" / "tri_modal_clinical_graph_v1"
DIAGNOSTIC_CANONICAL_NODE_ROLE = "diagnostic_canonical"
ENGINEERING_OR_PROVENANCE_NODE_ROLE = "engineering_or_provenance"
VALID_V2_NODE_ROLES = frozenset(
    {
        DIAGNOSTIC_CANONICAL_NODE_ROLE,
        ENGINEERING_OR_PROVENANCE_NODE_ROLE,
    }
)


@dataclass(frozen=True)
class ClinicalGraphTensor:
    node_ids: tuple[str, ...]
    canonical_values: tuple[str, ...]
    node_roles: tuple[str, ...]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_type: torch.Tensor
    modality_scope: torch.Tensor
    relation_types: tuple[str, ...]
    node_type_vocab: tuple[str, ...]
    modality_scope_vocab: tuple[str, ...]
    semantic_validation: dict[str, Any]
    schema_version: str = "tri_modal_clinical_graph_v1"
    vocabulary_schema_version: str = "tri_modal_clinical_graph_v1"

    @property
    def diagnostic_semantic_eligibility_mask(self) -> torch.Tensor:
        """Return the canonical node mask allowed in the formal semantic graph."""

        if not self.schema_version.startswith("tri_modal_clinical_graph_v2"):
            return torch.ones(len(self.node_ids), dtype=torch.bool)
        return torch.tensor(
            [role == DIAGNOSTIC_CANONICAL_NODE_ROLE for role in self.node_roles],
            dtype=torch.bool,
        )


@dataclass(frozen=True)
class ObservedClinicalGraphTensor:
    """A batch of evidence-only induced subgraphs over one canonical schema.

    Nodes and edges keep canonical indices for stable schema compatibility, but
    ``observed_mask`` and ``edge_mask`` are hard masks.  Consumers must apply
    those masks at every message-passing and pooling operation.
    """

    base_graph: ClinicalGraphTensor
    node_values: torch.Tensor
    observed_mask: torch.Tensor
    edge_mask: torch.Tensor

    @property
    def node_ids(self) -> tuple[str, ...]:
        return self.base_graph.node_ids


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stable_hash_feature(text: str, dim: int) -> torch.Tensor:
    values = []
    seed = text.encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    while len(values) < dim:
        for byte in digest:
            values.append((float(byte) / 127.5) - 1.0)
            if len(values) >= dim:
                break
        digest = hashlib.sha256(digest).digest()
    tensor = torch.tensor(values, dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, dim=0)


def _node_text(row: dict[str, str]) -> str:
    return " | ".join(
        str(row.get(key, "")).strip()
        for key in ("node_id", "node_name", "node_type", "modality_scope", "canonical_value", "aliases", "description")
        if str(row.get(key, "")).strip()
    )


def _edge_exists(edges: list[dict[str, str]], source: str, target: str, relation: str) -> bool:
    return any(
        str(edge.get("source_node_id", "")).strip() == source
        and str(edge.get("target_node_id", "")).strip() == target
        and str(edge.get("relation_type", "")).strip() == relation
        for edge in edges
    )


def validate_graph_semantic_structure(
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    node_by_id = {str(row.get("node_id", "")).strip(): row for row in nodes}
    node_ids = set(node_by_id)

    required_node_fields = ("node_id", "node_type", "modality_scope", "canonical_value")
    for row in nodes:
        node_id = str(row.get("node_id", "")).strip()
        for field in required_node_fields:
            if not str(row.get(field, "")).strip():
                errors.append(f"node {node_id!r} is missing stable field {field}.")
    for row in edges:
        edge_id = str(row.get("edge_id", "")).strip()
        if not str(row.get("relation_type", "")).strip():
            errors.append(f"edge {edge_id!r} is missing relation_type.")

    is_v2 = any("node_role" in row for row in nodes)
    if is_v2:
        for row in nodes:
            node_id = str(row.get("node_id", "")).strip()
            if "node_role" not in row:
                errors.append(f"node {node_id!r} is missing node_role in a V2 schema.")
                continue
            node_role = str(row.get("node_role", "")).strip()
            if node_role not in VALID_V2_NODE_ROLES:
                errors.append(f"node {node_id!r} has invalid V2 node_role: {node_role!r}.")
        for edge in edges:
            edge_id = str(edge.get("edge_id", "")).strip()
            source = str(edge.get("source_node_id", "")).strip()
            target = str(edge.get("target_node_id", "")).strip()
            if source not in node_ids or target not in node_ids:
                errors.append(
                    f"edge {edge_id!r} references missing endpoint: {source!r}->{target!r}."
                )
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "semantic_structure_version": "tri_modal_graph_semantic_v2",
        }

    for required in ("laterality.left", "laterality.right", "laterality.bilateral"):
        if required not in node_ids:
            errors.append(f"required laterality node missing: {required}")
        elif not _edge_exists(edges, required, "laterality", "is_a"):
            errors.append(f"{required} must have child->parent is_a edge to laterality.")

    for required in ("benign_malignant.benign", "benign_malignant.malignant"):
        if required not in node_ids:
            errors.append(f"required benign/malignant node missing: {required}")
        elif not _edge_exists(edges, required, "benign_malignant", "is_a"):
            errors.append(f"{required} must have child->parent is_a edge to benign_malignant.")

    anatomical = node_by_id.get("anatomical_location")
    location_children = [node_id for node_id in node_ids if node_id.startswith("anatomical_location.")]
    if anatomical and not location_children:
        if str(anatomical.get("is_required_for_final", "")).strip().lower() == "true":
            errors.append("anatomical_location cannot be final-required without stable child location nodes.")
        else:
            warnings.append("anatomical_location is not final-required because stable fields are unavailable.")

    for index in range(7):
        node_id = f"mammography.assessment.birads_{index}"
        if node_id not in node_ids:
            errors.append(f"BI-RADS node missing: {node_id}")
        elif not _edge_exists(edges, node_id, "assessment", "is_a"):
            errors.append(f"{node_id} must have child->parent is_a edge to assessment.")

    invalid_density_edges = [
        edge.get("edge_id", "")
        for edge in edges
        if str(edge.get("source_node_id", "")).startswith("mammography.density.")
        and str(edge.get("target_node_id", "")).startswith("mammography.view.")
    ]
    if invalid_density_edges:
        errors.append(f"density nodes must not connect to mammography views: {invalid_density_edges}")
    for density in ("a", "b", "c", "d"):
        node_id = f"mammography.density.{density}"
        if not _edge_exists(edges, node_id, "breast_density", "is_a"):
            errors.append(f"{node_id} must connect to breast_density.")

    for response in (
        "complete_response",
        "partial_response",
        "stable_disease",
        "progressive_disease",
        "residual_disease",
    ):
        node_id = f"mri.treatment_response.{response}"
        if node_id not in node_ids:
            errors.append(f"MRI treatment response node missing: {node_id}")
        elif not (
            _edge_exists(edges, node_id, "treatment_response", "is_a")
            or _edge_exists(edges, node_id, "pathology_outcome", "maps_to_shared_concept")
        ):
            errors.append(f"{node_id} must map to treatment_response or pathology_outcome.")

    forbidden_birads_malignant = [
        edge.get("edge_id", "")
        for edge in edges
        if str(edge.get("source_node_id", "")) in {
            "mammography.assessment.birads_4",
            "mammography.assessment.birads_5",
        }
        and str(edge.get("target_node_id", "")) in {
            "benign_malignant",
            "benign_malignant.malignant",
        }
    ]
    if forbidden_birads_malignant:
        errors.append(f"BI-RADS 4/5 cannot directly map to malignant semantics: {forbidden_birads_malignant}")

    child_parent_relations = {
        "is_a",
        "maps_to_shared_concept",
        "has_attribute",
        "observed_from_dataset_field",
    }
    for edge in edges:
        rel = str(edge.get("relation_type", "")).strip()
        if rel not in child_parent_relations:
            continue
        src = str(edge.get("source_node_id", "")).strip()
        tgt = str(edge.get("target_node_id", "")).strip()
        source_node = node_by_id.get(src, {})
        target_node = node_by_id.get(tgt, {})
        if src in {"modality", "laterality", "assessment"} and tgt != "pathology_outcome":
            errors.append(f"edge {edge.get('edge_id')} has parent->child direction for {rel}: {src}->{tgt}")
        if rel == "is_a" and str(source_node.get("parent_node_id", "")).strip() and str(source_node.get("parent_node_id", "")).strip() != tgt:
            errors.append(f"edge {edge.get('edge_id')} conflicts with node parent_node_id for {src}: target={tgt}")
        if rel == "maps_to_shared_concept" and target_node and str(target_node.get("modality_scope", "")).strip() not in {"all", ""}:
            errors.append(f"edge {edge.get('edge_id')} maps to non-shared target {tgt}.")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "semantic_structure_version": "tri_modal_graph_semantic_v2",
    }


def materialize_graph_tensor(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    node_feature_dim: int = 64,
    parent_schema_root: str | Path | None = None,
) -> ClinicalGraphTensor:
    nodes_location, edges_location = Path(nodes_path), Path(edges_path)
    vocabulary_schema_version: str | None = None
    if nodes_location.is_dir() or edges_location.is_dir():
        if not nodes_location.is_dir() or not edges_location.is_dir() or nodes_location.resolve() != edges_location.resolve():
            raise ValueError("Clinical vocabulary overlay requires the same schema directory for nodes_path and edges_path.")
        # V2.2 is an additive overlay and must re-verify its complete read-only
        # V2/V2.1 parent authority; older schema directories ignore this option.
        vocabulary = load_clinical_vocabulary(
            nodes_location,
            parent_schema_root=Path(parent_schema_root) if parent_schema_root is not None else None,
        )
        nodes, edges = vocabulary.nodes, vocabulary.edges
        vocabulary_schema_version = vocabulary.vocabulary_version
    else:
        nodes, edges = _read_csv(nodes_location), _read_csv(edges_location)
    validation = validate_graph_semantic_structure(nodes, edges)
    if not validation["ok"]:
        raise ValueError("Clinical graph semantic validation failed: " + "; ".join(validation["errors"]))

    node_ids = tuple(str(row["node_id"]).strip() for row in nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_type_vocab = tuple(sorted({str(row["node_type"]).strip() for row in nodes}))
    modality_scope_vocab = tuple(sorted({str(row["modality_scope"]).strip() for row in nodes}))
    relation_types = tuple(sorted({str(row["relation_type"]).strip() for row in edges}))
    node_type_index = {value: index for index, value in enumerate(node_type_vocab)}
    modality_scope_index = {value: index for index, value in enumerate(modality_scope_vocab)}
    relation_type_index = {value: index for index, value in enumerate(relation_types)}

    edge_pairs: list[tuple[int, int]] = []
    edge_type_values: list[int] = []
    for edge in edges:
        edge_pairs.append((node_index[edge["source_node_id"].strip()], node_index[edge["target_node_id"].strip()]))
        edge_type_values.append(relation_type_index[edge["relation_type"].strip()])

    is_v2 = any("node_role" in row for row in nodes)
    schema_version = vocabulary_schema_version or (
        "tri_modal_clinical_graph_v2" if is_v2 else "tri_modal_clinical_graph_v1"
    )
    node_roles = (
        tuple(str(row["node_role"]).strip() for row in nodes)
        if is_v2
        else (DIAGNOSTIC_CANONICAL_NODE_ROLE,) * len(nodes)
    )
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return ClinicalGraphTensor(
        node_ids=node_ids,
        canonical_values=tuple(str(row["canonical_value"]).strip() for row in nodes),
        node_roles=node_roles,
        node_features=torch.stack([_stable_hash_feature(_node_text(row), node_feature_dim) for row in nodes], dim=0),
        edge_index=edge_index,
        edge_type=torch.tensor(edge_type_values, dtype=torch.long),
        node_type=torch.tensor([node_type_index[str(row["node_type"]).strip()] for row in nodes], dtype=torch.long),
        modality_scope=torch.tensor([modality_scope_index[str(row["modality_scope"]).strip()] for row in nodes], dtype=torch.long),
        relation_types=relation_types,
        node_type_vocab=node_type_vocab,
        modality_scope_vocab=modality_scope_vocab,
        semantic_validation=validation,
        schema_version=schema_version,
        vocabulary_schema_version=vocabulary_schema_version or schema_version,
    )


def materialize_observed_subgraph_tensor(
    graph_tensor: ClinicalGraphTensor,
    *,
    node_values: torch.Tensor,
    observed_mask: torch.Tensor,
    node_index: torch.Tensor,
) -> ObservedClinicalGraphTensor:
    """Build a per-sample observed subgraph contract over canonical graph indices."""

    node_count = len(graph_tensor.node_ids)
    if node_values.dtype != torch.float32 or observed_mask.dtype != torch.bool or node_index.dtype != torch.long:
        raise TypeError("Observed subgraph tensors require float32 node_values, bool observed_mask, and int64 node_index.")
    if node_values.ndim != 2 or observed_mask.ndim != 2 or node_index.ndim != 1:
        raise ValueError("Observed subgraph node_values/mask must be [B, N] and node_index must be [N].")
    if node_values.shape != observed_mask.shape or int(node_values.shape[1]) != node_count or int(node_index.numel()) != node_count:
        raise ValueError("Observed subgraph tensors do not match the canonical graph node count.")
    expected_index = torch.arange(node_count, device=node_index.device, dtype=torch.long)
    if not torch.equal(node_index, expected_index):
        raise ValueError("Observed subgraph node_index must match the canonical graph node order.")
    if not torch.isfinite(node_values).all():
        raise ValueError("Observed subgraph node_values must be finite.")

    normalized_mask = observed_mask.to(dtype=torch.bool)
    normalized_values = node_values.to(dtype=torch.float32) * normalized_mask.to(dtype=torch.float32)
    edge_index = graph_tensor.edge_index.to(device=normalized_mask.device)
    if edge_index.numel() == 0:
        edge_mask = torch.zeros(
            (int(normalized_mask.shape[0]), 0),
            dtype=torch.bool,
            device=normalized_mask.device,
        )
    else:
        edge_mask = normalized_mask[:, edge_index[0]] & normalized_mask[:, edge_index[1]]
    return ObservedClinicalGraphTensor(
        base_graph=graph_tensor,
        node_values=normalized_values,
        observed_mask=normalized_mask,
        edge_mask=edge_mask,
    )


def _default_paths(graph_dir: Path) -> tuple[Path, Path]:
    if (graph_dir / "schema_manifest_v2_2.json").is_file():
        return graph_dir, graph_dir
    if (graph_dir / "schema_manifest_v2_1.json").is_file():
        return graph_dir, graph_dir
    v2_nodes = graph_dir / "breast_multimodal_nodes_v2.csv"
    v2_edges = graph_dir / "breast_multimodal_edges_v2.csv"
    if v2_nodes.is_file() and v2_edges.is_file():
        return v2_nodes, v2_edges
    return graph_dir / "breast_multimodal_nodes_v1.csv", graph_dir / "breast_multimodal_edges_v1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize tri-modal clinical graph tensors.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--parent-schema-root",
        type=Path,
        help="Read-only complete V2/V2.1 authority root; required only for V2.2.",
    )
    parser.add_argument("--node-feature-dim", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    nodes_path, edges_path = _default_paths(args.graph_dir)
    tensor = materialize_graph_tensor(
        nodes_path,
        edges_path,
        node_feature_dim=args.node_feature_dim,
        parent_schema_root=args.parent_schema_root,
    )
    print(json.dumps({
        "ok": True,
        "nodes": len(tensor.node_ids),
        "edges": int(tensor.edge_index.shape[1]),
        "node_features": list(tensor.node_features.shape),
        "relation_types": list(tensor.relation_types),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
