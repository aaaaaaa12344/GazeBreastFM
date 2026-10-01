from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConceptVector:
    image_id: str
    concept_values: dict[str, str] = field(default_factory=dict)
    observed_masks: dict[str, int] = field(default_factory=dict)
    confidences: dict[str, str] = field(default_factory=dict)


class MatrixExport:
    """Exports concept vectors and matrices for clinical graph sidecar consumption."""

    @staticmethod
    def load_nodes(nodes_path: Path) -> list[dict[str, str]]:
        with nodes_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def load_edges(edges_path: Path) -> list[dict[str, str]]:
        with edges_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def export_concept_vectors(vectors: list[ConceptVector], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for vec in vectors:
                handle.write(json.dumps({
                    "image_id": vec.image_id,
                    "concept_values": vec.concept_values,
                    "observed_masks": vec.observed_masks,
                    "confidences": vec.confidences,
                }, ensure_ascii=True) + "\n")

    @staticmethod
    def validate_coverage(
        nodes: list[dict[str, str]],
        vectors: list[ConceptVector],
        required_only: bool = True,
    ) -> dict[str, Any]:
        node_ids = {n["node_id"] for n in nodes}
        required_nodes = {
            n["node_id"] for n in nodes
            if str(n.get("is_required_for_final", "")).lower() == "true"
        } if required_only else node_ids
        covered: set[str] = set()
        for vec in vectors:
            covered.update(vec.concept_values.keys())
        missing = required_nodes - covered
        return {
            "total_nodes": len(node_ids),
            "required_nodes": len(required_nodes),
            "covered_nodes": len(covered & required_nodes),
            "missing_required": sorted(missing),
            "coverage_ratio": len(covered & required_nodes) / max(1, len(required_nodes)),
        }

    @staticmethod
    def build_matrices(
        concept_vectors: list[dict[str, Any]],
        node_ids: list[str],
    ) -> tuple[Any, Any]:
        """Build separate concept-value and observed-mask matrices.

        Positive concept values are permitted only for observed, present nodes.
        Missing, unknown, and not-mentioned values never become positives.
        """
        import numpy as np

        concept_values = np.zeros((len(concept_vectors), len(node_ids)), dtype=np.float32)
        observed_masks = np.zeros((len(concept_vectors), len(node_ids)), dtype=np.uint8)
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        for row_index, vector in enumerate(concept_vectors):
            values = vector.get("concept_values", {})
            masks = vector.get("observed_mask", {})
            statuses = vector.get("status", {})
            for node_id, mask in masks.items():
                column = node_index.get(node_id)
                if column is None or int(mask) != 1:
                    continue
                observed_masks[row_index, column] = 1
                if statuses.get(node_id) in {"present", "observed", "confirmed"} and values.get(node_id) is not None:
                    concept_values[row_index, column] = 1.0
        return concept_values, observed_masks

    @staticmethod
    def summarize_matrices(
        concept_value_matrix: Any,
        observed_mask_matrix: Any,
        node_ids: list[str],
    ) -> dict[str, Any]:
        import numpy as np

        if observed_mask_matrix.ndim != 2 or concept_value_matrix.shape != observed_mask_matrix.shape:
            raise ValueError("Concept-value and observed-mask matrices must have equal rank-2 shape.")
        case_count, node_count = observed_mask_matrix.shape
        observed_counts = observed_mask_matrix.sum(axis=0)
        ordered = sorted(
            (
                {"node_id": node_ids[index], "observed_count": int(count)}
                for index, count in enumerate(observed_counts)
                if int(count) > 0
            ),
            key=lambda item: (-item["observed_count"], item["node_id"]),
        )
        return {
            "num_cases": int(case_count),
            "num_concept_nodes": int(node_count),
            "concept_value_matrix_shape": [int(case_count), int(node_count)],
            "observed_mask_matrix_shape": [int(case_count), int(node_count)],
            "concept_value_matrix_sparsity": float(np.mean(concept_value_matrix)) if case_count else 0.0,
            "observed_ratio_per_case": float(observed_mask_matrix.mean()) if case_count and node_count else 0.0,
            "mean_observed_node_count_per_case": float(observed_mask_matrix.sum(axis=1).mean()) if case_count else 0.0,
            "most_observed_nodes": ordered[:10],
        }

