"""Load immutable V2, V2.1 and additive V2.2/V2.3 clinical graph authorities."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


TREE_HASH_ALGORITHM = "clinical_graph_schema_tree_hash_v1"


@dataclass(frozen=True)
class ClinicalVocabularyAssets:
    vocabulary_version: str
    nodes: list[dict[str, str]]
    edges: list[dict[str, str]]
    schema_manifest_sha256: str = ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_file_inventory(root: Path) -> list[dict[str, str]]:
    """Return the complete, normalized regular-file closure for an authority."""
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
    if not files:
        raise ValueError(f"Clinical vocabulary authority tree is empty: {root}")
    return [{"relative_path": path.relative_to(root).as_posix(), "sha256": _sha256(path)} for path in files]


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for item in tree_file_inventory(root):
        relative = item["relative_path"].encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(item["sha256"]))
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_unique(rows: list[dict[str, str]], field: str, label: str) -> None:
    values = [str(row.get(field, "")).strip() for row in rows]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Clinical vocabulary {label} has missing or duplicate {field}.")


def _required(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Clinical vocabulary misses {label}")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Clinical vocabulary {label} is absent: {path}")
    return path


def _required_directory(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Clinical vocabulary misses {label}")
    path = (root / relative).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Clinical vocabulary {label} is absent: {path}")
    return path


def _validate_edges(nodes: list[dict[str, str]], edges: list[dict[str, str]]) -> None:
    node_ids = {str(row["node_id"]).strip() for row in nodes}
    if any(str(row.get("source_node_id", "")).strip() not in node_ids or str(row.get("target_node_id", "")).strip() not in node_ids for row in edges):
        raise ValueError("Clinical vocabulary edge references an unknown canonical node.")
    if any(not str(row.get("relation_type", "")).strip() for row in edges):
        raise ValueError("Clinical vocabulary edge has an empty relation_type.")


def _load_v2(root: Path) -> ClinicalVocabularyAssets:
    nodes_path = _required(root, "breast_multimodal_nodes_v2.csv", "V2 nodes")
    edges_path = _required(root, "breast_multimodal_edges_v2.csv", "V2 edges")
    nodes, edges = _read_csv(nodes_path), _read_csv(edges_path)
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2", nodes, edges)


def _load_v2_1(root: Path) -> ClinicalVocabularyAssets:
    manifest_path = _required(root, "schema_manifest_v2_1.json", "V2.1 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tri_modal_clinical_graph_v2_1":
        raise ValueError("Unsupported clinical vocabulary V2.1 schema version.")
    base_root = (root / str(manifest.get("base_schema_directory", ""))).resolve()
    base = _load_v2(base_root)
    for name, expected in (("breast_multimodal_nodes_v2.csv", manifest.get("base_nodes_sha256")), ("breast_multimodal_edges_v2.csv", manifest.get("base_edges_sha256"))):
        if _sha256(base_root / name) != expected:
            raise ValueError(f"Clinical vocabulary V2.1 base hash mismatch: {name}")
    additions = manifest.get("additions")
    if not isinstance(additions, dict):
        raise ValueError("Clinical vocabulary V2.1 additions are missing.")
    extra_nodes = _required(root, additions.get("nodes_path"), "V2.1 additive nodes")
    extra_edges = _required(root, additions.get("edges_path"), "V2.1 additive edges")
    if _sha256(extra_nodes) != additions.get("nodes_sha256") or _sha256(extra_edges) != additions.get("edges_sha256"):
        raise ValueError("Clinical vocabulary V2.1 additive file hash mismatch.")
    nodes, edges = [*base.nodes, *_read_csv(extra_nodes)], [*base.edges, *_read_csv(extra_edges)]
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2_1", nodes, edges, _sha256(manifest_path))


def _load_v2_2(root: Path, parent_schema_root: Path | None) -> ClinicalVocabularyAssets:
    manifest_path = _required(root, "schema_manifest_v2_2.json", "V2.2 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tri_modal_clinical_graph_v2_2" or manifest.get("parent_tree_hash_algorithm") != TREE_HASH_ALGORITHM:
        raise ValueError("Unsupported V2.2 schema manifest or tree hash algorithm.")
    if parent_schema_root is None:
        raise ValueError("V2.2 requires an explicit read-only parent_schema_root authority.")
    parent_root = parent_schema_root.resolve()
    v2_root, v21_root = parent_root / "tri_modal_clinical_graph_v2", parent_root / "tri_modal_clinical_graph_v2_1"
    expected_trees = manifest.get("parent_tree_sha256")
    if not isinstance(expected_trees, dict) or tree_sha256(v2_root) != expected_trees.get("tri_modal_clinical_graph_v2") or tree_sha256(v21_root) != expected_trees.get("tri_modal_clinical_graph_v2_1"):
        raise ValueError("V2.2 parent schema tree authority mismatch.")
    parent = _load_v2_1(v21_root)
    additions = manifest.get("additions")
    if not isinstance(additions, dict):
        raise ValueError("Clinical vocabulary V2.2 additions are missing.")
    nodes_path = _required(root, additions.get("nodes_path"), "V2.2 additive nodes")
    edges_path = _required(root, additions.get("edges_path"), "V2.2 additive edges")
    if _sha256(nodes_path) != additions.get("nodes_sha256") or _sha256(edges_path) != additions.get("edges_sha256"):
        raise ValueError("Clinical vocabulary V2.2 additive file hash mismatch.")
    nodes, edges = [*parent.nodes, *_read_csv(nodes_path)], [*parent.edges, *_read_csv(edges_path)]
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2_2", nodes, edges, _sha256(manifest_path))


def _load_v2_3(root: Path, parent_schema_root: Path | None) -> ClinicalVocabularyAssets:
    """Load a sealed V2.3 append-only delta over the verified V2.2 authority."""
    manifest_path = _required(root, "schema_manifest_v2_3.json", "V2.3 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "tri_modal_clinical_graph_v2_3"
            or manifest.get("parent_schema") != "tri_modal_clinical_graph_v2_2"
            or manifest.get("parent_tree_hash_algorithm") != TREE_HASH_ALGORITHM):
        raise ValueError("Unsupported V2.3 schema manifest.")
    parent_root = _required_directory(root, manifest.get("parent_schema_directory"), "V2.3 parent schema directory")
    parent = _load_v2_2(parent_root, parent_schema_root)
    if _sha256(parent_root / "schema_manifest_v2_2.json") != manifest.get("parent_schema_manifest_sha256") or tree_sha256(parent_root) != manifest.get("parent_schema_tree_sha256"):
        raise ValueError("Clinical vocabulary V2.3 parent schema authority mismatch.")
    receipt_path = _required(root, manifest.get("schema_receipt_path"), "V2.3 schema receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "tri_modal_clinical_graph_v2_3" or receipt.get("schema_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Clinical vocabulary V2.3 receipt identity mismatch.")
    additions = manifest.get("additions")
    if not isinstance(additions, dict):
        raise ValueError("Clinical vocabulary V2.3 additions are missing.")
    nodes_path = _required(root, additions.get("nodes_path"), "V2.3 additive nodes")
    edges_path = _required(root, additions.get("edges_path"), "V2.3 additive edges")
    if _sha256(nodes_path) != additions.get("nodes_sha256") or _sha256(edges_path) != additions.get("edges_sha256"):
        raise ValueError("Clinical vocabulary V2.3 additive file hash mismatch.")
    nodes, edges = [*parent.nodes, *_read_csv(nodes_path)], [*parent.edges, *_read_csv(edges_path)]
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    expected_counts = manifest.get("expected_counts")
    if not isinstance(expected_counts, dict) or expected_counts.get("nodes") != len(nodes) or expected_counts.get("edges") != len(edges):
        raise ValueError("Clinical vocabulary V2.3 expected counts mismatch.")
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2_3", nodes, edges, _sha256(manifest_path))


def _load_v2_4(root: Path, parent_schema_root: Path | None) -> ClinicalVocabularyAssets:
    """Load the sealed V2.4 append-only delta over the verified V2.3 authority."""
    manifest_path = _required(root, "schema_manifest_v2_4.json", "V2.4 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "tri_modal_clinical_graph_v2_4"
            or manifest.get("parent_schema") != "tri_modal_clinical_graph_v2_3"
            or manifest.get("parent_tree_hash_algorithm") != TREE_HASH_ALGORITHM):
        raise ValueError("Unsupported V2.4 schema manifest.")
    parent_root = _required_directory(root, manifest.get("parent_schema_directory"), "V2.4 parent schema directory")
    parent = _load_v2_3(parent_root, parent_schema_root)
    if _sha256(parent_root / "schema_manifest_v2_3.json") != manifest.get("parent_schema_manifest_sha256") or tree_sha256(parent_root) != manifest.get("parent_schema_tree_sha256"):
        raise ValueError("Clinical vocabulary V2.4 parent schema authority mismatch.")
    receipt_path = _required(root, manifest.get("schema_receipt_path"), "V2.4 schema receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "tri_modal_clinical_graph_v2_4" or receipt.get("schema_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Clinical vocabulary V2.4 receipt identity mismatch.")
    additions = manifest.get("additions")
    if not isinstance(additions, dict):
        raise ValueError("Clinical vocabulary V2.4 additions are missing.")
    nodes_path = _required(root, additions.get("nodes_path"), "V2.4 additive nodes")
    edges_path = _required(root, additions.get("edges_path"), "V2.4 additive edges")
    if _sha256(nodes_path) != additions.get("nodes_sha256") or _sha256(edges_path) != additions.get("edges_sha256"):
        raise ValueError("Clinical vocabulary V2.4 additive file hash mismatch.")
    nodes, edges = [*parent.nodes, *_read_csv(nodes_path)], [*parent.edges, *_read_csv(edges_path)]
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    expected_counts = manifest.get("expected_counts")
    if not isinstance(expected_counts, dict) or expected_counts.get("nodes") != len(nodes) or expected_counts.get("edges") != len(edges):
        raise ValueError("Clinical vocabulary V2.4 expected counts mismatch.")
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2_4", nodes, edges, _sha256(manifest_path))


def _load_v2_5(root: Path, parent_schema_root: Path | None) -> ClinicalVocabularyAssets:
    """Load the sealed V2.5 append-only delta over the verified V2.4 authority."""
    manifest_path = _required(root, "schema_manifest_v2_5.json", "V2.5 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "tri_modal_clinical_graph_v2_5"
            or manifest.get("parent_schema") != "tri_modal_clinical_graph_v2_4"
            or manifest.get("parent_tree_hash_algorithm") != TREE_HASH_ALGORITHM):
        raise ValueError("Unsupported V2.5 schema manifest.")
    parent_root = _required_directory(root, manifest.get("parent_schema_directory"), "V2.5 parent schema directory")
    parent = _load_v2_4(parent_root, parent_schema_root)
    if _sha256(parent_root / "schema_manifest_v2_4.json") != manifest.get("parent_schema_manifest_sha256") or tree_sha256(parent_root) != manifest.get("parent_schema_tree_sha256"):
        raise ValueError("Clinical vocabulary V2.5 parent schema authority mismatch.")
    receipt_path = _required(root, manifest.get("schema_receipt_path"), "V2.5 schema receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "tri_modal_clinical_graph_v2_5" or receipt.get("schema_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Clinical vocabulary V2.5 receipt identity mismatch.")
    additions = manifest.get("additions")
    if not isinstance(additions, dict):
        raise ValueError("Clinical vocabulary V2.5 additions are missing.")
    nodes_path = _required(root, additions.get("nodes_path"), "V2.5 additive nodes")
    edges_path = _required(root, additions.get("edges_path"), "V2.5 additive edges")
    if _sha256(nodes_path) != additions.get("nodes_sha256") or _sha256(edges_path) != additions.get("edges_sha256"):
        raise ValueError("Clinical vocabulary V2.5 additive file hash mismatch.")
    nodes, edges = [*parent.nodes, *_read_csv(nodes_path)], [*parent.edges, *_read_csv(edges_path)]
    _validate_unique(nodes, "node_id", "nodes")
    _validate_unique(edges, "edge_id", "edges")
    _validate_edges(nodes, edges)
    expected_counts = manifest.get("expected_counts")
    if not isinstance(expected_counts, dict) or expected_counts.get("nodes") != len(nodes) or expected_counts.get("edges") != len(edges):
        raise ValueError("Clinical vocabulary V2.5 expected counts mismatch.")
    return ClinicalVocabularyAssets("tri_modal_clinical_graph_v2_5", nodes, edges, _sha256(manifest_path))


def load_clinical_vocabulary(path: Path, *, parent_schema_root: Path | None = None) -> ClinicalVocabularyAssets:
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"Clinical vocabulary must be a directory: {root}")
    if (root / "schema_manifest_v2_2.json").is_file():
        return _load_v2_2(root, parent_schema_root)
    if (root / "schema_manifest_v2_3.json").is_file():
        return _load_v2_3(root, parent_schema_root)
    if (root / "schema_manifest_v2_5.json").is_file():
        return _load_v2_5(root, parent_schema_root)
    if (root / "schema_manifest_v2_4.json").is_file():
        return _load_v2_4(root, parent_schema_root)
    if (root / "schema_manifest_v2_1.json").is_file():
        return _load_v2_1(root)
    if any(root.glob("schema_manifest_v*.json")):
        raise ValueError("Unsupported explicit clinical vocabulary schema manifest.")
    return _load_v2(root)
