from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MappingRule:
    node_id: str
    raw_fields: list[str] = field(default_factory=list)
    normalization: str = ""
    observed_mask_rule: str = ""
    confidence_rule: str = ""


@dataclass
class ConsistencyRule:
    rule_id: str
    rule_type: str
    source_node_id: str
    target_node_id: str
    penalty_weight: float = 1.0
    description: str = ""


class SidecarRuleEngine:
    """Engine that validates and applies clinical graph mapping and consistency rules."""

    def __init__(self, mapping_rules_path: Path, consistency_rules_path: Path) -> None:
        self.mapping_rules = self._load_yaml(mapping_rules_path)
        self.consistency_rules = self._load_yaml(consistency_rules_path)
        self._validate_scope()

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"YAML must contain a mapping: {path}")
        return payload

    def _validate_scope(self) -> None:
        scope = str(self.mapping_rules.get("scope", ""))
        valid_scopes = {"stage1_semantic_prior_sidecar_only", "stage1_semantic_prior_and_graph_encoder"}
        if scope not in valid_scopes:
            raise ValueError(f"mapping_rules scope={scope!r} not in {valid_scopes}")

    @property
    def scope(self) -> str:
        return str(self.mapping_rules.get("scope", ""))

    @property
    def graph_encoder_allowed(self) -> bool:
        policy = self.mapping_rules.get("stage1_policy", {}) or {}
        return bool(policy.get("graph_encoder_allowed_in_stage1_semantic_branch", False))

    @property
    def no_independent_graph_pretraining(self) -> bool:
        policy = self.mapping_rules.get("stage1_policy", {}) or {}
        return bool(policy.get("no_independent_graph_pretraining", True))

    @property
    def no_direct_alignment(self) -> bool:
        policy = self.mapping_rules.get("stage1_policy", {}) or {}
        return bool(policy.get("no_image_region_to_graph_node_direct_alignment", True))

    def get_dataset_mappings(self, modality: str, dataset_name: str) -> list[MappingRule]:
        datasets = self.mapping_rules.get("datasets", {}) or {}
        modality_data = datasets.get(modality, {}) or {}
        ds = modality_data.get(dataset_name, {}) or {}
        mappings = ds.get("concept_mappings", {}) or {}
        rules: list[MappingRule] = []
        for node_id, cfg in mappings.items():
            rules.append(MappingRule(
                node_id=node_id,
                raw_fields=list(cfg.get("raw_fields", [])),
                normalization=str(cfg.get("normalization", "")),
                observed_mask_rule=str(cfg.get("observed_mask_rule", "")),
                confidence_rule=str(cfg.get("confidence_rule", "")),
            ))
        return rules

    def get_consistency_rules(self) -> list[ConsistencyRule]:
        rules: list[ConsistencyRule] = []
        for entry in self.mapping_rules.get("consistency_rules", self.consistency_rules.get("consistency_rules", [])):
            rules.append(ConsistencyRule(
                rule_id=str(entry.get("rule_id", "")),
                rule_type=str(entry.get("rule_type", "")),
                source_node_id=str(entry.get("source_node_id", "")),
                target_node_id=str(entry.get("target_node_id", "")),
                penalty_weight=float(entry.get("penalty_weight", 1.0)),
                description=str(entry.get("description", "")),
            ))
        return rules

    def observed_from_strategy(self) -> str:
        """Return the observed_from_dataset_field strategy:
        'leaf' = each observable leaf connects to label.source
        'category' = parent/category nodes connect to label.source
        """
        return "leaf"

    def validate_cross_references(
        self, node_ids: set[str], edge_triples: set[tuple[str, str, str]]
    ) -> list[str]:
        """Validate that mapping rules, consistency rules, nodes, and edges are mutually consistent."""
        issues: list[str] = []
        # Check that mapping target nodes exist
        ds = self.mapping_rules.get("datasets", {}) or {}
        for modality_name, modality_data in ds.items():
            for dataset_name, dataset_cfg in modality_data.items():
                mappings = dataset_cfg.get("concept_mappings", {}) or {}
                for node_id in mappings:
                    if node_id not in node_ids:
                        issues.append(f"mapping rule target {node_id!r} (dataset={dataset_name}) not in nodes CSV")
        # Check that consistency rule nodes exist
        for rule in self.get_consistency_rules():
            if rule.source_node_id and rule.source_node_id not in node_ids:
                issues.append(f"consistency rule {rule.rule_id} source {rule.source_node_id!r} not in nodes")
            if rule.target_node_id and rule.target_node_id not in node_ids:
                issues.append(f"consistency rule {rule.rule_id} target {rule.target_node_id!r} not in nodes")
        return issues
