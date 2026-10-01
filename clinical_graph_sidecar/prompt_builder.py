from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptTemplate:
    modality: str
    template: str
    fields: Any = field(default_factory=dict)
    example: str = ""


class PromptBuilder:
    """Builds structured clinical prompts from graph prompt templates and concept values."""

    def __init__(self, templates_path: Path) -> None:
        raw = yaml.safe_load(templates_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt templates must be a mapping: {templates_path}")
        self._raw = raw
        self._templates: dict[str, PromptTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        templates = self._raw.get("templates", {}) or {}
        for modality, cfg in templates.items():
            self._templates[modality] = PromptTemplate(
                modality=modality,
                template=str(cfg.get("template", "")),
                fields=cfg.get("fields", {}) or {},
                example=str(cfg.get("example", "")),
            )

    @property
    def scope(self) -> str:
        return str(self._raw.get("scope", ""))

    @property
    def policy(self) -> dict[str, Any]:
        return self._raw.get("policy", {}) or {}

    def get_template(self, modality: str) -> PromptTemplate | None:
        return self._templates.get(modality)

    def build_prompt(self, modality: str, concept_values: dict[str, str]) -> str:
        tmpl = self.get_template(modality)
        if tmpl is None:
            return ""
        missing_phrase = str(self.policy.get("missing_attribute_phrase", "unknown"))
        values: dict[str, str] = {}
        for field_name, field_cfg in tmpl.fields.items():
            node_ids = field_cfg.get("node_ids") or ([field_cfg.get("node_id")] if field_cfg.get("node_id") else [])
            resolved = missing_phrase
            for nid in node_ids:
                if nid in concept_values and concept_values[nid].strip():
                    resolved = concept_values[nid].strip()
                    break
            values[field_name] = resolved
        return tmpl.template.format(**values)

    def build_observed_prompt(
        self,
        modality: str,
        concept_values: dict[str, Any],
        observed_mask: dict[str, int],
    ) -> str:
        """Render V2 fields only when their chosen node is observed."""
        config = (self._raw.get("templates", {}) or {}).get(modality, {})
        if not config or "fields" not in config:
            return ""
        parts = [str(config.get("prefix", "")).strip()]
        for field in config.get("fields", []):
            for node_id in field.get("node_ids", []):
                if observed_mask.get(node_id) != 1 or concept_values.get(node_id) is None:
                    continue
                value = node_id.rsplit(".", 1)[-1].replace("_", " ")
                parts.append(f"{field['label']}={value}.")
                break
        return " ".join(part for part in parts if part)

    def validate_nodes_exist(self, node_ids: set[str]) -> list[str]:
        issues: list[str] = []
        for modality, tmpl in self._templates.items():
            if isinstance(tmpl.fields, dict):
                items = list(tmpl.fields.items())
            elif isinstance(tmpl.fields, list):
                items = []
                for index, field_cfg in enumerate(tmpl.fields):
                    if not isinstance(field_cfg, dict):
                        raise ValueError(f"prompt template {modality}.fields[{index}] must be a mapping")
                    items.append((str(field_cfg.get("label", index)), field_cfg))
            else:
                raise ValueError(f"prompt template {modality}.fields must be a mapping or list")
            for field_name, field_cfg in items:
                refs = field_cfg.get("node_ids") or ([field_cfg.get("node_id")] if field_cfg.get("node_id") else [])
                if not isinstance(refs, list) or not refs:
                    raise ValueError(f"prompt template {modality}.{field_name} requires non-empty node_ids")
                for nid in refs:
                    if nid not in node_ids:
                        issues.append(f"prompt template {modality}.{field_name} references unknown node {nid!r}")
        if issues:
            raise ValueError("; ".join(issues))
        return issues
