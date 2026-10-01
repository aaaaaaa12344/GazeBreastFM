from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_bool(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {raw_value!r}.")


def _resolve_optional_path(raw_value: Any, project_root: Path) -> Path | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _tuple(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple)):
        return tuple(str(item).strip() for item in raw_value if str(item).strip())
    value = str(raw_value).strip()
    return (value,) if value else ()


@dataclass(frozen=True)
class GraphEncoderConfig:
    enabled: bool = False
    source_model_family: str = "adapted_from_ultrasound_clip_udaf_heterogeneous_graph_encoder_for_tri_modal_breast_graph"
    implementation: str = "vendored_in_repo"
    mode: str = "stage1_semantic_graph_encoder"
    nodes_path: Path | None = None
    edges_path: Path | None = None
    graph_tensor_path: Path | None = None
    node_feature_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    semantic_prior_weight: float = 0.15
    text_fusion_weight: float = 0.10
    # Defaults to true to preserve historical experiment behavior. Formal V6
    # must explicitly disable this because its alignment target is the raw
    # Effective Report embedding, not a graph-enhanced representation.
    alignment_graph_text_fusion_enabled: bool = True
    fusion_target: tuple[str, ...] = (
        "semantic_soft_labels",
        "concept_prototypes",
        "graph_consistency",
    )
    forbid_image_region_graph_node_alignment: bool = True

    @classmethod
    def from_mapping(
        cls,
        raw_block: dict[str, Any] | None,
        *,
        project_root: Path,
        clinical_graph_nodes_path: Path | None = None,
        clinical_graph_edges_path: Path | None = None,
    ) -> "GraphEncoderConfig | None":
        if raw_block is None:
            return None
        if not isinstance(raw_block, dict):
            raise ValueError("graph_encoder must be a mapping when provided.")

        nodes_path = _resolve_optional_path(raw_block.get("nodes_path"), project_root) or clinical_graph_nodes_path
        edges_path = _resolve_optional_path(raw_block.get("edges_path"), project_root) or clinical_graph_edges_path
        config = cls(
            enabled=_as_bool(raw_block.get("enabled"), default=False),
            source_model_family=str(
                raw_block.get(
                    "source_model_family",
                    "adapted_from_ultrasound_clip_udaf_heterogeneous_graph_encoder_for_tri_modal_breast_graph",
                )
            ).strip(),
            implementation=str(raw_block.get("implementation", "vendored_in_repo")).strip(),
            mode=str(raw_block.get("mode", "stage1_semantic_graph_encoder")).strip(),
            nodes_path=nodes_path,
            edges_path=edges_path,
            graph_tensor_path=_resolve_optional_path(raw_block.get("graph_tensor_path"), project_root),
            node_feature_dim=int(raw_block.get("node_feature_dim", 64)),
            hidden_dim=int(raw_block.get("hidden_dim", 64)),
            num_layers=max(1, int(raw_block.get("num_layers", 2))),
            dropout=float(raw_block.get("dropout", 0.0)),
            semantic_prior_weight=float(raw_block.get("semantic_prior_weight", 0.15)),
            text_fusion_weight=float(raw_block.get("text_fusion_weight", 0.10)),
            alignment_graph_text_fusion_enabled=_as_bool(
                raw_block.get("alignment_graph_text_fusion_enabled"),
                default=True,
            ),
            fusion_target=_tuple(raw_block.get("fusion_target"))
            or ("semantic_soft_labels", "concept_prototypes", "graph_consistency"),
            forbid_image_region_graph_node_alignment=_as_bool(
                raw_block.get("forbid_image_region_graph_node_alignment"),
                default=True,
            ),
        )
        if config.enabled:
            config.validate()
        return config

    def validate(self) -> None:
        if self.implementation != "vendored_in_repo":
            raise ValueError("graph_encoder.implementation must be vendored_in_repo.")
        if self.mode != "stage1_semantic_graph_encoder":
            raise ValueError("graph_encoder.mode must be stage1_semantic_graph_encoder.")
        if not self.forbid_image_region_graph_node_alignment:
            raise ValueError("graph_encoder must forbid direct image-region graph-node alignment.")
        if self.nodes_path is None or self.edges_path is None:
            raise ValueError("graph_encoder requires nodes_path and edges_path.")
        if self.node_feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("graph_encoder dimensions must be positive.")
