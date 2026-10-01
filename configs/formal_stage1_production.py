from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.data.stage1_v6_contract import V6_STANDARD_FILENAMES, validate_stage1_v6_bundle
from breast_pretrain.models.highres_backbone_registry import PREFLIGHT_HIGHRES_STUB_BACKEND
from breast_pretrain.train.stage1_joint.concept_head_policy import (
    parse_concept_head_policy,
    validate_concept_head_policy,
)
from breast_pretrain.train.stage1_joint.config_validation import FORMAL_SOURCE_INTENSITY_TOLERANCE
from breast_pretrain.train.stage1_joint.p0b_config_contract import validate_formal_p0b_mapping


FORMAL_STAGE1_BUNDLE_TOKEN = "FORMAL_STAGE1_BUNDLE"
FORMAL_BACKBONE_WEIGHTS_TOKEN = "FORMAL_BACKBONE_WEIGHTS"
PRODUCTION_STATUS = "formal_production"
MINIMAL_PATCH_ENCODER = "minimal_patch_encoder"
REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_BACKEND_NAMES = {
    "pretrained_visual_encoder",
    "local_torch_checkpoint",
    "generic_torch_checkpoint",
}

REQUIRED_BUNDLE_FILES = (
    *V6_STANDARD_FILENAMES.values(),
)
PROJECTION_METADATA_CANDIDATES = (
    "projection_metadata.json",
    "projection_metadata.jsonl",
    "stage1_gaze_projection_metadata.json",
    "stage1_gaze_projection_metadata.jsonl",
    "patch_gaze_weights_dynamic/projection_metadata.json",
    "patch_gaze_weights_dynamic/projection_metadata.jsonl",
)


@dataclass(frozen=True)
class FormalStage1ProductionResult:
    output_yaml: Path
    resolved_config_sha256: str
    required_paths_checked: tuple[str, ...]
    run_tier: str
    compliance_status: str


# Known config keys whose values are file or directory paths (relative to project_root).
_PATH_KEYS: frozenset[str] = frozenset({
    "evaluation_config",
    "image_manifest_path",
    "attention_map_dir",
    "teacher_latent_dir",
    "text_prompt_path",
    "output_dir",
    "pretrained_weight_path",
    "prompt_embedding_path",
    "semantic_soft_label_path",
    "semantic_soft_label_topk_path",
    "semantic_manifest_path",
    "birads_prior_manifest_path",
    "nodes_path",
    "edges_path",
    "mapping_rules_path",
    "consistency_rules_path",
    "prompt_templates_path",
    "sidecar_case_concept_vector_path",
    "concept_schema_path",
    "semantic_contract_path",
    "sparse_target_path",
    "concept_runtime_npz_path",
    "concept_runtime_manifest_path",
    "prototype_asset_path",
})


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Formal Stage 1 draft YAML must contain a mapping: {path}")
    return payload


def _replace_placeholders(value: Any, bundle_root: Path, backbone_weight_path: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_placeholders(item, bundle_root, backbone_weight_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_placeholders(item, bundle_root, backbone_weight_path) for item in value]
    if not isinstance(value, str):
        return value

    resolved = value.replace(FORMAL_STAGE1_BUNDLE_TOKEN, str(bundle_root))
    if FORMAL_BACKBONE_WEIGHTS_TOKEN in resolved:
        resolved = resolved.replace(FORMAL_BACKBONE_WEIGHTS_TOKEN, str(backbone_weight_path.parent))
    return resolved


def _find_formal_placeholders(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            hits.extend(_find_formal_placeholders(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_find_formal_placeholders(item, f"{path}[{index}]"))
    elif isinstance(value, str) and "FORMAL_" in value:
        hits.append(path or "<root>")
    return hits


def _require_file(path: Path, label: str, checked: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    checked.append(str(path))


def _require_dir(path: Path, label: str, checked: list[str]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    checked.append(str(path))


def _validate_bundle_paths(bundle_root: Path, backbone_weight_path: Path) -> list[str]:
    checked: list[str] = []
    _require_dir(bundle_root, "formal Stage 1 bundle root", checked)
    for relative in REQUIRED_BUNDLE_FILES:
        _require_file(bundle_root / relative, relative, checked)
    bundle_report = validate_stage1_v6_bundle(bundle_root)
    if bundle_report["status"] != "PASS_FORMAL":
        raise ValueError(
            "Formal Stage 1 production bundle must satisfy the V6 entry contract:\n  "
            + "\n  ".join(bundle_report["errors"])
        )
    _require_file(backbone_weight_path, "high-resolution visual encoder weight", checked)
    return checked


def _validate_v6_runtime_boundary(config: dict[str, Any]) -> None:
    if bool(config.get("require_teacher_latents", False)):
        raise ValueError("Formal Stage 1 production config must set require_teacher_latents=false.")
    teacher_latent_dir = str(config.get("teacher_latent_dir") or "").strip()
    if teacher_latent_dir:
        raise ValueError("Formal Stage 1 production config must not set teacher_latent_dir.")

    model_block = config.get("model") if isinstance(config.get("model"), dict) else {}
    if bool(model_block.get("allow_missing_pretrained_fallback", False)):
        raise ValueError("Formal Stage 1 production config must set model.allow_missing_pretrained_fallback=false.")
    backend = str(model_block.get("vision_encoder_name") or "").strip().lower()
    if backend == PREFLIGHT_HIGHRES_STUB_BACKEND:
        raise ValueError("Formal Stage 1 production config must not use preflight_highres_hierarchical_stub.")
    if backend == MINIMAL_PATCH_ENCODER or backend in WRAPPER_BACKEND_NAMES:
        raise ValueError(
            "Formal Stage 1 production config must use a real visual encoder backend, "
            f"not {backend!r}."
        )
    local_branch = model_block.get("local_high_conf_branch")
    if isinstance(local_branch, dict) and bool(local_branch.get("enabled", False)):
        if not bool(local_branch.get("local_branch_training_ready", False)):
            raise ValueError(
                "Formal Stage 1 production config cannot enable local_high_conf_branch "
                "until local ROI crop, stride-8 token shape, and loss-shape tests are recorded as training ready."
            )


def _validate_formal_concept_head_policy(config: dict[str, Any]) -> None:
    semantic = config.get("semantic") if isinstance(config.get("semantic"), dict) else {}
    losses = config.get("losses") if isinstance(config.get("losses"), dict) else {}
    # P0-B's frozen schema supersedes the historical confirmed-label policy.
    # The runtime uses sparse target masks, not the old metadata-head list.
    if isinstance(config.get("formal_p0b"), dict):
        if not tuple(semantic.get("active_concept_heads", ())):
            raise ValueError("BLOCKED_CONCEPT_SCHEMA: schema-derived formal head set is empty.")
        return
    policy = parse_concept_head_policy(semantic.get("concept_head_policy"))
    active_heads = tuple(str(item).strip() for item in semantic.get("active_concept_heads", ()) if str(item).strip())
    pending_heads = tuple(str(item).strip() for item in semantic.get("pending_concept_heads", ()) if str(item).strip())
    concept_head_weights = {
        str(key).strip(): float(value)
        for key, value in (losses.get("concept_head_weights") or {}).items()
        if str(key).strip()
    }
    concept_consistency_head_weights = {
        str(key).strip(): float(value)
        for key, value in (losses.get("concept_consistency_head_weights") or {}).items()
        if str(key).strip()
    }
    validate_concept_head_policy(
        policy=policy,
        active_heads=active_heads,
        pending_heads=pending_heads,
        concept_head_weights=concept_head_weights,
        concept_consistency_head_weights=concept_consistency_head_weights,
        run_tier=PRODUCTION_STATUS,
    )


def _mark_production_ready_candidate(config: dict[str, Any]) -> None:
    config["run_tier"] = PRODUCTION_STATUS
    config["compliance_status"] = PRODUCTION_STATUS
    config["entry_contract_version"] = "v6"
    metadata = config.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["run_tier"] = PRODUCTION_STATUS
        metadata["compliance_status"] = PRODUCTION_STATUS


def _resolve_config_paths(config: dict[str, Any], project_root: Path) -> None:
    """Convert all known relative path values to absolute in-place."""
    resolved_project = str(project_root.resolve())
    config["project_root"] = resolved_project

    def _walk(d: dict[str, Any]) -> None:
        for key, value in d.items():
            if key in _PATH_KEYS and isinstance(value, str) and value.strip():
                p = Path(value)
                if not p.is_absolute():
                    d[key] = str((project_root / p).resolve())
                else:
                    d[key] = str(p.resolve())
            elif isinstance(value, dict):
                _walk(value)

    _walk(config)


def _resolve_project_root(draft_path: Path, config: dict[str, Any]) -> Path:
    raw_root = str(config.get("project_root", "")).strip()
    if raw_root:
        if raw_root.replace("\\", "/") == "../..":
            return REPO_ROOT.resolve()
        p = Path(raw_root)
        if not p.is_absolute():
            p = (draft_path.parent / p).resolve()
        return p
    return draft_path.parent.resolve()


def _validate_all_paths_from_bundle_root(config: dict[str, Any], bundle_root: Path) -> None:
    """Ensure all bundle-relative paths resolve to the same formal_bundle_root."""
    resolved_bundle_root = bundle_root.resolve()
    bundle_key_map = {
        "image_manifest_path": "image manifest",
        "text_prompt_path": "text prompts",
        "semantic.prompt_embedding_path": "prompt embeddings",
        "semantic.semantic_soft_label_path": "semantic soft labels",
        "semantic.semantic_soft_label_topk_path": "semantic soft labels top-k",
        "semantic.semantic_manifest_path": "semantic manifest",
        "semantic.birads_prior_manifest_path": "BI-RADS prior manifest",
        "clinical_graph.nodes_path": "clinical graph nodes",
        "clinical_graph.edges_path": "clinical graph edges",
        "clinical_graph.sidecar_case_concept_vector_path": "clinical graph sidecar",
        "graph_encoder.nodes_path": "graph encoder nodes",
        "graph_encoder.edges_path": "graph encoder edges",
    }
    for key_path, label in bundle_key_map.items():
        value = _get_nested(config, key_path)
        if value is None:
            continue
        resolved_path = Path(str(value)).resolve()
        try:
            resolved_path.relative_to(resolved_bundle_root)
        except ValueError:
            raise ValueError(
                f"Bundle path mismatch: {label} ({key_path}) resolves to "
                f"{resolved_path}, which is outside the formal bundle root {resolved_bundle_root}. "
                "All Stage 1 bundle paths must come from the same formal_bundle_root."
            )


def _get_nested(config: dict[str, Any], dotted_key: str) -> str | None:
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return str(current) if current else None


def _inject_source_intensity_audit_block(
    config: dict[str, Any],
    artifact_dir: Path,
) -> None:
    """Ensure the source_intensity_audit block is present in the resolved config."""
    existing = config.get("source_intensity_audit")
    if isinstance(existing, dict) and bool(existing.get("required", False)):
        audit_path = str(artifact_dir / "source_intensity_audit.json")
        existing["audit_report_path"] = audit_path
        config["source_intensity_audit"] = existing


def _inject_formal_audit_defaults(config: dict[str, Any]) -> None:
    """Add production audit fields that do not change training data semantics."""
    config.setdefault("evaluation_config", "configs/evaluation/stage1_foundation_eval_smoke.yaml")
    semantic = config.setdefault("semantic", {})
    if isinstance(semantic, dict):
        semantic.setdefault("clinical_graph_scope", "stage1_semantic_prior_and_graph_encoder")


def _validate_source_intensity_audit_contract(config: dict[str, Any]) -> None:
    audit = config.get("source_intensity_audit")
    if not isinstance(audit, dict):
        raise ValueError("Formal Stage 1 draft must define source_intensity_audit.")
    if not bool(audit.get("required", False)):
        raise ValueError("Formal Stage 1 draft must set source_intensity_audit.required=true.")
    approved = float(audit.get("approved_tolerance", -1))
    if approved != FORMAL_SOURCE_INTENSITY_TOLERANCE:
        raise ValueError(
            "Formal Stage 1 draft must set source_intensity_audit.approved_tolerance "
            f"exactly {FORMAL_SOURCE_INTENSITY_TOLERANCE:g}; got {approved:g}."
        )


def _validate_visible_salient_floor_contract(config: dict[str, Any]) -> None:
    masking = config.get("masking")
    if not isinstance(masking, dict) or "min_visible_salient_fraction" not in masking:
        raise ValueError(
            "Formal Stage 1 draft must explicitly set masking.min_visible_salient_fraction."
        )
    try:
        floor = float(masking["min_visible_salient_fraction"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Formal Stage 1 masking.min_visible_salient_fraction must be a finite number in (0, 1]."
        ) from exc
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError(
            "Formal Stage 1 masking.min_visible_salient_fraction must be a finite number in (0, 1]."
        )


def build_formal_stage1_production_config(
    *,
    draft_yaml: str | Path,
    formal_stage1_bundle_root: str | Path,
    formal_backbone_weight_path: str | Path,
    output_yaml: str | Path,
) -> FormalStage1ProductionResult:
    draft_path = Path(draft_yaml).expanduser().resolve()
    bundle_root = Path(formal_stage1_bundle_root).expanduser().resolve()
    backbone_weight_path = Path(formal_backbone_weight_path).expanduser().resolve()
    output_path = Path(output_yaml).expanduser().resolve()

    raw_config = _load_yaml(draft_path)
    _validate_source_intensity_audit_contract(raw_config)
    _validate_visible_salient_floor_contract(raw_config)
    checked_paths: list[str] = list(_validate_bundle_paths(bundle_root, backbone_weight_path))
    resolved_config = _replace_placeholders(
        copy.deepcopy(raw_config),
        bundle_root=bundle_root,
        backbone_weight_path=backbone_weight_path,
    )
    if isinstance(resolved_config.get("model"), dict):
        resolved_config["model"]["pretrained_weight_path"] = str(backbone_weight_path)

    placeholder_hits = _find_formal_placeholders(resolved_config)
    if placeholder_hits:
        joined = ", ".join(placeholder_hits)
        raise ValueError(f"Unresolved FORMAL_* placeholder(s) remain in production config: {joined}")

    _inject_formal_audit_defaults(resolved_config)
    project_root = _resolve_project_root(draft_path, resolved_config)
    _resolve_config_paths(resolved_config, project_root)
    _inject_source_intensity_audit_block(resolved_config, Path(str(output_yaml)).parent)
    # -- write bundle root and backbone path explicitly -------------------------
    resolved_config["formal_bundle_root"] = str(bundle_root)
    resolved_config["formal_backbone_weight_path"] = str(backbone_weight_path)

    # -- validate all bundle-relative paths resolve to the same bundle_root ------
    _validate_all_paths_from_bundle_root(resolved_config, bundle_root)

    _validate_v6_runtime_boundary(resolved_config)
    # P0-B owns formal concept-head authority.  Legacy YAML head lists are only
    # compatibility views after this schema-derived validation succeeds.
    formal_heads, formal_dims = validate_formal_p0b_mapping(resolved_config)
    semantic_block = resolved_config.setdefault("semantic", {})
    if isinstance(semantic_block, dict):
        semantic_block["active_concept_heads"] = list(formal_heads)
        semantic_block["pending_concept_heads"] = []
        semantic_block["concept_head_output_dims"] = formal_dims
    losses_block = resolved_config.setdefault("losses", {})
    if isinstance(losses_block, dict):
        losses_block["concept_head_weights"] = {head: 1.0 for head in formal_heads}
        losses_block["concept_consistency_head_weights"] = {}
    _validate_formal_concept_head_policy(resolved_config)
    _mark_production_ready_candidate(resolved_config)

    yaml_text = yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=False)
    resolved_config_sha256 = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_text, encoding="utf-8")
    return FormalStage1ProductionResult(
        output_yaml=output_path,
        resolved_config_sha256=resolved_config_sha256,
        required_paths_checked=tuple(checked_paths),
        run_tier=PRODUCTION_STATUS,
        compliance_status=PRODUCTION_STATUS,
    )


__all__ = [
    "FormalStage1ProductionResult",
    "build_formal_stage1_production_config",
]
