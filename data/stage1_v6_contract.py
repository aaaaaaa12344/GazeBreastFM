"""Formal Stage 1 V6 validation, isolated from the legacy V5 entry contract."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from breast_pretrain.data.semantic_soft_label_contract import validate_semantic_soft_labels


V6_STANDARD_FILENAMES = {
    "manifest": "manifest_stage1_v6.csv",
    "accepted_reports": "accepted_effective_report_manifest.jsonl",
    "text_prompts": "text_prompts.jsonl",
    "embeddings": "stage1_prompt_embeddings.json",
    "concept_targets": "stage1_concept_targets.jsonl",
    "soft_labels": "stage1_semantic_soft_labels_topk.npz",
    "soft_labels_jsonl": "stage1_semantic_soft_labels_topk.jsonl",
    "soft_labels_manifest": "stage1_semantic_soft_label_manifest.csv",
    "birads_priors": "stage1_birads_prior_manifest.csv",
    "graph_nodes": "report_derived_graph_nodes.jsonl",
    "graph_edges": "report_derived_graph_edges.jsonl",
    "runtime_graph_nodes": "stage1_report_derived_graph_nodes.csv",
    "runtime_graph_edges": "stage1_report_derived_graph_edges.csv",
    "runtime_graph_sidecar": "clinical_graph_v6_case_sidecar.jsonl",
    "final_rejected": "final_rejected_case_manifest.jsonl",
    "discarded_images": "discarded_image_manifest.jsonl",
    "stage0_nonusable": "stage0_nonusable_image_manifest.jsonl",
}
_ACCEPTED = {"ACCEPTED_REAL_REPORT", "ACCEPTED_GENERATED_ORIGINAL", "ACCEPTED_GENERATED_AFTER_REPAIR", "MANUAL_ACCEPTED"}
_REQUIRED_COLUMNS = {
    "image_id", "image_path", "modality", "patient_id", "case_id", "report_unit_id", "split", "dataset_id",
    "source_image_sha256", "final_case_route", "train_eligible", "visual_training_enabled", "gaze_training_enabled",
    "text_semantic_enabled", "clinical_graph_enabled", "concept_target_enabled", "semantic_soft_label_enabled", "report_id",
    "effective_report_sha256", "text_prompt_key", "embedding_key", "graph_key", "patch_geometry_key", "patch_token_order_version",
    "valid_content_mask_key", "patch_asset_mode", "gaze_prior_available", "gaze_prior_quality",
    "stage0_usable",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STAGE0_NONUSABLE_HASH_KEYS = (
    "authority_sha256",
    "source_authority_sha256",
    "manifest_sha256",
    "source_sha256",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL item at {path}:{line_number} is not an object.")
            values.append(value)
    return values


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), set(reader.fieldnames or [])


def _enabled(row: dict[str, str], name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"1", "true", "yes"}


def _validate_stage0_nonusable_manifest(
    path: Path,
    image_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    nonusable_ids: set[str] = set()
    schema_versions: set[str] = set()
    authority_hashes: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            _append(errors, f"stage0_nonusable row {row_number} lacks image_id.")
            continue
        if image_id in nonusable_ids:
            _append(errors, f"stage0_nonusable has duplicate image_id: {image_id}")
        nonusable_ids.add(image_id)
        if _enabled(row, "stage0_usable"):
            _append(errors, f"stage0_nonusable image_id={image_id} must set stage0_usable=false.")
        schema_version = str(row.get("schema_version") or row.get("manifest_schema_version") or "").strip()
        if not schema_version:
            _append(warnings, f"stage0_nonusable image_id={image_id} lacks schema_version (historical audit metadata only).")
        else:
            schema_versions.add(schema_version)
        reason = str(
            row.get("exclusion_reason_code")
            or row.get("reason_code")
            or row.get("status")
            or row.get("route")
            or ""
        ).strip()
        if not reason:
            _append(warnings, f"stage0_nonusable image_id={image_id} lacks an exclusion reason/status (historical audit metadata only).")
        authority_hash = next(
            (str(row.get(key) or "").strip().lower() for key in _STAGE0_NONUSABLE_HASH_KEYS if str(row.get(key) or "").strip()),
            "",
        )
        if not _SHA256_RE.fullmatch(authority_hash):
            _append(errors, f"stage0_nonusable image_id={image_id} lacks a valid authority SHA256.")
        else:
            authority_hashes.add(authority_hash)
    if len(schema_versions) > 1:
        _append(warnings, "stage0_nonusable manifest contains inconsistent schema_version values (historical audit metadata only).")
    if len(authority_hashes) > 1:
        _append(warnings, "stage0_nonusable manifest contains multiple source authority SHA256 values (historical audit metadata only).")
    overlap = sorted(nonusable_ids & image_ids)
    if overlap:
        _append(
            errors,
            "stage0_nonusable image_ids overlap the formal V6 manifest: "
            + ", ".join(overlap[:5]),
        )
    return {
        "row_count": len(rows),
        "image_count": len(nonusable_ids),
        "formal_manifest_overlap_count": len(overlap),
        "schema_versions": sorted(schema_versions),
        "authority_sha256_count": len(authority_hashes),
    }


def _append(errors: list[str], message: str) -> None:
    if len(errors) < 200:
        errors.append(message)


def _config_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _validate_runtime_key(value: str, *, image_id: str, name: str, errors: list[str]) -> None:
    text = value.strip()
    if not text or "\x00" in text or len(text) > 4096:
        _append(errors, f"manifest image_id={image_id} has unparseable {name}.")
        return
    if name in {
        "patch_geometry_key",
        "valid_content_mask_key",
        "gaze_prior_key",
        "gaze_to_patch_projection_key",
    } and ":" not in text:
        _append(errors, f"manifest image_id={image_id} {name} must use a stable namespace.")


def _validate_concept_targets(
    path: Path,
    manifest_by_image: dict[str, dict[str, str]],
    manifest_hashes: dict[str, str],
    errors: list[str],
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    by_image = {str(row.get("image_id", "")).strip(): row for row in rows}
    image_ids = set(manifest_by_image)
    if len(by_image) != len(rows):
        _append(errors, "stage1_concept_targets has duplicate image_id values.")
    if set(by_image) != image_ids:
        _append(errors, "stage1_concept_targets image_id set differs from V6 manifest.")
    required_targets = {"view", "laterality", "density", "finding", "birads", "cancer_label", "benign_malignant_label"}
    for image_id, row in by_image.items():
        if str(row.get("effective_report_sha256", "")) != manifest_hashes.get(image_id, ""):
            _append(errors, f"concept target hash mismatch for image_id={image_id}.")
        fields = row.get("trainer_manifest_fields")
        if not isinstance(fields, dict) or not required_targets.issubset(fields):
            _append(errors, f"concept target image_id={image_id} does not expose the trainer manifest fields.")
        elif any(str(fields.get(field, "")) != str(manifest_by_image[image_id].get(field, "")) for field in required_targets):
            _append(errors, f"concept target image_id={image_id} does not match the trainer manifest values.")
        try:
            row_index = int(row.get("manifest_row_index"))
        except (TypeError, ValueError):
            _append(errors, f"concept target image_id={image_id} has invalid manifest_row_index.")
            continue
        if row_index < 0 or row_index >= len(image_ids):
            _append(errors, f"concept target image_id={image_id} has out-of-range manifest_row_index.")
    return {"row_count": len(rows), "trainer_format": "manifest_row_projection_v1"}


def _validate_runtime_graph(
    nodes_path: Path,
    edges_path: Path,
    sidecar_path: Path,
    image_ids: set[str],
    errors: list[str],
) -> dict[str, Any]:
    nodes, node_columns = _read_csv(nodes_path)
    edges, edge_columns = _read_csv(edges_path)
    required_nodes = {"node_id", "node_type", "modality_scope", "canonical_value", "node_role"}
    required_edges = {"edge_id", "source_node_id", "target_node_id", "relation_type"}
    if missing := sorted(required_nodes - node_columns):
        _append(errors, "runtime ClinicalGraph nodes miss columns: " + ", ".join(missing))
    if missing := sorted(required_edges - edge_columns):
        _append(errors, "runtime ClinicalGraph edges miss columns: " + ", ".join(missing))
    node_ids = [str(row.get("node_id", "")).strip() for row in nodes]
    if not node_ids or any(not item for item in node_ids) or len(set(node_ids)) != len(node_ids):
        _append(errors, "runtime ClinicalGraph node_id values must be non-empty and unique.")
    node_set = set(node_ids)
    for edge in edges:
        source = str(edge.get("source_node_id", "")).strip()
        target = str(edge.get("target_node_id", "")).strip()
        if source not in node_set or target not in node_set or not str(edge.get("relation_type", "")).strip():
            _append(errors, f"runtime ClinicalGraph edge has invalid endpoint/relation: {edge.get('edge_id')}")
    # The frozen sidecar is 14.58GB. Preserve every validation rule while
    # streaming records so Final Entry cannot recreate the eager-memory path.
    seen_image_ids: set[str] = set()
    sidecar_row_count = 0
    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sidecar_row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {sidecar_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL item at {sidecar_path}:{line_number} is not an object.")
            image_id = str(row.get("image_id", "")).strip()
            if image_id in seen_image_ids:
                _append(errors, f"runtime ClinicalGraph sidecar has duplicate image_id={image_id!r}.")
            seen_image_ids.add(image_id)
            values, masks, statuses = (
                row.get("concept_values"), row.get("observed_mask"), row.get("status"),
            )
            raw_sources = row.get("source_type")
            sources = raw_sources if isinstance(raw_sources, dict) else {
                node_id: str(raw_sources or "") for node_id in node_set
            }
            if not all(isinstance(item, dict) for item in (values, masks, statuses, sources)):
                _append(errors, f"runtime ClinicalGraph sidecar image_id={image_id} has invalid tensor-index mappings.")
                continue
            if any(set(item) != node_set for item in (values, masks, statuses, sources)):
                _append(errors, f"runtime ClinicalGraph sidecar image_id={image_id} is not aligned to canonical node ids.")
                continue
            for node_id in node_set:
                status = str(statuses[node_id]).strip()
                source = str(sources[node_id]).strip()
                mask = masks[node_id]
                if status not in {"present", "explicit_absent", "not_mentioned", "unknown", "not_applicable", "inferred_low_confidence", "inferred_high_confidence"}:
                    _append(errors, f"runtime ClinicalGraph sidecar has invalid status for image_id={image_id}, node={node_id}.")
                if mask not in {0, 1}:
                    _append(errors, f"runtime ClinicalGraph sidecar has invalid observed_mask for image_id={image_id}, node={node_id}.")
                if status == "inferred_high_confidence" and source != "model_inference":
                    _append(errors, f"runtime ClinicalGraph inferred node lacks model_inference source for image_id={image_id}, node={node_id}.")
    if sidecar_row_count != len(seen_image_ids) or seen_image_ids != image_ids:
        _append(errors, "runtime ClinicalGraph sidecar image_id set differs from V6 manifest.")
    return {"node_count": len(nodes), "edge_count": len(edges), "sidecar_row_count": sidecar_row_count}


def _validate_birads_priors(path: Path, image_ids: set[str], errors: list[str]) -> dict[str, Any]:
    rows, fields = _read_csv(path)
    if not {"image_id", "prior_path"}.issubset(fields):
        _append(errors, "stage1_birads_prior_manifest misses image_id/prior_path columns.")
        return {"row_count": len(rows)}
    by_image = {str(row.get("image_id", "")).strip(): row for row in rows}
    if set(by_image) != image_ids or len(by_image) != len(rows):
        _append(errors, "stage1_birads_prior_manifest image_id set differs from V6 manifest.")
    for image_id, row in by_image.items():
        prior_path = path.parent / str(row.get("prior_path", "")).strip()
        if not prior_path.is_file():
            _append(errors, f"BI-RADS prior file is missing for image_id={image_id}.")
            continue
        try:
            payload = json.loads(prior_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _append(errors, f"BI-RADS prior JSON is invalid for image_id={image_id}: {exc.msg}")
            continue
        if payload.get("schema_version") != "stage1_birads_prior_v1" or not isinstance(payload.get("concepts"), dict):
            _append(errors, f"BI-RADS prior schema is invalid for image_id={image_id}.")
    return {"row_count": len(rows)}


def validate_stage1_v6_bundle(bundle_dir: str | Path, resolved_config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    paths = {key: root / filename for key, filename in V6_STANDARD_FILENAMES.items()}
    for key, path in paths.items():
        # A valid report-derived graph can contain isolated canonical nodes;
        # rejection manifests are also intentionally empty when no case failed.
        allow_empty_manifest = key in {"graph_edges", "runtime_graph_edges", "final_rejected", "discarded_images"}
        if not path.exists() or (path.stat().st_size == 0 and not allow_empty_manifest):
            _append(errors, f"Missing or empty V6 asset {key}: {path}")
    if errors:
        return {"status": "BLOCKED_PENDING_ASSETS", "errors": errors, "warnings": warnings, "sections": {}}
    manifest, columns = _read_csv(paths["manifest"])
    missing_columns = sorted(_REQUIRED_COLUMNS - columns)
    if missing_columns:
        _append(errors, "V6 manifest misses columns: " + ", ".join(missing_columns))
    image_ids: set[str] = set()
    manifest_cases: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    manifest_by_image: dict[str, dict[str, str]] = {}
    for index, row in enumerate(manifest, start=1):
        missing = [key for key in _REQUIRED_COLUMNS if not str(row.get(key, "")).strip()]
        if missing:
            _append(errors, f"manifest row {index} misses required V6 values: {', '.join(missing)}")
            continue
        image_id = row["image_id"].strip()
        if image_id in image_ids:
            _append(errors, f"duplicate manifest image_id: {image_id}")
        image_ids.add(image_id)
        manifest_by_image[image_id] = row
        manifest_cases.add(row["case_id"].strip())
        manifest_hashes[image_id] = row["effective_report_sha256"].strip()
        if row["final_case_route"].strip() not in _ACCEPTED:
            _append(errors, f"manifest image_id={image_id} has non-accepted final route.")
        for field in ("train_eligible", "visual_training_enabled", "text_semantic_enabled", "clinical_graph_enabled", "concept_target_enabled", "semantic_soft_label_enabled"):
            if not _enabled(row, field):
                _append(errors, f"manifest image_id={image_id} has disabled required field {field}.")
        if not _enabled(row, "stage0_usable"):
            _append(errors, f"manifest image_id={image_id} must have stage0_usable=1 for formal training.")
        if not _enabled(row, "gaze_training_enabled"):
            _append(errors, f"manifest image_id={image_id} must have gaze_training_enabled=1 for formal training.")
        if row["patch_asset_mode"].strip() not in {"deterministic_runtime", "materialized"}:
            _append(errors, f"manifest image_id={image_id} has invalid patch_asset_mode.")
        for key in ("patch_geometry_key", "patch_token_order_version", "valid_content_mask_key"):
            _validate_runtime_key(str(row.get(key, "")), image_id=image_id, name=key, errors=errors)
        gaze_available = str(row.get("gaze_prior_available", "")).strip().lower()
        gaze_quality = str(row.get("gaze_prior_quality", "")).strip().lower()
        if gaze_available not in {"0", "1", "true", "false"}:
            _append(errors, f"manifest image_id={image_id} has invalid gaze_prior_available.")
        if not gaze_quality:
            _append(errors, f"manifest image_id={image_id} lacks gaze_prior_quality.")
        if _enabled(row, "gaze_training_enabled"):
            if gaze_available not in {"1", "true"}:
                _append(errors, f"manifest image_id={image_id} enables gaze training without an available gaze prior.")
            for key in ("gaze_prior_key", "gaze_to_patch_projection_key"):
                _validate_runtime_key(str(row.get(key, "")), image_id=image_id, name=key, errors=errors)
    reports = _read_jsonl(paths["accepted_reports"])
    accepted_by_case = {str(row.get("case_id", "")): row for row in reports}
    if len(accepted_by_case) != len(reports):
        _append(errors, "accepted_effective_report_manifest has duplicate case_id values.")
    for case_id in manifest_cases:
        report = accepted_by_case.get(case_id)
        if report is None:
            _append(errors, f"manifest case_id={case_id} lacks accepted Effective Report.")
            continue
        if str(report.get("final_case_route", "")) not in _ACCEPTED:
            _append(errors, f"accepted report case_id={case_id} has invalid terminal route.")
        if not str(report.get("effective_report_text", "")).strip() or not str(report.get("effective_report_sha256", "")).strip():
            _append(errors, f"accepted report case_id={case_id} lacks frozen text or hash.")
    prompts = _read_jsonl(paths["text_prompts"])
    prompt_by_image = {str(row.get("image_id", "")): row for row in prompts}
    if len(prompt_by_image) != len(prompts):
        _append(errors, "text_prompts has duplicate image_id values.")
    for image_id in image_ids:
        prompt = prompt_by_image.get(image_id)
        if prompt is None or not str(prompt.get("text_prompt", "")).strip():
            _append(errors, f"missing Effective Report prompt for image_id={image_id}.")
            continue
        if str(prompt.get("effective_report_sha256", "")) != manifest_hashes[image_id]:
            _append(errors, f"text prompt hash mismatch for image_id={image_id}.")
    try:
        embedding_payload = json.loads(paths["embeddings"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _append(errors, f"invalid embedding cache JSON: {exc.msg}")
        embedding_payload = {}
    embeddings = embedding_payload.get("prompt_embeddings", {})
    report_index = embedding_payload.get("v6_report_index", {})
    if not isinstance(embeddings, dict) or not isinstance(report_index, dict):
        _append(errors, "V6 embedding cache misses prompt_embeddings or v6_report_index.")
    else:
        dimensions: set[int] = set()
        for image_id, prompt in prompt_by_image.items():
            vector = embeddings.get(prompt.get("text_prompt"))
            if not isinstance(vector, list) or not vector:
                _append(errors, f"missing embedding for image_id={image_id}.")
                continue
            dimensions.add(len(vector))
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector):
                _append(errors, f"non-finite embedding for image_id={image_id}.")
            report = report_index.get(str(prompt.get("report_id", "")), {})
            if str(report.get("effective_report_sha256", "")) != str(prompt.get("effective_report_sha256", "")):
                _append(errors, f"embedding report-index hash mismatch for image_id={image_id}.")
        if len(dimensions) > 1:
            _append(errors, f"inconsistent embedding dimensions: {sorted(dimensions)}")
        expected_dimension: int | None = None
        if resolved_config_path is not None:
            try:
                config_payload = yaml.safe_load(Path(resolved_config_path).read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                _append(errors, f"invalid resolved V6 config YAML: {exc}")
                config_payload = {}
            raw_dimension = _config_value(config_payload, "model", "text_dim")
            if isinstance(raw_dimension, int) and raw_dimension > 0:
                expected_dimension = raw_dimension
            if expected_dimension is not None and dimensions and dimensions != {expected_dimension}:
                _append(errors, f"prompt embedding dimension {sorted(dimensions)} differs from config model.text_dim={expected_dimension}.")
    nodes = _read_jsonl(paths["graph_nodes"])
    edges = _read_jsonl(paths["graph_edges"])
    graph_cases = {str(node.get("case_id", "")) for node in nodes}
    if graph_cases != manifest_cases:
        _append(errors, "Clinical Graph case set differs from formal manifest case set.")
    valid_sources = {"official_fact", "report_explicit", "model_inference"}
    node_instance_ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_instance_id", "")).strip()
        source_type = str(node.get("source_type", "")).strip()
        confidence = node.get("confidence")
        if not node_id or node_id in node_instance_ids:
            _append(errors, f"Clinical Graph node_instance_id must be non-empty and unique: {node_id!r}")
        node_instance_ids.add(node_id)
        if source_type not in valid_sources or not isinstance(confidence, (int, float)):
            _append(errors, f"invalid graph node source/confidence: {node_id}")
        elif not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            _append(errors, f"graph node confidence must be finite in [0, 1]: {node_id}")
        supporting = node.get("supporting_node_ids")
        rationale = str(node.get("inference_rationale", "")).strip()
        if source_type == "model_inference":
            if not rationale:
                _append(errors, f"model-inference graph node lacks inference_rationale: {node_id}")
            if not isinstance(supporting, list) or not supporting:
                _append(errors, f"model-inference graph node lacks supporting_node_ids: {node_id}")
        case = str(node.get("case_id", ""))
        report = accepted_by_case.get(case, {})
        if str(node.get("effective_report_sha256", "")) != str(report.get("effective_report_sha256", "")):
            _append(errors, f"graph node hash mismatch for case_id={case}.")
    for edge in edges:
        edge_id = str(edge.get("relation_instance_id", "")).strip()
        source_type = str(edge.get("source_type", "")).strip()
        confidence = edge.get("confidence")
        if source_type not in valid_sources or not isinstance(confidence, (int, float)):
            _append(errors, f"invalid graph edge source/confidence: {edge_id}")
        elif not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            _append(errors, f"graph edge confidence must be finite in [0, 1]: {edge_id}")
        if not all(str(edge.get(field, "")).strip() for field in ("source_node_instance_id", "target_node_instance_id", "relation_type", "inference_rationale")):
            _append(errors, f"graph edge misses runtime relation fields: {edge_id}")
        if str(edge.get("source_node_instance_id", "")).strip() not in node_instance_ids or str(edge.get("target_node_instance_id", "")).strip() not in node_instance_ids:
            _append(errors, f"graph edge has an endpoint outside the graph node set: {edge_id}")
        supporting = edge.get("supporting_node_ids")
        if not isinstance(supporting, list) or not supporting:
            _append(errors, f"graph edge misses supporting_node_ids: {edge_id}")
        elif any(str(node_id).strip() not in node_instance_ids for node_id in supporting):
            _append(errors, f"graph edge has a supporting node outside the graph node set: {edge_id}")
    semantic_errors, semantic_warnings, semantic_summary = validate_semantic_soft_labels(
        manifest,
        None,
        paths["soft_labels_manifest"],
        semantic_soft_label_format="sparse_topk",
        semantic_soft_label_topk_path=paths["soft_labels"],
    )
    for message in semantic_errors:
        _append(errors, "V6 sparse semantic runtime contract: " + message)
    warnings.extend("V6 sparse semantic runtime contract: " + message for message in semantic_warnings)
    concept_summary = _validate_concept_targets(paths["concept_targets"], manifest_by_image, manifest_hashes, errors)
    runtime_graph_summary = _validate_runtime_graph(
        paths["runtime_graph_nodes"],
        paths["runtime_graph_edges"],
        paths["runtime_graph_sidecar"],
        image_ids,
        errors,
    )
    birads_summary = _validate_birads_priors(paths["birads_priors"], image_ids, errors)
    rejected_cases = {str(row.get("case_id", "")) for row in _read_jsonl(paths["final_rejected"])}
    discarded_images = {str(row.get("image_id", "")) for row in _read_jsonl(paths["discarded_images"])}
    stage0_nonusable_summary = _validate_stage0_nonusable_manifest(
        paths["stage0_nonusable"], image_ids, errors, warnings
    )
    if rejected_cases & manifest_cases:
        _append(errors, "final_rejected_case_ids overlap formal manifest cases.")
    if discarded_images & image_ids:
        _append(errors, "discarded_image_ids overlap formal manifest images.")
    if rejected_cases & graph_cases:
        _append(errors, "final_rejected_case_ids overlap Clinical Graph cases.")
    if resolved_config_path is not None:
        config_text = Path(resolved_config_path).read_text(encoding="utf-8")
        legacy_runtime_marker = "runtime" + "_structured_fallback"
        forbidden_enabled_value = "allow_structured_prompt_" + "fallback: true"
        if forbidden_enabled_value in config_text or legacy_runtime_marker in config_text:
            _append(errors, "Resolved V6 config enables a prohibited Structured Prompt fallback.")
    status = "PASS_FORMAL" if not errors else ("BLOCKED_FINAL_REJECT_CONTAMINATION" if any("overlap" in error for error in errors) else "BLOCKED_CONTRACT_VIOLATION")
    return {"status": status, "errors": errors, "warnings": warnings, "sections": {"manifest": {"image_count": len(image_ids), "case_count": len(manifest_cases)}, "hash_lineage": {"prompt_count": len(prompt_by_image), "graph_case_count": len(graph_cases)}, "semantic_soft_labels": semantic_summary, "concept_targets": concept_summary, "runtime_graph": runtime_graph_summary, "birads_priors": birads_summary, "final_reject_exclusion": {"final_rejected_case_count": len(rejected_cases), "discarded_image_count": len(discarded_images), "stage0_nonusable": stage0_nonusable_summary}}}
