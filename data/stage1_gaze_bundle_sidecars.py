from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from breast_pretrain.data.bucketed_stage1_dataloader import normalize_stage1_modality
from breast_pretrain.text.prompt_embedding import _deterministic_prompt_vector


MANIFEST_NAME = "manifest_stage1_semantic.csv"
TEXT_PROMPTS_NAME = "text_prompts.jsonl"
PROMPT_EMBEDDINGS_NAME = "stage1_prompt_embeddings.json"
SEMANTIC_MANIFEST_NAME = "stage1_semantic_soft_label_manifest.csv"
SEMANTIC_TOPK_NPZ_NAME = "stage1_semantic_soft_labels_topk.npz"
SEMANTIC_TOPK_JSONL_NAME = "stage1_semantic_soft_labels_topk.jsonl"
BIRADS_PRIOR_MANIFEST_NAME = "stage1_birads_prior_manifest.csv"
CASE_CONCEPT_VECTOR_NAME = "tri_modal_case_concept_vector.jsonl"
TEXT_DIM = 768

BUILDER_MANAGED_FILES = (
    MANIFEST_NAME,
    "manifest_excluded_diagnostic_only.csv",
    "gaze_enabled_bundle_build_summary.json",
    "gaze_enabled_bundle_linkage_audit.csv",
    TEXT_PROMPTS_NAME,
    PROMPT_EMBEDDINGS_NAME,
    SEMANTIC_MANIFEST_NAME,
    SEMANTIC_TOPK_NPZ_NAME,
    SEMANTIC_TOPK_JSONL_NAME,
    "semantic_soft_label_summary.json",
    "semantic_soft_label_audit_examples.jsonl",
    BIRADS_PRIOR_MANIFEST_NAME,
    CASE_CONCEPT_VECTOR_NAME,
)
BUILDER_MANAGED_DIRS = ("stage1_birads_priors",)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _key(image_id: Any, modality: Any) -> tuple[str, str]:
    return _clean(image_id), normalize_stage1_modality(modality)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            rows.append(payload)
    return rows


def _portable_path(path: Path, output_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return os.path.relpath(resolved, output_root.expanduser().resolve()).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _resolve_existing(raw_value: str, base_root: Path, field_name: str, image_id: str) -> Path:
    value = _clean(raw_value)
    if not value:
        raise ValueError(f"Prior row {image_id} has empty {field_name}.")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [base_root / path, _project_root() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing {field_name} for {image_id}: {value}. Checked: {checked}")


def clean_builder_managed_outputs(output_root: Path) -> None:
    root = output_root.expanduser().resolve()
    for relative in BUILDER_MANAGED_FILES:
        path = root / relative
        if path.is_file():
            path.unlink()
    for relative in BUILDER_MANAGED_DIRS:
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path)


def _prompt_for_row(row: dict[str, str]) -> str:
    return (
        "Structured tri-modal Stage 1 audited gaze prompt: "
        f"modality={row.get('modality', '')}; "
        f"source_dataset={row.get('source_dataset', '')}; "
        f"view={row.get('view', 'unknown')}; "
        f"laterality={row.get('laterality', 'unknown')}; "
        f"finding={row.get('finding', 'unknown')}; "
        f"birads={row.get('birads', 'unknown')}; "
        f"benign_malignant_label={row.get('benign_malignant_label', 'unknown')}."
    )


def filter_text_prompts_and_embeddings(
    *,
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    allow_fixture_semantic_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prompt_path = source_root / TEXT_PROMPTS_NAME
    missing_prompt_count = 0
    generic_fallback_prompt_count = 0
    if prompt_path.is_file():
        source_prompts = _read_jsonl(prompt_path)
    elif allow_fixture_semantic_fallback:
        source_prompts = []
    else:
        raise FileNotFoundError(f"Missing required source text prompt sidecar: {prompt_path}")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in source_prompts:
        key = _key(item.get("image_id"), item.get("modality"))
        if not key[0]:
            continue
        if key in by_key:
            raise ValueError(f"Duplicate source prompt key: {key}")
        by_key[key] = item

    prompts: list[dict[str, Any]] = []
    for row in rows:
        key = _key(row.get("image_id"), row.get("modality"))
        item = dict(by_key.get(key) or {})
        if not item:
            missing_prompt_count += 1
            if not allow_fixture_semantic_fallback:
                continue
            generic_fallback_prompt_count += 1
            item = {
                "image_id": row["image_id"],
                "modality": row["modality"],
                "source_dataset": row.get("source_dataset", ""),
                "prompt_type": "fixture_semantic_fallback",
                "text_prompt": _prompt_for_row(row),
            }
        item["image_id"] = row["image_id"]
        item["modality"] = row["modality"]
        prompts.append(item)
    if missing_prompt_count and not allow_fixture_semantic_fallback:
        raise ValueError(f"Missing source text prompts for selected strict rows: {missing_prompt_count}")
    write_jsonl(output_root / TEXT_PROMPTS_NAME, prompts)

    embedding_stats = filter_prompt_embeddings(
        source_root=source_root,
        output_root=output_root,
        prompts=prompts,
        allow_fixture_semantic_fallback=allow_fixture_semantic_fallback,
    )
    return prompts, {
        "missing_prompt_count": missing_prompt_count,
        "generic_fallback_prompt_count": generic_fallback_prompt_count,
        **embedding_stats,
    }


def filter_prompt_embeddings(
    *,
    source_root: Path,
    output_root: Path,
    prompts: list[dict[str, Any]],
    allow_fixture_semantic_fallback: bool,
) -> dict[str, int]:
    source_path = source_root / PROMPT_EMBEDDINGS_NAME
    used_prompts = [_clean(item.get("text_prompt")) for item in prompts if _clean(item.get("text_prompt"))]
    payload: dict[str, Any] = {"text_dim": TEXT_DIM, "prompt_embeddings": {}}
    if source_path.is_file():
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source_payload, dict):
            raise ValueError(f"Prompt embedding sidecar must be a JSON object: {source_path}")
        payload["text_dim"] = int(source_payload.get("text_dim") or TEXT_DIM)
        source_embeddings = source_payload.get("prompt_embeddings", source_payload)
        if not isinstance(source_embeddings, dict):
            raise ValueError(f"prompt_embeddings must be a JSON object: {source_path}")
        payload["prompt_embeddings"] = {
            prompt: source_embeddings[prompt]
            for prompt in used_prompts
            if prompt in source_embeddings
        }
    elif not allow_fixture_semantic_fallback:
        raise FileNotFoundError(f"Missing required prompt embedding sidecar: {source_path}")

    text_dim = int(payload["text_dim"])
    embeddings = payload["prompt_embeddings"]
    missing_prompt_embedding_count = 0
    deterministic_fallback_embedding_count = 0
    for prompt in used_prompts:
        if prompt in embeddings:
            continue
        missing_prompt_embedding_count += 1
        if allow_fixture_semantic_fallback:
            deterministic_fallback_embedding_count += 1
            embeddings[prompt] = _deterministic_prompt_vector(prompt, text_dim=text_dim).tolist()
    if missing_prompt_embedding_count and not allow_fixture_semantic_fallback:
        raise ValueError(f"Missing source prompt embeddings for selected strict rows: {missing_prompt_embedding_count}")
    write_json(output_root / PROMPT_EMBEDDINGS_NAME, payload)
    return {
        "missing_prompt_embedding_count": missing_prompt_embedding_count,
        "deterministic_fallback_embedding_count": deterministic_fallback_embedding_count,
    }


def filter_case_concept_vectors(
    *,
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    allow_fixture_semantic_fallback: bool,
) -> dict[str, int]:
    source_path = source_root / CASE_CONCEPT_VECTOR_NAME
    if not source_path.is_file():
        if allow_fixture_semantic_fallback:
            return _write_fixture_case_concept_vectors(output_root, rows)
        raise FileNotFoundError(f"Missing required source clinical graph concept sidecar: {source_path}")

    source_rows = _read_jsonl(source_path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in source_rows:
        key = _key(item.get("image_id"), item.get("modality"))
        if not key[0]:
            continue
        if key in by_key:
            raise ValueError(f"Duplicate source concept vector key: {key}")
        by_key[key] = item

    output_rows: list[dict[str, Any]] = []
    missing = 0
    for index, row in enumerate(rows):
        key = _key(row.get("image_id"), row.get("modality"))
        item = by_key.get(key)
        if item is None:
            missing += 1
            continue
        copied = dict(item)
        copied["row_index"] = index
        output_rows.append(copied)
    if missing:
        raise ValueError(f"Source concept vector sidecar is missing selected strict rows: {missing}")
    write_jsonl(output_root / CASE_CONCEPT_VECTOR_NAME, output_rows)
    return {
        "case_concept_vector_rows": len(output_rows),
        "case_concept_vector_missing_count": 0,
        "case_concept_vector_fixture_fallback_count": 0,
    }


def _write_fixture_case_concept_vectors(output_root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    concept_rows = []
    for index, row in enumerate(rows):
        concept_rows.append(
            {
                "row_index": index,
                "image_id": row["image_id"],
                "modality": row.get("modality", ""),
                "concept_values": {"modality": row.get("modality", "unknown")},
                "observed_mask": {"modality": 1},
                "raw_unconfirmed_enabled": False,
            }
        )
    write_jsonl(output_root / CASE_CONCEPT_VECTOR_NAME, concept_rows)
    return {
        "case_concept_vector_rows": len(concept_rows),
        "case_concept_vector_missing_count": 0,
        "case_concept_vector_fixture_fallback_count": len(concept_rows),
    }


def filter_birads_prior_manifest_and_files(
    *,
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, str]],
    allow_fixture_semantic_fallback: bool,
) -> dict[str, int]:
    source_path = source_root / BIRADS_PRIOR_MANIFEST_NAME
    if not source_path.is_file():
        if allow_fixture_semantic_fallback:
            return _write_fixture_prior_manifest(output_root, rows)
        raise FileNotFoundError(f"Missing required source BI-RADS prior manifest: {source_path}")

    import csv

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    if "image_id" not in fieldnames or "prior_path" not in fieldnames:
        raise ValueError(f"{source_path} must contain image_id and prior_path columns.")
    has_modality = "modality" in fieldnames
    by_key: dict[tuple[str, str] | tuple[str], dict[str, str]] = {}
    for item in source_rows:
        key = _key(item.get("image_id"), item.get("modality")) if has_modality else (_clean(item.get("image_id")),)
        if not key[0]:
            continue
        if key in by_key:
            raise ValueError(f"Duplicate BI-RADS prior manifest key: {key}")
        by_key[key] = item

    selected_keys = [_key(row.get("image_id"), row.get("modality")) for row in rows]
    if not has_modality:
        image_counts: dict[str, int] = {}
        for image_id, _modality in selected_keys:
            image_counts[image_id] = image_counts.get(image_id, 0) + 1
        ambiguous = [image_id for image_id, count in image_counts.items() if count > 1]
        if ambiguous:
            raise ValueError("BI-RADS prior manifest lacks modality and selected image_id is duplicated: " + ", ".join(ambiguous))

    output_rows: list[dict[str, str]] = []
    missing = 0
    for row in rows:
        key = _key(row.get("image_id"), row.get("modality")) if has_modality else (_clean(row.get("image_id")),)
        source_item = by_key.get(key)
        if source_item is None:
            missing += 1
            continue
        copied = dict(source_item)
        source_prior = _resolve_existing(copied["prior_path"], source_root, "prior_path", row["image_id"])
        target_prior = output_root / copied["prior_path"]
        target_prior.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_prior, target_prior)
        output_rows.append(copied)
    if missing:
        raise ValueError(f"Source BI-RADS prior manifest is missing selected strict rows: {missing}")
    write_csv(output_root / BIRADS_PRIOR_MANIFEST_NAME, fieldnames, output_rows)
    return {
        "birads_prior_manifest_rows": len(output_rows),
        "birads_prior_missing_count": 0,
        "birads_prior_fixture_fallback_count": 0,
    }


def _write_fixture_prior_manifest(output_root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    prior_rows: list[dict[str, Any]] = []
    prior_dir = output_root / "stage1_birads_priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_id = row["image_id"]
        stem = "".join(ch if ch.isalnum() else "_" for ch in image_id).strip("_") or "image"
        prior_path = f"stage1_birads_priors/{stem}.json"
        prior_rows.append(
            {
                "image_id": image_id,
                "modality": row.get("modality", ""),
                "prior_path": prior_path,
                "birads_observed_mask": row.get("birads_observed_mask", "0"),
            }
        )
        write_json(
            output_root / prior_path,
            {
                "image_id": image_id,
                "modality": row.get("modality", ""),
                "observed_masks": {"birads": int(row.get("birads_observed_mask") or 0)},
            },
        )
    write_csv(output_root / BIRADS_PRIOR_MANIFEST_NAME, ["image_id", "modality", "prior_path", "birads_observed_mask"], prior_rows)
    return {
        "birads_prior_manifest_rows": len(prior_rows),
        "birads_prior_missing_count": 0,
        "birads_prior_fixture_fallback_count": len(prior_rows),
    }
