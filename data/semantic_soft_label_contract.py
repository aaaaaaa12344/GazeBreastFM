from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_semantic_soft_labels(
    manifest_rows: list[dict[str, str]],
    semantic_soft_label_path: Path | None,
    semantic_manifest_path: Path | None,
    semantic_soft_label_format: str = "dense",
    semantic_soft_label_topk_path: Path | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    label_format = _clean_text(semantic_soft_label_format).lower() or "dense"
    if label_format not in {"dense", "sparse_topk"}:
        errors.append(f"unsupported semantic soft-label format: {semantic_soft_label_format!r}")
        return errors, warnings, {"format": label_format}
    if semantic_soft_label_path is None or not semantic_soft_label_path.exists():
        if label_format == "dense":
            errors.append("stage1_semantic_soft_labels.npy is missing.")
            return errors, warnings, {"format": label_format, "matrix_exists": False, "manifest_exists": False}
    if semantic_manifest_path is None or not semantic_manifest_path.exists():
        errors.append("stage1_semantic_soft_label_manifest.csv is missing.")
        return errors, warnings, {"format": label_format, "matrix_exists": True, "manifest_exists": False}

    if label_format == "sparse_topk":
        return _validate_sparse_topk_semantic_soft_labels(
            manifest_rows,
            semantic_manifest_path=semantic_manifest_path,
            semantic_soft_label_topk_path=semantic_soft_label_topk_path,
        )
    return _validate_dense_semantic_soft_labels(
        manifest_rows,
        semantic_soft_label_path=semantic_soft_label_path,
        semantic_manifest_path=semantic_manifest_path,
        label_format=label_format,
    )


def _validate_dense_semantic_soft_labels(
    manifest_rows: list[dict[str, str]],
    semantic_soft_label_path: Path,
    semantic_manifest_path: Path,
    label_format: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    matrix = np.load(semantic_soft_label_path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        errors.append(
            "stage1_semantic_soft_labels.npy must be a square 2D matrix, got %s."
            % (tuple(matrix.shape),)
        )

    semantic_rows, fieldnames = _read_csv_rows(semantic_manifest_path)
    required_manifest_fields = {"row_index", "image_id"}
    if not required_manifest_fields.issubset(fieldnames):
        errors.append(
            "stage1_semantic_soft_label_manifest.csv is missing required columns: "
            + ", ".join(sorted(required_manifest_fields - set(fieldnames)))
        )

    semantic_image_ids: list[str] = []
    row_index_errors = 0
    for expected_index, row in enumerate(semantic_rows):
        semantic_image_ids.append(_clean_text(row.get("image_id") or row.get("sample_id")))
        if _clean_text(row.get("row_index")) != str(expected_index):
            row_index_errors += 1
    if row_index_errors:
        errors.append("stage1_semantic_soft_label_manifest.csv row_index values must be contiguous from 0.")
    if matrix.shape[0] != len(semantic_rows):
        errors.append(
            "semantic soft-label matrix size does not match semantic manifest row count: "
            f"{matrix.shape[0]} vs {len(semantic_rows)}."
        )

    manifest_image_ids = [
        _clean_text(row.get("image_id")) for row in manifest_rows if _clean_text(row.get("image_id"))
    ]
    if semantic_image_ids and set(semantic_image_ids) != set(manifest_image_ids):
        errors.append("semantic soft-label manifest image_id set does not match manifest_stage1_semantic.csv.")

    if matrix.size and not np.allclose(matrix, matrix.T):
        warnings.append("semantic soft-label matrix is not exactly symmetric.")
    if matrix.size and not np.allclose(np.diag(matrix), 1.0):
        warnings.append("semantic soft-label matrix diagonal is not all ones.")

    return errors, warnings, {
        "format": label_format,
        "matrix_exists": True,
        "manifest_exists": True,
        "matrix_shape": list(matrix.shape),
        "semantic_manifest_row_count": len(semantic_rows),
        "row_index_error_count": row_index_errors,
    }


def _validate_sparse_topk_semantic_soft_labels(
    manifest_rows: list[dict[str, str]],
    semantic_manifest_path: Path,
    semantic_soft_label_topk_path: Path | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    if semantic_soft_label_topk_path is None or not semantic_soft_label_topk_path.exists():
        errors.append("stage1_semantic_soft_labels_topk.npz is missing.")
        return errors, warnings, {"format": "sparse_topk", "topk_exists": False}

    jsonl_path = semantic_soft_label_topk_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        errors.append("stage1_semantic_soft_labels_topk.jsonl is missing.")

    semantic_rows, fieldnames = _read_csv_rows(semantic_manifest_path)
    # V6.1 formal runtime resolves sparse labels through the frozen
    # SemanticUnitSoftLabelAdapter.  Its immutable authority is 64940 semantic
    # units plus an image-to-unit mapping; it is not an image-level dense
    # soft-label matrix and must not be expanded for validation.
    if {"semantic_unit_index", "semantic_unit_id"}.issubset(fieldnames):
        mapping_path = semantic_manifest_path.parent / "stage1_image_to_semantic_unit.csv"
        if not mapping_path.is_file():
            errors.append("semantic-unit sparse runtime is missing stage1_image_to_semantic_unit.csv.")
            return errors, warnings, {"format": "sparse_topk", "semantic_unit_adapter": True}
        mapping_rows, mapping_fields = _read_csv_rows(mapping_path)
        required_mapping_fields = {"image_id", "semantic_unit_index", "semantic_unit_id"}
        if not required_mapping_fields.issubset(mapping_fields):
            errors.append(
                "semantic-unit mapping is missing required columns: "
                + ", ".join(sorted(required_mapping_fields - set(mapping_fields)))
            )
            return errors, warnings, {"format": "sparse_topk", "semantic_unit_adapter": True}
        unit_indices = [int(row["semantic_unit_index"]) for row in semantic_rows]
        if unit_indices != list(range(len(semantic_rows))):
            errors.append("semantic-unit manifest indices must be contiguous from 0.")
        unit_ids = {row["semantic_unit_id"] for row in semantic_rows}
        manifest_image_ids = {_clean_text(row.get("image_id")) for row in manifest_rows if _clean_text(row.get("image_id"))}
        mapping_by_image = {_clean_text(row.get("image_id")): row for row in mapping_rows}
        if set(mapping_by_image) != manifest_image_ids or len(mapping_by_image) != len(mapping_rows):
            errors.append("semantic-unit mapping image_id set does not match formal manifest.")
        for image_id, row in mapping_by_image.items():
            if _clean_text(row.get("semantic_unit_id")) not in unit_ids:
                errors.append(f"semantic-unit mapping image_id={image_id} has an unknown semantic_unit_id.")
                break
        payload = np.load(semantic_soft_label_topk_path, allow_pickle=False)
        required_arrays = {"indptr", "indices", "scores", "query_semantic_unit_index", "neighbor_semantic_unit_index", "neighbor_score"}
        missing_arrays = sorted(required_arrays.difference(payload.files))
        if missing_arrays:
            errors.append("semantic-unit sparse top-k is missing arrays: " + ", ".join(missing_arrays))
        elif len(payload["indptr"]) != len(semantic_rows) + 1 or int(payload["indptr"][0]) != 0 or int(payload["indptr"][-1]) != len(payload["indices"]):
            errors.append("semantic-unit sparse top-k indptr does not match frozen semantic-unit count.")
        elif len(payload["indices"]) != len(payload["scores"]) or np.any(~np.isfinite(payload["scores"])):
            errors.append("semantic-unit sparse top-k scores are invalid.")
        return errors, warnings, {
            "format": "sparse_topk",
            "semantic_unit_adapter": True,
            "semantic_unit_count": len(semantic_rows),
            "image_binding_count": len(mapping_rows),
            "edge_count": int(len(payload["indices"])) if "indices" in payload.files else 0,
        }
    required_manifest_fields = {"row_index", "image_id"}
    if not required_manifest_fields.issubset(fieldnames):
        errors.append(
            "stage1_semantic_soft_label_manifest.csv is missing required columns: "
            + ", ".join(sorted(required_manifest_fields - set(fieldnames)))
        )

    semantic_image_ids: list[str] = []
    row_index_errors = 0
    for expected_index, row in enumerate(semantic_rows):
        semantic_image_ids.append(_clean_text(row.get("image_id") or row.get("sample_id")))
        if _clean_text(row.get("row_index")) != str(expected_index):
            row_index_errors += 1
    if row_index_errors:
        errors.append("stage1_semantic_soft_label_manifest.csv row_index values must be contiguous from 0.")

    payload = np.load(semantic_soft_label_topk_path, allow_pickle=False)
    required_arrays = {"indptr", "indices", "scores", "row_indices", "image_ids"}
    missing_arrays = sorted(required_arrays.difference(payload.files))
    if missing_arrays:
        errors.append("stage1_semantic_soft_labels_topk.npz is missing arrays: " + ", ".join(missing_arrays))
        return errors, warnings, {"format": "sparse_topk", "topk_exists": True}

    indptr = payload["indptr"]
    indices = payload["indices"]
    scores = payload["scores"]
    row_indices = payload["row_indices"]
    npz_image_ids = [str(item) for item in payload["image_ids"].tolist()]
    if len(indptr) and int(indptr[0]) != 0:
        errors.append("sparse top-k indptr[0] must be 0.")
    if len(indptr) != len(semantic_rows) + 1:
        errors.append("sparse top-k indptr length must equal semantic manifest row count + 1.")
    elif int(indptr[-1]) != len(indices):
        errors.append("sparse top-k indptr[-1] must equal len(indices).")
    if len(indptr) > 1 and np.any(np.diff(indptr) < 0):
        errors.append("sparse top-k indptr must be monotonically non-decreasing.")
    if len(indices) != len(scores):
        errors.append("sparse top-k indices and scores lengths must match.")
    if np.any(~np.isfinite(indices)) or np.any(~np.isfinite(scores)):
        errors.append("sparse top-k indices and scores must not contain NaN or inf.")
    if len(scores) and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
        errors.append("sparse top-k scores must be within [0, 1].")
    if "top_k" in payload.files and len(indptr) == len(semantic_rows) + 1:
        top_k = int(np.asarray(payload["top_k"]).reshape(-1)[0])
        if top_k > 0 and np.any(np.diff(indptr) > top_k):
            errors.append("sparse top-k row edge count must not exceed top_k.")
    if list(row_indices) != list(range(len(semantic_rows))):
        errors.append("sparse top-k row_indices must be contiguous from 0.")
    if npz_image_ids != semantic_image_ids:
        errors.append("sparse top-k image_ids must align with semantic manifest row order.")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(semantic_rows)):
        errors.append("sparse top-k target row index is out of range.")

    manifest_image_ids = [
        _clean_text(row.get("image_id")) for row in manifest_rows if _clean_text(row.get("image_id"))
    ]
    if semantic_image_ids and set(semantic_image_ids) != set(manifest_image_ids):
        errors.append("semantic soft-label manifest image_id set does not match manifest_stage1_semantic.csv.")

    return errors, warnings, {
        "format": "sparse_topk",
        "topk_exists": True,
        "topk_jsonl_exists": jsonl_path.exists(),
        "manifest_exists": True,
        "semantic_manifest_row_count": len(semantic_rows),
        "edge_count": int(len(indices)),
        "row_index_error_count": row_index_errors,
    }
