from __future__ import annotations

from pathlib import Path
from typing import Any

from breast_pretrain.data import load_joint_pretrain_manifest_rows
from breast_pretrain.gaze.prior_io import (
    inspect_prior_array,
    load_optional_json_dict,
    resolve_prior_path,
)


_FALLBACK_PRIOR_VERSIONS = {"no_gaze", "center_prior", "random_prior"}


def _normalize_prior_version(raw_value: Any) -> str:
    return str(raw_value or "").strip().lower()


def _resolve_attention_map_path(
    row: dict[str, Any],
    manifest_dir: Path,
    attention_map_dir: Path | None,
) -> Path | None:
    attention_path = resolve_prior_path(row.get("attention_map_path"), manifest_dir)
    if attention_path is None and attention_map_dir is not None:
        image_id = str(row.get("image_id") or "").strip()
        if image_id:
            return (attention_map_dir / f"{image_id}.png").resolve()
    return attention_path


def _resolve_summary_path(row: dict[str, Any], manifest_dir: Path) -> Path | None:
    summary_path = resolve_prior_path(row.get("trajectory_consensus_summary_path"), manifest_dir)
    if summary_path is not None:
        return summary_path
    sample_dir = resolve_prior_path(row.get("gaze_prior_dir"), manifest_dir)
    if sample_dir is not None:
        return (sample_dir / "trajectory_consensus_summary.json").resolve()
    return None


def _int_from_any(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            text = str(value).strip()
            if not text:
                continue
            return int(float(text))
        except (TypeError, ValueError):
            continue
    return None


def _has_any_hint(values: list[str], hints: tuple[str, ...]) -> bool:
    return any(hint in value for value in values for hint in hints)


def _infer_prior_version(
    row: dict[str, Any],
    summary_payload: dict[str, Any],
    trajectory_payload: dict[str, Any],
    *,
    attention_loaded: bool,
) -> str:
    explicit_versions = [
        _normalize_prior_version(row.get("prior_version")),
        _normalize_prior_version(summary_payload.get("prior_version")),
        _normalize_prior_version(trajectory_payload.get("prior_version")),
    ]
    for version in explicit_versions:
        if version and version not in {"unknown", "missing"}:
            return version

    type_hints = [
        _normalize_prior_version(row.get("prior_type")),
        _normalize_prior_version(row.get("prior_source")),
        _normalize_prior_version(summary_payload.get("prior_type")),
        _normalize_prior_version(summary_payload.get("prior_source")),
        _normalize_prior_version(trajectory_payload.get("prior_type")),
        _normalize_prior_version(trajectory_payload.get("prior_source")),
    ]
    for version in _FALLBACK_PRIOR_VERSIONS:
        if version in type_hints:
            return version

    path_hints = [
        _normalize_prior_version(row.get("summary_json_path")),
        _normalize_prior_version(row.get("trajectory_qc_path")),
        _normalize_prior_version(row.get("attention_map_path")),
        _normalize_prior_version(summary_payload.get("summary_json_path")),
        _normalize_prior_version(trajectory_payload.get("summary_json_path")),
    ]
    sample_count = _int_from_any(
        row.get("num_samples_found"),
        summary_payload.get("num_samples_found"),
        trajectory_payload.get("num_samples_found"),
    )
    aggregated_hints = ("aggregated", "consensus", "_agg", "attention_map_agg", "high_conf_mask_agg")
    if (sample_count is not None and sample_count > 1) or _has_any_hint(
        type_hints + path_hints,
        aggregated_hints,
    ):
        return "aggregated"

    single_hints = ("single_trajectory", "diffeye", "trajectory_qc")
    if (sample_count is not None and sample_count == 1) or _has_any_hint(
        type_hints + path_hints,
        single_hints,
    ):
        return "single_trajectory"

    if attention_loaded:
        return "existing_attention_map_unknown_version"
    return "missing"


def audit_joint_pretrain_gaze_priors(
    manifest_path: str | Path,
    *,
    attention_map_dir: str | Path | None = None,
    max_samples: int | None = None,
    expected_image_size: int | None = None,
) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_dir = resolved_manifest_path.parent
    rows = load_joint_pretrain_manifest_rows(resolved_manifest_path)
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    resolved_attention_map_dir = (
        Path(attention_map_dir).expanduser().resolve()
        if attention_map_dir is not None
        else None
    )

    missing_prior_count = 0
    aggregated_count = 0
    single_trajectory_count = 0
    fallback_count = 0
    existing_attention_map_count = 0
    existing_high_conf_mask_count = 0
    missing_attention_file_count = 0
    missing_high_conf_mask_file_count = 0
    prior_version_unknown_count = 0
    loaded_attention_map_shape_mismatch_count = 0
    loaded_high_conf_mask_shape_mismatch_count = 0
    sample_summaries: list[dict[str, Any]] = []

    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        attention_path = _resolve_attention_map_path(row, manifest_dir, resolved_attention_map_dir)
        high_conf_mask_path = resolve_prior_path(row.get("high_conf_mask_path"), manifest_dir)
        summary_path = _resolve_summary_path(row, manifest_dir)
        trajectory_qc_path = resolve_prior_path(row.get("trajectory_qc_path"), manifest_dir)

        attention_info = inspect_prior_array(
            attention_path,
            expected_image_size=expected_image_size,
        )
        high_conf_info = inspect_prior_array(
            high_conf_mask_path,
            expected_image_size=expected_image_size,
        )

        if attention_info["exists"]:
            existing_attention_map_count += 1
        else:
            missing_attention_file_count += 1
        if high_conf_info["exists"]:
            existing_high_conf_mask_count += 1
        else:
            missing_high_conf_mask_file_count += 1
        if attention_info["shape_mismatch"]:
            loaded_attention_map_shape_mismatch_count += 1
        if high_conf_info["shape_mismatch"]:
            loaded_high_conf_mask_shape_mismatch_count += 1

        summary_payload = load_optional_json_dict(summary_path)
        trajectory_payload = load_optional_json_dict(trajectory_qc_path)
        prior_version = _infer_prior_version(
            row,
            summary_payload,
            trajectory_payload,
            attention_loaded=bool(attention_info["loaded"]),
        )

        if prior_version == "missing":
            missing_prior_count += 1
        elif prior_version == "aggregated":
            aggregated_count += 1
        elif prior_version == "single_trajectory":
            single_trajectory_count += 1
        elif prior_version in _FALLBACK_PRIOR_VERSIONS:
            fallback_count += 1
        elif prior_version == "existing_attention_map_unknown_version":
            prior_version_unknown_count += 1

        sample_summaries.append(
            {
                "image_id": image_id,
                "prior_version": prior_version,
                "attention_map_path": attention_info["path"],
                "high_conf_mask_path": high_conf_info["path"],
                "attention_map_exists": attention_info["exists"],
                "high_conf_mask_exists": high_conf_info["exists"],
                "attention_map_loaded": attention_info["loaded"],
                "high_conf_mask_loaded": high_conf_info["loaded"],
                "attention_map_shape": attention_info["spatial_shape"],
                "high_conf_mask_shape": high_conf_info["spatial_shape"],
                "attention_map_shape_mismatch": attention_info["shape_mismatch"],
                "high_conf_mask_shape_mismatch": high_conf_info["shape_mismatch"],
                "attention_map_error": attention_info["error"],
                "high_conf_mask_error": high_conf_info["error"],
            }
        )

    has_warnings = any(
        (
            missing_prior_count,
            fallback_count,
            prior_version_unknown_count,
            missing_high_conf_mask_file_count,
            loaded_attention_map_shape_mismatch_count,
            loaded_high_conf_mask_shape_mismatch_count,
        )
    )
    return {
        "status": "pass_with_warnings" if has_warnings else "pass",
        "sample_count": len(rows),
        "missing_prior_count": missing_prior_count,
        "aggregated_count": aggregated_count,
        "single_trajectory_count": single_trajectory_count,
        "fallback_count": fallback_count,
        "existing_attention_map_count": existing_attention_map_count,
        "existing_high_conf_mask_count": existing_high_conf_mask_count,
        "missing_attention_file_count": missing_attention_file_count,
        "missing_high_conf_mask_file_count": missing_high_conf_mask_file_count,
        "prior_version_unknown_count": prior_version_unknown_count,
        "loaded_attention_map_shape_mismatch_count": loaded_attention_map_shape_mismatch_count,
        "loaded_high_conf_mask_shape_mismatch_count": loaded_high_conf_mask_shape_mismatch_count,
        "samples": sample_summaries[:50],
    }
