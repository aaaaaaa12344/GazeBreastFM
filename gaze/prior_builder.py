from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from breast_pretrain.gaze.prior_aggregation import aggregate_trajectory_points
from breast_pretrain.gaze.prior_io import save_consensus_summary, save_prior_array
from breast_pretrain.gaze.patch_weights import build_patch_gaze_weight_array


def _resolve_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_existing_map(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return array.astype(np.float32, copy=False)
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3:
        array = np.squeeze(array, axis=0)
    return np.asarray(array, dtype=np.float32)


def _build_center_prior(image_size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, num=image_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    squared_distance = grid_x**2 + grid_y**2
    prior = np.exp(-squared_distance / 0.25).astype(np.float32)
    prior /= max(float(prior.max()), 1e-6)
    return prior


def build_gaze_priors_from_manifest(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    image_size: int,
    patch_size: int,
    fallback_mode: str = "center_prior",
) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    resolved_output_root = Path(output_root).expanduser().resolve()
    rows = _resolve_manifest_rows(resolved_manifest_path)
    built_count = 0
    fallback_count = 0
    single_trajectory_count = 0
    aggregated_count = 0
    samples: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            continue
        sample_dir = resolved_output_root / image_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        attention_map_path = str(row.get("attention_map_path") or "").strip()
        high_conf_mask_path = str(row.get("high_conf_mask_path") or "").strip()
        trajectory_paths_raw = str(row.get("trajectory_paths_json") or "").strip()

        if trajectory_paths_raw:
            trajectory_paths = json.loads(trajectory_paths_raw)
            trajectories = [
                np.asarray(np.load(Path(item).expanduser().resolve(), allow_pickle=False), dtype=np.float32)
                for item in trajectory_paths
            ]
            attention_map, high_conf_mask, consensus_summary = aggregate_trajectory_points(
                trajectories,
                image_width=image_size,
                image_height=image_size,
                output_size=image_size,
                sigma=5.0,
                mask_percentile=90.0,
            )
            aggregated_count += 1
        elif attention_map_path:
            resolved_attention_path = Path(attention_map_path)
            if not resolved_attention_path.is_absolute():
                resolved_attention_path = (resolved_manifest_path.parent / resolved_attention_path).resolve()
            attention_map = _load_existing_map(resolved_attention_path)
            if high_conf_mask_path:
                resolved_mask_path = Path(high_conf_mask_path)
                if not resolved_mask_path.is_absolute():
                    resolved_mask_path = (resolved_manifest_path.parent / resolved_mask_path).resolve()
                high_conf_mask = _load_existing_map(resolved_mask_path)
            else:
                high_conf_mask = (attention_map >= np.percentile(attention_map[attention_map > 0], 90)).astype(np.float32) if np.any(attention_map > 0) else np.zeros_like(attention_map, dtype=np.float32)
            consensus_summary = {
                "prior_version": "single_trajectory",
                "num_trajectories": 1,
                "inside_ratio_mean": 1.0,
            }
            single_trajectory_count += 1
        else:
            if fallback_mode == "random_prior":
                rng = np.random.default_rng(42)
                attention_map = rng.random((image_size, image_size), dtype=np.float32)
                attention_map /= max(float(attention_map.max()), 1e-6)
            elif fallback_mode == "no_gaze":
                attention_map = np.zeros((image_size, image_size), dtype=np.float32)
            else:
                attention_map = _build_center_prior(image_size)
            high_conf_mask = (attention_map >= np.percentile(attention_map, 90)).astype(np.float32)
            consensus_summary = {
                "prior_version": fallback_mode,
                "num_trajectories": 0,
                "inside_ratio_mean": 0.0,
            }
            fallback_count += 1

        patch_gaze_weight = build_patch_gaze_weight_array(
            attention_map=attention_map,
            high_conf_mask=high_conf_mask,
            patch_size=patch_size,
        )
        save_prior_array(sample_dir / "attention_map_agg.npy", attention_map)
        save_prior_array(sample_dir / "high_conf_mask_agg.npy", high_conf_mask)
        save_prior_array(sample_dir / "patch_gaze_weight.npy", patch_gaze_weight)
        save_consensus_summary(sample_dir / "trajectory_consensus_summary.json", consensus_summary)
        samples.append(
            {
                "image_id": image_id,
                "output_dir": str(sample_dir),
                "prior_version": consensus_summary["prior_version"],
            }
        )
        built_count += 1

    return {
        "manifest_path": str(resolved_manifest_path),
        "output_root": str(resolved_output_root),
        "sample_count": built_count,
        "aggregated_count": aggregated_count,
        "single_trajectory_count": single_trajectory_count,
        "fallback_count": fallback_count,
        "samples": samples,
    }
