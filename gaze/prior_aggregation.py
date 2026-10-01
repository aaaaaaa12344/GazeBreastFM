from __future__ import annotations

from typing import Any

import numpy as np

from breast_pretrain.gaze.attention_map import (
    make_high_confidence_mask,
    points_to_soft_attention_map,
    summarize_trajectory_quality,
)


def aggregate_soft_attention_maps(attention_maps: list[np.ndarray]) -> np.ndarray:
    if not attention_maps:
        raise ValueError("attention_maps must contain at least one array.")
    stacked = np.stack([np.asarray(item, dtype=np.float32) for item in attention_maps], axis=0)
    aggregated = stacked.mean(axis=0)
    max_value = float(aggregated.max()) if aggregated.size > 0 else 0.0
    if max_value > 0.0:
        aggregated = aggregated / max_value
    return aggregated.astype(np.float32, copy=False)


def aggregate_trajectory_points(
    trajectories_xy: list[np.ndarray],
    *,
    image_width: int,
    image_height: int,
    output_size: int,
    sigma: float,
    mask_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not trajectories_xy:
        raise ValueError("trajectories_xy must contain at least one trajectory.")
    attention_maps: list[np.ndarray] = []
    quality_rows: list[dict[str, Any]] = []
    for trajectory in trajectories_xy:
        attention_maps.append(
            points_to_soft_attention_map(
                trajectory,
                image_width=image_width,
                image_height=image_height,
                output_size=output_size,
                sigma=sigma,
            )
        )
        quality_rows.append(
            summarize_trajectory_quality(
                trajectory,
                image_width=image_width,
                image_height=image_height,
            )
        )
    attention_map = aggregate_soft_attention_maps(attention_maps)
    high_conf_mask = make_high_confidence_mask(attention_map, percentile=mask_percentile)
    inside_ratios = [float(item["inside_ratio"]) for item in quality_rows]
    return attention_map, high_conf_mask, {
        "prior_version": "aggregated",
        "num_trajectories": len(trajectories_xy),
        "inside_ratio_mean": float(np.mean(inside_ratios)) if inside_ratios else 0.0,
        "trajectory_quality_rows": quality_rows,
    }
