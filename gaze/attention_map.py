from __future__ import annotations

from typing import Any

import numpy as np


def _as_points_array(points_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must be an array-like value with shape [N, 2].")
    return points


def _sanitize_points(
    points_xy: Any,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = _as_points_array(points_xy)
    num_points = int(points.shape[0])
    if num_points == 0:
        return points, {
            "num_points": 0,
            "num_finite_points": 0,
            "num_inside_points": 0,
            "inside_ratio": 0.0,
            "finite_ratio": 0.0,
            "any_oob_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "step_mean_px": 0.0,
            "step_p95_px": 0.0,
            "step_max_px": 0.0,
            "jump_100_ratio": 0.0,
            "has_valid_points": False,
        }

    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]
    inside_mask = (
        (finite_points[:, 0] >= 0.0)
        & (finite_points[:, 0] < float(image_width))
        & (finite_points[:, 1] >= 0.0)
        & (finite_points[:, 1] < float(image_height))
    )
    inside_points = finite_points[inside_mask]
    num_finite_points = int(finite_points.shape[0])
    num_inside_points = int(inside_points.shape[0])
    num_oob_points = num_finite_points - num_inside_points

    if num_inside_points >= 2:
        deltas = np.diff(inside_points, axis=0)
        step_lengths = np.linalg.norm(deltas, axis=1)
        step_mean_px = float(np.mean(step_lengths))
        step_p95_px = float(np.percentile(step_lengths, 95))
        step_max_px = float(np.max(step_lengths))
        jump_100_ratio = float(np.mean(step_lengths > 100.0))
    else:
        step_mean_px = 0.0
        step_p95_px = 0.0
        step_max_px = 0.0
        jump_100_ratio = 0.0

    if num_inside_points == 0:
        bbox_area_ratio = 0.0
    else:
        x_min = float(np.min(inside_points[:, 0]))
        x_max = float(np.max(inside_points[:, 0]))
        y_min = float(np.min(inside_points[:, 1]))
        y_max = float(np.max(inside_points[:, 1]))
        bbox_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        full_area = max(1.0, float(image_width * image_height))
        bbox_area_ratio = float(bbox_area / full_area)

    summary = {
        "num_points": num_points,
        "num_finite_points": num_finite_points,
        "num_inside_points": num_inside_points,
        "inside_ratio": float(num_inside_points / num_points),
        "finite_ratio": float(num_finite_points / num_points),
        "any_oob_ratio": float(num_oob_points / num_points),
        "bbox_area_ratio": bbox_area_ratio,
        "step_mean_px": step_mean_px,
        "step_p95_px": step_p95_px,
        "step_max_px": step_max_px,
        "jump_100_ratio": jump_100_ratio,
        "has_valid_points": bool(num_inside_points > 0),
    }
    return inside_points, summary


def summarize_trajectory_quality(
    points_xy: Any,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    _, summary = _sanitize_points(points_xy, image_width=image_width, image_height=image_height)
    return summary


def points_to_soft_attention_map(
    points_xy: Any,
    image_width: int,
    image_height: int,
    output_size: int = 224,
    sigma: float = 5.0,
) -> np.ndarray:
    inside_points, _ = _sanitize_points(
        points_xy,
        image_width=image_width,
        image_height=image_height,
    )
    if inside_points.shape[0] == 0:
        return np.zeros((output_size, output_size), dtype=np.float32)

    sigma = max(float(sigma), 1e-6)
    width_scale = float(output_size) / max(float(image_width), 1.0)
    height_scale = float(output_size) / max(float(image_height), 1.0)

    xs = np.arange(output_size, dtype=np.float32)
    ys = np.arange(output_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    attention = np.zeros((output_size, output_size), dtype=np.float32)
    for x_coord, y_coord in inside_points:
        x_scaled = x_coord * width_scale
        y_scaled = y_coord * height_scale
        squared_distance = (grid_x - x_scaled) ** 2 + (grid_y - y_scaled) ** 2
        attention += np.exp(-squared_distance / (2.0 * sigma * sigma)).astype(np.float32)

    max_value = float(attention.max())
    if max_value > 0.0:
        attention /= max_value
    return attention.astype(np.float32)


def make_high_confidence_mask(
    soft_map: Any,
    percentile: float = 90,
) -> np.ndarray:
    array = np.asarray(soft_map, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("soft_map must be a 2D array.")

    max_value = float(array.max()) if array.size > 0 else 0.0
    if max_value <= 0.0:
        return np.zeros_like(array, dtype=np.float32)

    positive_values = array[array > 0.0]
    if positive_values.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    percentile = float(np.clip(percentile, 0.0, 100.0))
    threshold = float(np.percentile(positive_values, percentile))
    if threshold <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array >= threshold) & (array > 0.0)).astype(np.float32)
