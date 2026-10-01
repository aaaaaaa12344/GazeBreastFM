from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def resolve_prior_path(raw_value: Any, base_dir: str | Path) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir).expanduser().resolve() / path).resolve()
    return path


def save_prior_array(path: str | Path, array: np.ndarray) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(resolved_path, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return resolved_path


def load_prior_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path).expanduser().resolve(), allow_pickle=False)


def infer_single_channel_spatial_shape(array: np.ndarray) -> tuple[int, int]:
    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3 and array.shape[0] == 1:
        return int(array.shape[1]), int(array.shape[2])
    if array.ndim == 3 and array.shape[-1] == 1:
        return int(array.shape[0]), int(array.shape[1])
    raise ValueError(
        f"Unsupported single-channel .npy shape: {tuple(array.shape)}. "
        "Expected (H, W), (1, H, W), or (H, W, 1)."
    )


def inspect_prior_array(
    path: str | Path | None,
    *,
    expected_image_size: int | None = None,
) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "loaded": False,
            "shape": None,
            "spatial_shape": None,
            "shape_mismatch": False,
            "error": None,
        }

    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        return {
            "path": str(resolved_path),
            "exists": False,
            "loaded": False,
            "shape": None,
            "spatial_shape": None,
            "shape_mismatch": False,
            "error": None,
        }

    try:
        if resolved_path.suffix.lower() == ".npy":
            array = load_prior_array(resolved_path)
            shape = [int(item) for item in array.shape]
            spatial_shape = infer_single_channel_spatial_shape(np.asarray(array))
        else:
            with Image.open(resolved_path) as image:
                spatial_shape = (int(image.height), int(image.width))
            shape = [1, int(spatial_shape[0]), int(spatial_shape[1])]
    except Exception as exc:
        return {
            "path": str(resolved_path),
            "exists": True,
            "loaded": False,
            "shape": None,
            "spatial_shape": None,
            "shape_mismatch": False,
            "error": str(exc),
        }

    expected_shape = None
    if expected_image_size is not None:
        expected_shape = (int(expected_image_size), int(expected_image_size))
    return {
        "path": str(resolved_path),
        "exists": True,
        "loaded": True,
        "shape": shape,
        "spatial_shape": [int(item) for item in spatial_shape],
        "shape_mismatch": bool(expected_shape is not None and spatial_shape != expected_shape),
        "error": None,
    }


def save_consensus_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return resolved_path


def load_consensus_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def load_optional_json_dict(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        return {}
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {resolved_path}, got {type(payload).__name__}.")
    return payload
