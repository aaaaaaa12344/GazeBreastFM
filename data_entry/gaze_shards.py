"""Bounded immutable array shards used by the E2 gaze release.

The writer deliberately holds only one shard in memory.  Each manifest row
references a slice of the resulting NPY instead of an individual patch file or
an unbounded release-wide array.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from breast_pretrain.data_entry.release_common import atomic_write_bytes, sha256_bytes, sha256_file


STORAGE_VERSION = "gaze_weight_npy_shard_v1"


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


@dataclass
class _Pending:
    row: dict[str, Any]
    values: np.ndarray


class BoundedGazeShardWriter:
    """Write float32 vectors in bounded NPY shards and annotate source rows."""

    def __init__(self, *, release_root: Path, max_images_per_shard: int, max_shard_bytes: int,
                 start_shard_index: int = 0,
                 on_flush: Callable[[int, list[dict[str, Any]]], None] | None = None) -> None:
        if max_images_per_shard <= 0 or max_shard_bytes <= 0:
            raise ValueError("E2 shard limits must be positive.")
        self.release_root = release_root
        self.max_images_per_shard = int(max_images_per_shard)
        self.max_shard_bytes = int(max_shard_bytes)
        self._pending: list[_Pending] = []
        self._pending_bytes = 0
        if start_shard_index < 0:
            raise ValueError("E2 start_shard_index must be non-negative.")
        self._index = int(start_shard_index)
        self._on_flush = on_flush

    def append(self, *, row: dict[str, Any], values: np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("E2 gaze weights must be a non-empty finite float32 vector.")
        if self._pending and (
            len(self._pending) >= self.max_images_per_shard
            or self._pending_bytes + vector.nbytes > self.max_shard_bytes
        ):
            self.flush()
        self._pending.append(_Pending(row=row, values=vector.copy()))
        self._pending_bytes += int(vector.nbytes)

    def flush(self) -> None:
        if not self._pending:
            return
        payload = np.concatenate([item.values for item in self._pending])
        relative = f"assets/gaze_weight_shards/weights_{self._index:05d}.npy"
        path = self.release_root / relative
        atomic_write_bytes(path, _npy_bytes(payload))
        shard_sha256 = sha256_file(path)
        offset = 0
        for item in self._pending:
            length = int(item.values.size)
            item.row.update(
                {
                    "gaze_weight_shard_key": relative,
                    "shard_sha256": shard_sha256,
                    "offset": offset,
                    "length": length,
                    "dtype": "float32",
                    "shape": [length],
                    "gaze_weight_sha256": sha256_bytes(item.values.tobytes(order="C")),
                    "slice_sha256": sha256_bytes(item.values.tobytes(order="C")),
                    "storage_version": STORAGE_VERSION,
                }
            )
            offset += length
        flushed_rows = [item.row for item in self._pending]
        if self._on_flush is not None:
            self._on_flush(self._index, flushed_rows)
        self._pending.clear()
        self._pending_bytes = 0
        self._index += 1

    def close(self) -> None:
        self.flush()


class GazeShardReader:
    """Hash-check and mmap each E2 shard once per validation pass."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._arrays: dict[str, np.ndarray] = {}
        self._verified: set[str] = set()

    def read(self, row: dict[str, Any], *, expected_length: int) -> np.ndarray:
        key = str(row.get("gaze_weight_shard_key") or "")
        path = self.root / key
        if not key or not path.is_file():
            raise FileNotFoundError(f"Missing E2 gaze shard: {path}")
        if str(row.get("storage_version") or "") != STORAGE_VERSION:
            raise ValueError("Unsupported E2 gaze shard storage version.")
        if key not in self._verified:
            if sha256_file(path) != str(row.get("shard_sha256") or ""):
                raise ValueError(f"E2 gaze shard SHA256 mismatch: {path}")
            self._verified.add(key)
        if key not in self._arrays:
            self._arrays[key] = np.load(path, allow_pickle=False, mmap_mode="r")
        array = self._arrays[key]
        offset = int(row.get("offset", -1))
        length = int(row.get("length", -1))
        if offset < 0 or length != expected_length or offset + length > array.size:
            raise ValueError("E2 gaze shard offset/length is invalid.")
        values = np.asarray(array[offset : offset + length], dtype=np.float32)
        digest = sha256_bytes(values.tobytes(order="C"))
        if digest != str(row.get("slice_sha256") or "") or digest != str(row.get("gaze_weight_sha256") or ""):
            raise ValueError("E2 gaze slice SHA256 mismatch.")
        if not np.isfinite(values).all():
            raise ValueError("E2 gaze slice contains NaN or Inf.")
        return values


__all__ = ["BoundedGazeShardWriter", "GazeShardReader", "STORAGE_VERSION"]

