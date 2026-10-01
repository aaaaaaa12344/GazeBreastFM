"""Bounded, compact Patch-mask shard writer and reader for E1 releases."""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from breast_pretrain.data_entry.release_common import atomic_write_bytes, sha256_file


MASK_SHARD_STORAGE_VERSION = "patch_mask_packbits_npy_shard_v2"
MASK_BIT_ORDER = "little"
MASK_KINDS = ("non_padding", "foreground", "valid")


def _npy_bytes(array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(payload, array, version=(1, 0), allow_pickle=False)
    return payload.getvalue()


@dataclass
class _PendingImage:
    row: dict[str, Any]
    masks: dict[str, np.ndarray]


class BoundedMaskShardWriter:
    """Collect only one bounded shard, never the full release's mask arrays."""

    def __init__(
        self,
        *,
        release_root: Path,
        images_per_shard: int,
        max_shard_bytes: int,
        start_shard_index: int = 0,
        on_flush: Callable[[int, list[dict[str, Any]]], None] | None = None,
    ) -> None:
        if images_per_shard <= 0 or max_shard_bytes <= 0:
            raise ValueError("images_per_shard and max_shard_bytes must be positive.")
        self.release_root = release_root
        self.images_per_shard = int(images_per_shard)
        self.max_shard_bytes = int(max_shard_bytes)
        if start_shard_index < 0:
            raise ValueError("start_shard_index must be non-negative.")
        self.pending: list[_PendingImage] = []
        self.pending_bytes = 0
        self.shard_index = int(start_shard_index)
        self.on_flush = on_flush

    @staticmethod
    def _packed(mask: np.ndarray) -> np.ndarray:
        values = np.asarray(mask, dtype=np.uint8).reshape(-1)
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Patch masks must be binary before compact storage.")
        return np.packbits(values, bitorder=MASK_BIT_ORDER)

    def append(self, *, row: dict[str, Any], masks: dict[str, np.ndarray]) -> None:
        if set(masks) != set(MASK_KINDS):
            raise ValueError("Every image must provide non_padding, foreground and valid masks.")
        packed_bytes = sum(int(self._packed(masks[key]).nbytes) for key in MASK_KINDS)
        if self.pending and (
            len(self.pending) >= self.images_per_shard
            or self.pending_bytes + packed_bytes > self.max_shard_bytes
        ):
            self.flush()
        self.pending.append(_PendingImage(row=row, masks=masks))
        self.pending_bytes += packed_bytes

    def flush(self) -> None:
        if not self.pending:
            return
        packed_by_kind = {
            key: [self._packed(item.masks[key]) for item in self.pending]
            for key in MASK_KINDS
        }
        shard_hashes: dict[str, str] = {}
        shard_keys: dict[str, str] = {}
        for key in MASK_KINDS:
            array = np.concatenate(packed_by_kind[key]) if len(packed_by_kind[key]) > 1 else packed_by_kind[key][0]
            relative = f"assets/arrays/{key}_patch_masks_{self.shard_index:05d}.npy"
            path = self.release_root / relative
            atomic_write_bytes(path, _npy_bytes(array.astype(np.uint8, copy=False)))
            shard_keys[key] = relative
            shard_hashes[key] = sha256_file(path)
        offsets = {key: 0 for key in MASK_KINDS}
        for item_index, item in enumerate(self.pending):
            refs: dict[str, dict[str, Any]] = {}
            for key in MASK_KINDS:
                logical = np.asarray(item.masks[key], dtype=np.uint8).reshape(-1)
                packed = packed_by_kind[key][item_index]
                refs[key] = {
                    "shard_key": shard_keys[key],
                    "offset": offsets[key],
                    "length": int(logical.size),
                    "storage_length": int(packed.size),
                    "dtype": "uint8",
                    "shape": [int(logical.size)],
                    "sha256": shard_hashes[key],
                    "storage_version": MASK_SHARD_STORAGE_VERSION,
                    "packing": "np.packbits",
                    "bit_order": MASK_BIT_ORDER,
                }
                offsets[key] += int(packed.size)
            item.row["non_padding_patch_mask_ref"] = refs["non_padding"]
            item.row["foreground_content_patch_mask_ref"] = refs["foreground"]
            item.row["valid_content_mask_ref"] = refs["valid"]
        if self.on_flush is not None:
            self.on_flush(self.shard_index, [item.row for item in self.pending])
        self.pending.clear()
        self.pending_bytes = 0
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


class MaskShardReader:
    """Validate immutable shard hashes while bounding simultaneously open mmaps."""

    def __init__(self, release_root: Path, *, max_open_shards: int | None = None) -> None:
        self.release_root = release_root
        if max_open_shards is None:
            max_open_shards = int(os.environ.get("HSM_E1_MASK_CACHE_SHARDS", "12"))
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer.")
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self._validated_hashes: set[str] = set()

    @staticmethod
    def _close_array(array: np.ndarray) -> None:
        mmap_handle = getattr(array, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def close(self) -> None:
        while self._arrays:
            _, array = self._arrays.popitem(last=False)
            self._close_array(array)

    def read(self, reference: dict[str, Any], *, expected_length: int) -> np.ndarray:
        required = {"shard_key", "offset", "length", "storage_length", "dtype", "shape", "sha256", "storage_version", "packing", "bit_order"}
        missing = sorted(required - set(reference))
        if missing:
            raise ValueError(f"Compact mask reference misses fields: {', '.join(missing)}")
        if reference["storage_version"] != MASK_SHARD_STORAGE_VERSION or reference["packing"] != "np.packbits":
            raise ValueError("Unsupported compact Patch-mask storage version.")
        if reference["bit_order"] != MASK_BIT_ORDER:
            raise ValueError("Unsupported compact Patch-mask bit order.")
        key = str(reference["shard_key"])
        path = self.release_root / key
        if not path.is_file():
            raise FileNotFoundError(f"Compact mask shard is missing: {path}")
        if key not in self._validated_hashes:
            if sha256_file(path) != str(reference["sha256"]):
                raise ValueError(f"Compact mask shard hash mismatch: {path}")
            self._validated_hashes.add(key)
        if key not in self._arrays:
            while len(self._arrays) >= self.max_open_shards:
                _, evicted = self._arrays.popitem(last=False)
                self._close_array(evicted)
            self._arrays[key] = np.load(path, allow_pickle=False, mmap_mode="r")
        else:
            self._arrays.move_to_end(key)
        values = self._arrays[key]
        offset, storage_length, length = int(reference["offset"]), int(reference["storage_length"]), int(reference["length"])
        if offset < 0 or storage_length != (length + 7) // 8 or length != expected_length or offset + storage_length > values.size:
            raise ValueError("Compact mask reference offset/length is invalid.")
        if str(reference["dtype"]) != "uint8" or list(reference["shape"]) != [length]:
            raise ValueError("Compact mask reference dtype/shape is invalid.")
        decoded = np.unpackbits(values[offset : offset + storage_length], bitorder=MASK_BIT_ORDER)[:length].astype(np.uint8)
        if not np.isin(decoded, [0, 1]).all():
            raise ValueError("Patch content mask must decode to binary values.")
        return decoded


__all__ = ["BoundedMaskShardWriter", "MASK_BIT_ORDER", "MASK_SHARD_STORAGE_VERSION", "MaskShardReader"]

