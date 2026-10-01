"""Non-authoritative, bounded local cache for source-runtime canonical tensors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


CACHE_SCHEMA_VERSION = "runtime_canonical_cache_v1"


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact C-contiguous float32 CHW tensor payload."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SourceRuntimeCacheConfig:
    enabled: bool = False
    root: Path | None = None
    max_bytes: int = 0
    write_enabled: bool = True
    verify_on_read: bool = True
    eviction_policy: str = "lru"
    fail_open: bool = True
    atomic_write: bool = True
    forbidden_roots: tuple[Path, ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SourceRuntimeCacheConfig":
        data = raw or {}
        enabled = bool(data.get("enabled", False))
        root_value = data.get("root")
        root = Path(str(root_value)).expanduser().resolve() if root_value else None
        if enabled and root is None:
            raise ValueError("source_runtime_cache.enabled=true requires an explicit cache root.")
        max_bytes = int(data.get("max_bytes", 0) or 0)
        if max_bytes < 0:
            raise ValueError("source_runtime_cache.max_bytes must be >= 0.")
        policy = str(data.get("eviction_policy", "lru")).strip().lower() or "lru"
        if policy != "lru":
            raise ValueError("source_runtime_cache.eviction_policy must be lru.")
        return cls(
            enabled=enabled, root=root, max_bytes=max_bytes,
            write_enabled=bool(data.get("write_enabled", True)),
            verify_on_read=bool(data.get("verify_on_read", True)),
            eviction_policy=policy, fail_open=bool(data.get("fail_open", True)),
            atomic_write=bool(data.get("atomic_write", True)),
            forbidden_roots=tuple(Path(str(item)).expanduser().resolve() for item in data.get("forbidden_roots", []) or []),
        )

    def to_resolved_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "root": str(self.root) if self.root else None,
            "max_bytes": self.max_bytes, "write_enabled": self.write_enabled,
            "verify_on_read": self.verify_on_read, "eviction_policy": self.eviction_policy,
            "fail_open": self.fail_open, "atomic_write": self.atomic_write,
            "forbidden_roots": [str(item) for item in self.forbidden_roots],
        }


@dataclass
class RuntimeCanonicalCacheStats:
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalid: int = 0
    cache_writes: int = 0
    cache_write_failures: int = 0
    cache_evictions: int = 0
    cache_bytes_written: int = 0
    source_reconstructions: int = 0
    cache_oversize_skips: int = 0
    cache_orphan_cleanups: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(vars(self))


@dataclass
class RuntimeCanonicalCache:
    config: SourceRuntimeCacheConfig
    stats: RuntimeCanonicalCacheStats = field(default_factory=RuntimeCanonicalCacheStats)

    def _paths(self, key: str) -> tuple[Path, Path]:
        if not self.config.root:
            raise RuntimeError("Cache root is unavailable while cache is disabled.")
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("canonical_reconstruction_key must be a lowercase SHA-256 hex digest.")
        return self.config.root / f"{key}.npy", self.config.root / f"{key}.json"

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def _fail_open(self, action: str, exc: Exception) -> None:
        if not self.config.fail_open:
            raise RuntimeError(f"Runtime canonical cache {action} failed: {exc}") from exc
        warnings.warn(f"Runtime canonical cache {action} failed; continuing without cache: {exc}", stacklevel=3)

    def get(self, key: str, expected: dict[str, Any]) -> torch.Tensor | None:
        if not self.config.enabled:
            return None
        try:
            array_path, metadata_path = self._paths(key)
            if not array_path.is_file() or not metadata_path.is_file():
                self.stats.cache_misses += 1
                return None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError("metadata schema is absent or incompatible")
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(f"metadata {field} differs from the frozen E1 contract")
            array = np.load(array_path, allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != metadata.get("tensor_shape"):
                raise ValueError("cached tensor dtype or shape is invalid")
            tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            if self.config.verify_on_read and canonical_tensor_sha256(tensor) != metadata.get("tensor_sha256"):
                raise ValueError("cached tensor SHA256 differs from metadata")
            os.utime(array_path, None)
            self.stats.cache_hits += 1
            return tensor
        except (OSError, ValueError, json.JSONDecodeError, EOFError) as exc:
            self.stats.cache_invalid += 1
            self._fail_open("read/verify", exc)
            return None

    def put(self, key: str, tensor: torch.Tensor, metadata: dict[str, Any]) -> None:
        if not self.config.enabled or not self.config.write_enabled:
            return
        try:
            array_path, metadata_path = self._paths(key)
            array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32, copy=False))
            payload_metadata = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "canonical_reconstruction_key": key,
                **metadata,
                "tensor_shape": list(array.shape),
                "tensor_dtype": "float32",
                "tensor_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "created_at": time.time(),
                "writer_pid": os.getpid(),
            }
            npy_buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
            np.save(npy_buffer, array, allow_pickle=False)
            npy_buffer.seek(0)
            self._atomic_write_bytes(array_path, npy_buffer.read())
            self._atomic_write_bytes(
                metadata_path,
                (json.dumps(payload_metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
            )
            self.stats.cache_writes += 1
            self.stats.cache_bytes_written += array_path.stat().st_size + metadata_path.stat().st_size
            self.evict(exclude_keys={key})
        except (OSError, ValueError, TypeError) as exc:
            self.stats.cache_write_failures += 1
            self._fail_open("write", exc)

    def evict(self, *, exclude_keys: set[str] | None = None) -> None:
        if not self.config.enabled or self.config.max_bytes <= 0 or not self.config.root:
            return
        try:
            excluded = exclude_keys or set()
            self.config.root.mkdir(parents=True, exist_ok=True)
            entries: list[tuple[float, str, Path, Path, int]] = []
            total = 0
            npy_paths = {path.stem: path for path in self.config.root.glob("*.npy") if len(path.stem) == 64 and all(char in "0123456789abcdef" for char in path.stem)}
            json_paths = {path.stem: path for path in self.config.root.glob("*.json") if len(path.stem) == 64 and all(char in "0123456789abcdef" for char in path.stem)}
            for key in sorted(set(npy_paths) | set(json_paths)):
                array_path = npy_paths.get(key)
                metadata_path = json_paths.get(key)
                if array_path is None or metadata_path is None:
                    for path in (array_path, metadata_path):
                        if path is not None:
                            path.unlink(missing_ok=True)
                    self.stats.cache_orphan_cleanups += 1
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    array = np.load(array_path, allow_pickle=False)
                    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION or array.dtype != np.float32 or list(array.shape) != metadata.get("tensor_shape") or hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest() != metadata.get("tensor_sha256"):
                        raise ValueError("invalid cache entry")
                    size = array_path.stat().st_size + metadata_path.stat().st_size
                    total += size
                    entries.append((array_path.stat().st_atime, key, array_path, metadata_path, size))
                except (OSError, ValueError, json.JSONDecodeError, EOFError):
                    array_path.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
                    self.stats.cache_orphan_cleanups += 1
            now = time.time()
            for temporary_path in self.config.root.glob(".*.tmp"):
                if now - temporary_path.stat().st_mtime > 3600 and len(temporary_path.name.split(".")) >= 4:
                    temporary_path.unlink(missing_ok=True); self.stats.cache_orphan_cleanups += 1
            for _atime, key, array_path, metadata_path, size in sorted(entries, key=lambda item: (item[1] in excluded, item[0])):
                if total <= self.config.max_bytes:
                    break
                if key in excluded and size <= self.config.max_bytes:
                    continue
                array_path.unlink(missing_ok=True); metadata_path.unlink(missing_ok=True)
                total -= size; self.stats.cache_evictions += 1
                if key in excluded and size > self.config.max_bytes:
                    self.stats.cache_oversize_skips += 1
        except OSError as exc:
            self._fail_open("LRU eviction", exc)


__all__ = [
    "CACHE_SCHEMA_VERSION", "RuntimeCanonicalCache", "RuntimeCanonicalCacheStats",
    "SourceRuntimeCacheConfig", "canonical_tensor_sha256",
]
