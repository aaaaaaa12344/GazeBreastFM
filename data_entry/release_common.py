"""Shared immutable-release helpers for Dataset Entry V2.

The helpers in this module deliberately avoid dataset and model semantics so
every Dataset Entry stage writes the same auditable manifest and receipt shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_bytes(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object.")
            rows.append(value)
    return rows


def _immutable_receipt_verify_mode() -> str:
    mode = os.environ.get("HSM_IMMUTABLE_RECEIPT_VERIFY_MODE", "full").strip().lower()
    if mode not in {"full", "signed_inventory"}:
        raise ValueError(
            "HSM_IMMUTABLE_RECEIPT_VERIFY_MODE must be 'full' or 'signed_inventory'."
        )
    return mode


def _validate_signed_inventory_schema(expected: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for number, item in enumerate(expected, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Signed output_inventory row {number} is not an object.")
        relative = str(item.get("path") or "").strip()
        digest = str(item.get("sha256") or "").strip().lower()
        size = item.get("bytes")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative in seen
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError(f"Signed output_inventory row {number} is invalid.")
        seen.add(relative)


def load_pass_receipt(root: Path) -> tuple[dict[str, Any], Path]:
    """Load and close an immutable formal release before consuming it."""
    candidates = (root / "receipts" / "terminal_receipt.json", root / "terminal_receipt.json")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing terminal receipt under release root: {root}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable terminal receipt: {path}") from exc
    if not isinstance(payload, dict) or not bool(payload.get("pass")):
        raise ValueError(f"Upstream terminal receipt is not PASS: {path}")
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal release lacks release_manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable release manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Release manifest must be an object: {manifest_path}")
    for key in ("schema_version", "release_id", "dataset_id"):
        if not str(payload.get(key) or "").strip() or payload.get(key) != manifest.get(key):
            raise ValueError(f"Terminal receipt/release manifest mismatch on {key}: {root}")
    inventory = payload.get("output_inventory")
    if not isinstance(inventory, list):
        raise ValueError(f"Formal terminal receipt lacks output_inventory: {path}")
    verify_mode = _immutable_receipt_verify_mode()
    if verify_mode == "full":
        verify_relative_file_inventory(root, inventory, exclude={path.relative_to(root).as_posix()})
    else:
        _validate_signed_inventory_schema(inventory)
    return payload, path


def relative_file_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return records


def verify_relative_file_inventory(
    root: Path,
    expected: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> None:
    """Require the signed inventory to equal the actual immutable files."""
    excluded = exclude or set()
    actual = [item for item in relative_file_inventory(root) if item["path"] not in excluded]
    if actual != expected:
        raise ValueError("Release output inventory does not match the actual release files.")


def write_release_receipt(
    *,
    release_root: Path,
    schema_version: str,
    release_id: str,
    dataset_id: str,
    upstream_release_ids: dict[str, str],
    resolved_config: dict[str, Any],
    terminal_status: str,
    counts: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> Path:
    receipt_path = release_root / "receipts" / "terminal_receipt.json"
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "release_id": release_id,
        "dataset_id": dataset_id,
        "upstream_release_ids": dict(sorted(upstream_release_ids.items())),
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_text(canonical_json(resolved_config)),
        # The terminal receipt is self-referential and therefore intentionally
        # excluded.  Everything else is signed before the receipt is written.
        "output_inventory": relative_file_inventory(release_root),
        "counts": counts,
        "terminal_status": terminal_status,
        "pass": terminal_status.startswith("PASS") or terminal_status.startswith("CLOSED_PASS"),
        "created_at": utc_now(),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(receipt_path, payload)
    return receipt_path


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "canonical_json",
    "read_jsonl",
    "load_pass_receipt",
    "relative_file_inventory",
    "verify_relative_file_inventory",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "utc_now",
    "write_release_receipt",
]

