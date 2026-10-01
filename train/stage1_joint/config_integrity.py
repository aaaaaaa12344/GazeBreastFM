from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def assert_formal_init_config_checksum(
    *,
    payload: dict[str, Any],
    current_config_checksum: str,
) -> None:
    stored = str(payload.get("config_checksum", "")).strip()
    if not stored:
        raise RuntimeError(
            "formal_init_state.pt missing required config_checksum for formal-production launch. "
            "Rebuild formal_init_state.pt from the resolved production config."
        )
    if stored != current_config_checksum:
        raise RuntimeError(
            "Config checksum mismatch: the resolved config has been modified since "
            "formal_init_state.pt was built. "
            f"Stored checksum: {stored[:16]}... Current checksum: {current_config_checksum[:16]}..."
        )


__all__ = ["assert_formal_init_config_checksum", "sha256_file"]
