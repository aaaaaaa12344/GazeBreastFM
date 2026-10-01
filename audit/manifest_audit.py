from __future__ import annotations

from pathlib import Path
from typing import Any

from breast_pretrain.data import load_joint_pretrain_manifest_rows, summarize_manifest_schema


def audit_joint_pretrain_manifest(manifest_path: str | Path) -> dict[str, Any]:
    rows = load_joint_pretrain_manifest_rows(manifest_path)
    schema_summary = summarize_manifest_schema(rows)
    missing_image_paths: list[str] = []
    missing_attention_entries = 0
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    for row in rows:
        image_path = str(row.get("image_path") or "").strip()
        if not image_path:
            continue
        resolved_image_path = Path(image_path)
        if not resolved_image_path.is_absolute():
            resolved_image_path = (resolved_manifest_path.parent / resolved_image_path).resolve()
        if not resolved_image_path.exists():
            missing_image_paths.append(str(resolved_image_path))
        if not str(row.get("attention_map_path") or "").strip():
            missing_attention_entries += 1
    status = "pass"
    if missing_image_paths or not schema_summary["required_columns_present"]:
        status = "fail"
    return {
        "status": status,
        "schema": schema_summary,
        "missing_image_path_count": len(missing_image_paths),
        "missing_image_paths": missing_image_paths[:20],
        "missing_attention_entry_count": missing_attention_entries,
    }
