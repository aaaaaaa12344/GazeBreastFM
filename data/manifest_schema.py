from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_JOINT_PRETRAIN_COLUMNS = ("image_id", "image_path", "modality")


def load_joint_pretrain_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    suffix = resolved_manifest_path.suffix.lower()
    if suffix == ".csv":
        with resolved_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = []
        with resolved_manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Manifest row in {resolved_manifest_path} must be an object."
                    )
                rows.append(payload)
    else:
        raise ValueError(
            f"Unsupported manifest format {resolved_manifest_path}. Expected .csv or .jsonl."
        )

    for row_index, row in enumerate(rows, start=1):
        row["__manifest_row_number__"] = row_index
    return rows


def summarize_manifest_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "required_columns_present": False,
            "missing_required_columns": list(REQUIRED_JOINT_PRETRAIN_COLUMNS),
            "present_columns": [],
        }
    present_columns = sorted({str(key) for row in rows for key in row.keys()})
    missing_required_columns = [
        column for column in REQUIRED_JOINT_PRETRAIN_COLUMNS if column not in present_columns
    ]
    return {
        "row_count": len(rows),
        "required_columns_present": not missing_required_columns,
        "missing_required_columns": missing_required_columns,
        "present_columns": present_columns,
    }
