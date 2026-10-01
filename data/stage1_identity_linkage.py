from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breast_pretrain.data.bucketed_stage1_dataloader import normalize_stage1_modality


IMAGE_PATH_ALIASES = ("image_path", "input_image_path", "stage0_inference_image_path")
EXACT_IMAGE_ID = "exact_image_id"
EXACT_CANONICAL_STAGE1_IMAGE_ID = "exact_canonical_stage1_image_id"
EXACT_RESOLVED_IMAGE_PATH = "exact_resolved_image_path"


@dataclass(frozen=True)
class IdentityLinkageMatch:
    source_index: int
    source_key: tuple[str, str]
    audited_key: tuple[str, str]
    audited_record: dict[str, Any]
    method: str
    canonical_stage1_image_id: str
    audited_prior_image_id: str
    stage0_inference_image_id: str
    source_image_path: str
    audited_image_path: str
    resolved_source_image_path: str
    resolved_audited_image_path: str
    linkage_unique: bool = True


@dataclass(frozen=True)
class IdentityLinkageResult:
    matches: list[IdentityLinkageMatch]
    excluded_source_indices: list[int]
    used_audited_keys: set[tuple[str, str]]
    summary: dict[str, Any]


def clean(value: Any) -> str:
    return str(value or "").strip()


def identity_key(image_id: Any, modality: Any) -> tuple[str, str]:
    return clean(image_id), normalize_stage1_modality(modality)


def _first_present(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = clean(row.get(alias))
        if value:
            return value
    return ""


def _resolve_existing_path(raw_value: str, *, base_dir: Path, field_name: str, image_id: str) -> Path:
    value = clean(raw_value)
    if not value:
        raise ValueError(f"Identity linkage row {image_id} has empty {field_name}.")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, _project_root() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Identity linkage row {image_id} missing {field_name}: {value}. Checked: {checked}")


def _try_resolve_existing_path(raw_value: str, *, base_dir: Path) -> str:
    value = clean(raw_value)
    if not value:
        return ""
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, _project_root() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return str(resolved)
    return ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage0_inference_image_id(audited_row: dict[str, str], audited_image_id: str) -> str:
    return (
        clean(audited_row.get("stage0_inference_image_id"))
        or clean(audited_row.get("inference_image_id"))
        or audited_image_id
    )


def _raw_audited_image_path(audited_row: dict[str, str]) -> str:
    return _first_present(audited_row, IMAGE_PATH_ALIASES)


def _audited_image_id(audited_record: dict[str, Any]) -> str:
    return clean(audited_record.get("audited_image_id")) or clean(audited_record["row"].get("image_id"))


def _audited_canonical_stage1_image_id(audited_record: dict[str, Any]) -> str:
    return clean(audited_record.get("canonical_stage1_image_id")) or clean(
        audited_record["row"].get("canonical_stage1_image_id")
    )


def _resolve_optional_existing_path(raw_value: str, *, base_dir: Path) -> str:
    return _try_resolve_existing_path(raw_value, base_dir=base_dir)


def _validate_exact_match_paths(
    *,
    method: str,
    source_key: tuple[str, str],
    source_path: str,
    audited_key: tuple[str, str],
    audited_path: str,
    source_manifest_dir: Path,
    audited_manifest_dir: Path,
) -> tuple[str, str]:
    resolved_source = _resolve_optional_existing_path(source_path, base_dir=source_manifest_dir)
    resolved_audited = _resolve_optional_existing_path(audited_path, base_dir=audited_manifest_dir)
    if resolved_source and resolved_audited and resolved_source != resolved_audited:
        raise ValueError(
            f"Identity linkage path conflict for {method}: source={source_key} "
            f"resolved_source_image_path={resolved_source}; audited={audited_key} "
            f"resolved_audited_image_path={resolved_audited}"
        )
    return resolved_source, resolved_audited


def _build_match(
    *,
    source_index: int,
    source_row: dict[str, str],
    source_key: tuple[str, str],
    audited_key: tuple[str, str],
    audited_record: dict[str, Any],
    method: str,
    source_manifest_dir: Path,
    resolved_source_path: str = "",
    resolved_audited_path: str = "",
) -> IdentityLinkageMatch:
    audited_row = audited_record["row"]
    audited_image_id = _audited_image_id(audited_record)
    source_image_path = clean(source_row.get("image_path"))
    audited_image_path = _raw_audited_image_path(audited_row)
    if resolved_source_path or resolved_audited_path:
        resolved_source = resolved_source_path
        resolved_audited = resolved_audited_path
    else:
        resolved_source, resolved_audited = _validate_exact_match_paths(
            method=method,
            source_key=source_key,
            source_path=source_image_path,
            audited_key=audited_key,
            audited_path=audited_image_path,
            source_manifest_dir=source_manifest_dir,
            audited_manifest_dir=Path(audited_record["manifest_path"]).parent,
        )
    return IdentityLinkageMatch(
        source_index=source_index,
        source_key=source_key,
        audited_key=audited_key,
        audited_record=audited_record,
        method=method,
        canonical_stage1_image_id=source_key[0],
        audited_prior_image_id=audited_image_id,
        stage0_inference_image_id=_stage0_inference_image_id(audited_row, audited_image_id),
        source_image_path=source_image_path,
        audited_image_path=audited_image_path,
        resolved_source_image_path=resolved_source,
        resolved_audited_image_path=resolved_audited,
    )


def _duplicate_path_keys(
    path_keys: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    counts = Counter(path_keys)
    return sorted(key for key, count in counts.items() if count > 1)


def _unique_audited_index(
    audited: dict[tuple[str, str], dict[str, Any]],
    *,
    id_getter: Any,
    label: str,
    skip_empty: bool,
) -> dict[tuple[str, str], tuple[tuple[str, str], dict[str, Any]]]:
    items: dict[tuple[str, str], list[tuple[tuple[str, str], dict[str, Any]]]] = {}
    for record_key, record in audited.items():
        identity = clean(id_getter(record))
        if not identity and skip_empty:
            continue
        index_key = identity_key(identity, record.get("modality"))
        items.setdefault(index_key, []).append((record_key, record))
    duplicates = sorted(key for key, values in items.items() if len(values) > 1)
    if duplicates:
        raise ValueError(f"Duplicate audited {label} identity linkage key(s): {duplicates[:5]}")
    return {key: values[0] for key, values in items.items()}


def resolve_stage1_identity_linkage(
    *,
    source_rows: list[dict[str, str]],
    audited: dict[tuple[str, str], dict[str, Any]],
    source_manifest_dir: Path,
) -> IdentityLinkageResult:
    matches_by_source_index: dict[int, IdentityLinkageMatch] = {}
    used_audited_keys: set[tuple[str, str]] = set()
    method_counts = Counter()
    method_counts_by_modality: dict[str, Counter[str]] = {}
    ambiguous_count = 0
    source_keys = [identity_key(row.get("image_id"), row.get("modality")) for row in source_rows]
    audited_image_index = _unique_audited_index(
        audited,
        id_getter=_audited_image_id,
        label="image_id",
        skip_empty=False,
    )
    audited_canonical_index = _unique_audited_index(
        audited,
        id_getter=_audited_canonical_stage1_image_id,
        label="canonical_stage1_image_id",
        skip_empty=True,
    )
    source_key_counts = Counter(source_keys)
    duplicate_source_exact_keys = sorted(
        key
        for key, count in source_key_counts.items()
        if count > 1 and (key in audited_image_index or key in audited_canonical_index)
    )
    if duplicate_source_exact_keys:
        raise ValueError(f"Duplicate source identity linkage key(s): {duplicate_source_exact_keys[:5]}")

    for source_index, (source_row, source_key) in enumerate(zip(source_rows, source_keys)):
        indexed = audited_image_index.get(source_key)
        if indexed is None:
            continue
        audited_key, audited_record = indexed
        matches_by_source_index[source_index] = _build_match(
            source_index=source_index,
            source_row=source_row,
            source_key=source_key,
            audited_key=identity_key(_audited_image_id(audited_record), audited_record.get("modality")),
            audited_record=audited_record,
            method=EXACT_IMAGE_ID,
            source_manifest_dir=source_manifest_dir,
        )
        used_audited_keys.add(audited_key)
        method_counts[EXACT_IMAGE_ID] += 1
        method_counts_by_modality.setdefault(source_key[1], Counter())[EXACT_IMAGE_ID] += 1

    for source_index, (source_row, source_key) in enumerate(zip(source_rows, source_keys)):
        if source_index in matches_by_source_index:
            continue
        indexed = audited_canonical_index.get(source_key)
        if indexed is None:
            continue
        audited_key, audited_record = indexed
        if audited_key in used_audited_keys:
            continue
        matches_by_source_index[source_index] = _build_match(
            source_index=source_index,
            source_row=source_row,
            source_key=source_key,
            audited_key=identity_key(_audited_canonical_stage1_image_id(audited_record), audited_record.get("modality")),
            audited_record=audited_record,
            method=EXACT_CANONICAL_STAGE1_IMAGE_ID,
            source_manifest_dir=source_manifest_dir,
        )
        used_audited_keys.add(audited_key)
        method_counts[EXACT_CANONICAL_STAGE1_IMAGE_ID] += 1
        method_counts_by_modality.setdefault(source_key[1], Counter())[EXACT_CANONICAL_STAGE1_IMAGE_ID] += 1

    remaining_audited = {
        key: record for key, record in audited.items() if key not in used_audited_keys
    }
    if remaining_audited:
        fallback_sources = [
            (source_index, source_row, source_keys[source_index])
            for source_index, source_row in enumerate(source_rows)
            if source_index not in matches_by_source_index
        ]
        source_path_items: list[tuple[tuple[str, str], int, dict[str, str], tuple[str, str]]] = []
        for source_index, source_row, source_key in fallback_sources:
            raw_path = clean(source_row.get("image_path"))
            if not raw_path:
                continue
            resolved = _resolve_existing_path(
                raw_path,
                base_dir=source_manifest_dir,
                field_name="source image_path",
                image_id=source_key[0],
            )
            path_key = (str(resolved), source_key[1])
            source_path_items.append((path_key, source_index, source_row, source_key))

        audited_path_items: list[tuple[tuple[str, str], tuple[str, str], dict[str, Any]]] = []
        for audited_key, audited_record in remaining_audited.items():
            audited_row = audited_record["row"]
            raw_path = _raw_audited_image_path(audited_row)
            resolved = _resolve_existing_path(
                raw_path,
                base_dir=Path(audited_record["manifest_path"]).parent,
                field_name="audited image_path",
                image_id=audited_key[0],
            )
            path_key = (str(resolved), audited_key[1])
            audited_path_items.append((path_key, audited_key, audited_record))

        duplicate_source_paths = _duplicate_path_keys([item[0] for item in source_path_items])
        duplicate_audited_paths = _duplicate_path_keys([item[0] for item in audited_path_items])
        if duplicate_source_paths or duplicate_audited_paths:
            ambiguous_count = len(duplicate_source_paths) + len(duplicate_audited_paths)
            details = []
            if duplicate_source_paths:
                details.append(f"duplicate source resolved image_path keys={duplicate_source_paths[:5]}")
            if duplicate_audited_paths:
                details.append(f"duplicate audited resolved image_path keys={duplicate_audited_paths[:5]}")
            raise ValueError("Ambiguous identity linkage by resolved image_path: " + "; ".join(details))

        source_by_path = {path_key: (source_index, source_row, source_key) for path_key, source_index, source_row, source_key in source_path_items}
        for path_key, audited_key, audited_record in audited_path_items:
            source_item = source_by_path.get(path_key)
            if source_item is None:
                continue
            source_index, source_row, source_key = source_item
            if source_index in matches_by_source_index:
                continue
            match = _build_match(
                source_index=source_index,
                source_row=source_row,
                source_key=source_key,
                audited_key=audited_key,
                audited_record=audited_record,
                method=EXACT_RESOLVED_IMAGE_PATH,
                source_manifest_dir=source_manifest_dir,
                resolved_source_path=path_key[0],
                resolved_audited_path=path_key[0],
            )
            matches_by_source_index[source_index] = match
            used_audited_keys.add(audited_key)
            method_counts[EXACT_RESOLVED_IMAGE_PATH] += 1
            method_counts_by_modality.setdefault(source_key[1], Counter())[EXACT_RESOLVED_IMAGE_PATH] += 1

    unresolved_audited_count = len([key for key in audited if key not in used_audited_keys])
    matches = [matches_by_source_index[index] for index in sorted(matches_by_source_index)]
    excluded_source_indices = [
        index for index in range(len(source_rows)) if index not in matches_by_source_index
    ]
    return IdentityLinkageResult(
        matches=matches,
        excluded_source_indices=excluded_source_indices,
        used_audited_keys=used_audited_keys,
        summary={
            "exact_image_id_linkage_count": int(method_counts.get(EXACT_IMAGE_ID, 0)),
            "exact_canonical_stage1_image_id_linkage_count": int(
                method_counts.get(EXACT_CANONICAL_STAGE1_IMAGE_ID, 0)
            ),
            "exact_resolved_image_path_linkage_count": int(method_counts.get(EXACT_RESOLVED_IMAGE_PATH, 0)),
            "unresolved_identity_linkage_count": int(unresolved_audited_count),
            "ambiguous_identity_linkage_count": int(ambiguous_count),
            "identity_linkage_count_by_modality": {
                modality: dict(sorted(counter.items()))
                for modality, counter in sorted(method_counts_by_modality.items())
            },
        },
    )


__all__ = [
    "EXACT_CANONICAL_STAGE1_IMAGE_ID",
    "EXACT_IMAGE_ID",
    "EXACT_RESOLVED_IMAGE_PATH",
    "IdentityLinkageMatch",
    "IdentityLinkageResult",
    "clean",
    "identity_key",
    "resolve_stage1_identity_linkage",
]
