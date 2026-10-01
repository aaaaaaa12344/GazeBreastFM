from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "stage1_semantic_soft_labels_topk_v1"
DEFAULT_AXIS_WEIGHTS = {
    "modality": 0.15,
    "view": 0.45,
    "laterality": 0.55,
    "density": 1.0,
    "finding": 2.5,
    "birads": 2.25,
    "benign_malignant_label": 2.0,
    "cancer_label": 2.0,
    "clinical_graph_node": 1.5,
    "raw_unconfirmed": 0.05,
}
RAW_UNCONFIRMED_FIELDS = ("label_raw", "density_raw")
UNKNOWN_VALUES = {"", "unknown", "unknown_not_provided", "missing", "na", "n/a", "none", "null"}
HIGH_VALUE_AXES = {"density", "finding", "birads", "benign_malignant_label", "cancer_label", "clinical_graph_node"}
LOW_INFORMATION_CANDIDATE_AXES = {"modality"}
DEFAULT_INCLUDE_AXES = tuple(
    axis for axis in DEFAULT_AXIS_WEIGHTS if axis != "raw_unconfirmed"
)
DEFAULT_PROFILE_CAP = 3
DEFAULT_MAX_POSTING_CANDIDATES = 256
DEFAULT_HIGH_FREQUENCY_TOKEN_THRESHOLD = 10000
DEFAULT_AUDIT_EXAMPLES = 50


@dataclass(frozen=True)
class ConceptToken:
    axis: str
    value: str
    token: str
    weight: float
    status: str


@dataclass(frozen=True)
class SampleProfile:
    row_index: int
    image_id: str
    modality: str
    tokens: tuple[ConceptToken, ...]
    excluded_raw_unconfirmed_axes: tuple[str, ...]

    @property
    def signature(self) -> str:
        return "|".join(sorted(token.token for token in self.tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage 1 sparse/top-k semantic soft labels without constructing "
            "a full NxN dense matrix."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-rules", type=Path, default=None)
    parser.add_argument("--text-prompts", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-axes", default=",".join(DEFAULT_INCLUDE_AXES))
    parser.add_argument("--exclude-axes", default="")
    parser.add_argument("--allow-raw-unconfirmed", action="store_true", default=False)
    parser.add_argument("--profile-cap", type=int, default=DEFAULT_PROFILE_CAP)
    parser.add_argument("--max-posting-candidates", type=int, default=DEFAULT_MAX_POSTING_CANDIDATES)
    parser.add_argument("--high-frequency-token-threshold", type=int, default=DEFAULT_HIGH_FREQUENCY_TOKEN_THRESHOLD)
    parser.add_argument("--audit-examples", type=int, default=DEFAULT_AUDIT_EXAMPLES)
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).lower()


def _is_known(value: Any) -> bool:
    return _norm(value) not in UNKNOWN_VALUES


def _parse_axis_set(value: str, default: tuple[str, ...] = ()) -> set[str]:
    values = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return set(values or default)


def _status(row: dict[str, str], axis: str) -> str:
    return _norm(row.get(f"{axis}_status"))


def _is_confirmed(row: dict[str, str], axis: str, value: str) -> bool:
    if not _is_known(value):
        return False
    status = _status(row, axis)
    if status:
        return status == "confirmed"
    return axis in {"modality", "view", "laterality"}


def _split_values(value: str) -> list[str]:
    normalized = value.replace(";", "|").replace(",", "|")
    return [item.strip() for item in normalized.split("|") if _is_known(item)]


def _token(axis: str, value: str, status: str, weights: dict[str, float]) -> ConceptToken:
    normalized = _norm(value)
    return ConceptToken(
        axis=axis,
        value=normalized,
        token=f"{axis}:{normalized}",
        weight=float(weights[axis]),
        status=status,
    )


def _extract_clinical_graph_tokens(row: dict[str, str], weights: dict[str, float]) -> list[ConceptToken]:
    raw = _clean(row.get("clinical_graph_nodes") or row.get("clinical_graph_concepts_json"))
    if not raw:
        return []
    tokens: list[ConceptToken] = []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else payload.get("nodes", []) if isinstance(payload, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = _norm(item.get("status"))
        value = _clean(item.get("concept_id") or item.get("id") or item.get("label"))
        if status == "confirmed" and _is_known(value):
            tokens.append(_token("clinical_graph_node", value, status, weights))
    return tokens


def _extract_profile(
    row_index: int,
    row: dict[str, str],
    include_axes: set[str],
    exclude_axes: set[str],
    allow_raw_unconfirmed: bool,
    weights: dict[str, float],
) -> SampleProfile:
    image_id = _clean(row.get("image_id") or row.get("sample_id"))
    if not image_id:
        raise ValueError(f"manifest row {row_index} is missing image_id/sample_id")
    modality = _norm(row.get("modality")) or "unknown"
    tokens: list[ConceptToken] = []
    excluded_raw: list[str] = []

    for axis in ("modality", "view", "laterality", "density", "birads", "benign_malignant_label", "cancer_label"):
        if axis not in include_axes or axis in exclude_axes:
            continue
        value = _clean(row.get(axis))
        if _is_confirmed(row, axis, value):
            tokens.append(_token(axis, value, _status(row, axis) or "confirmed", weights))

    if "finding" in include_axes and "finding" not in exclude_axes:
        for value in _split_values(_clean(row.get("finding") or row.get("finding_categories"))):
            if _is_confirmed(row, "finding", value):
                tokens.append(_token("finding", value, _status(row, "finding") or "confirmed", weights))

    if "clinical_graph_node" in include_axes and "clinical_graph_node" not in exclude_axes:
        tokens.extend(_extract_clinical_graph_tokens(row, weights))

    for raw_axis in RAW_UNCONFIRMED_FIELDS:
        raw_value = _clean(row.get(raw_axis))
        raw_status = _status(row, raw_axis) or _norm(row.get(f"{raw_axis}_status"))
        if raw_status == "raw_unconfirmed" and _is_known(raw_value):
            if allow_raw_unconfirmed:
                tokens.append(_token("raw_unconfirmed", f"{raw_axis}:{raw_value}", raw_status, weights))
            else:
                excluded_raw.append(raw_axis)

    deduped = tuple({token.token: token for token in tokens}.values())
    return SampleProfile(
        row_index=row_index,
        image_id=image_id,
        modality=modality,
        tokens=deduped,
        excluded_raw_unconfirmed_axes=tuple(sorted(set(excluded_raw))),
    )


def _posting_candidates(
    posting: list[int],
    source_index: int,
    limit: int,
) -> list[int]:
    if len(posting) <= limit:
        return [item for item in posting if item != source_index]
    selected: list[int] = []
    seen: set[int] = {source_index}
    source_position = bisect_left(posting, source_index)
    if source_position >= len(posting):
        source_position = len(posting) - 1
    offsets = list(range(1, min(limit, len(posting)) + 1))
    for offset in offsets:
        for direction in (-1, 1):
            candidate = posting[(source_position + direction * offset) % len(posting)]
            if candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
            if len(selected) >= limit:
                return selected
    return selected


def _score_pair(a: SampleProfile, b: SampleProfile) -> tuple[float, list[str], list[str]]:
    a_tokens = {token.token: token for token in a.tokens}
    b_tokens = {token.token: token for token in b.tokens}
    shared_tokens: list[str] = []
    shared_axes: set[str] = set()
    numerator = 0.0
    denominator = 0.0
    all_token_keys = set(a_tokens) | set(b_tokens)
    for key in all_token_keys:
        left = a_tokens.get(key)
        right = b_tokens.get(key)
        token = left or right
        if token is None:
            continue
        if a.modality != b.modality and token.axis in {"view", "laterality", "density"}:
            continue
        denominator += token.weight
        if left is not None and right is not None:
            numerator += token.weight
            shared_tokens.append(key)
            shared_axes.add(token.axis)
    if denominator <= 0.0:
        return 0.0, [], []
    return float(max(0.0, min(1.0, numerator / denominator))), sorted(shared_axes), sorted(shared_tokens)


def _select_topk(
    source: SampleProfile,
    scored: list[dict[str, Any]],
    top_k: int,
    profile_cap: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    profile_counts: Counter[str] = Counter()
    for candidate in sorted(scored, key=lambda item: (-float(item["score"]), item["profile_signature"], item["target_image_id"])):
        if len(selected) >= top_k:
            break
        if candidate["target_row_index"] == source.row_index:
            continue
        signature = str(candidate["profile_signature"])
        if profile_counts[signature] >= profile_cap:
            continue
        selected.append(candidate)
        profile_counts[signature] += 1
    return selected


def _build_profiles(
    rows: list[dict[str, str]],
    include_axes: set[str],
    exclude_axes: set[str],
    allow_raw_unconfirmed: bool,
    weights: dict[str, float],
) -> list[SampleProfile]:
    profiles: list[SampleProfile] = []
    seen_ids: set[str] = set()
    for expected_index, row in enumerate(rows):
        raw_index = _clean(row.get("row_index"))
        row_index = int(raw_index) if raw_index else expected_index
        if row_index != expected_index:
            raise ValueError(
                f"manifest row_index must be contiguous from 0: expected {expected_index}, got {row_index}"
            )
        profile = _extract_profile(
            row_index=row_index,
            row=row,
            include_axes=include_axes,
            exclude_axes=exclude_axes,
            allow_raw_unconfirmed=allow_raw_unconfirmed,
            weights=weights,
        )
        if profile.image_id in seen_ids:
            raise ValueError(f"duplicate image_id in manifest: {profile.image_id}")
        seen_ids.add(profile.image_id)
        profiles.append(profile)
    return profiles


def _build_topk(
    profiles: list[SampleProfile],
    top_k: int,
    profile_cap: int,
    max_posting_candidates: int,
    high_frequency_token_threshold: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for profile in profiles:
        for token in profile.tokens:
            inverted[token.token].append(profile.row_index)
    for posting in inverted.values():
        posting.sort()

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    stats = {
        "empty_neighbor_count": 0,
        "low_candidate_count": 0,
        "skipped_high_frequency_token_count": 0,
        "low_information_token_skipped_count": 0,
    }
    candidate_generation_axes: set[str] = set()
    for source in profiles:
        candidate_indices: set[int] = set()
        for token in source.tokens:
            if token.axis in LOW_INFORMATION_CANDIDATE_AXES:
                stats["low_information_token_skipped_count"] += 1
                continue
            posting = inverted[token.token]
            if len(posting) > high_frequency_token_threshold:
                stats["skipped_high_frequency_token_count"] += 1
                continue
            candidate_generation_axes.add(token.axis)
            candidate_indices.update(
                _posting_candidates(posting, source.row_index, max_posting_candidates)
            )
        scored: list[dict[str, Any]] = []
        for target_index in sorted(candidate_indices):
            target = profiles[target_index]
            score, shared_axes, shared_tokens = _score_pair(source, target)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "target_row_index": target.row_index,
                    "target_image_id": target.image_id,
                    "score": round(score, 6),
                    "matched_axes": shared_axes,
                    "matched_tokens": shared_tokens,
                    "profile_signature": target.signature,
                }
            )
        neighbors = _select_topk(source, scored, top_k=top_k, profile_cap=profile_cap)
        if not neighbors:
            stats["empty_neighbor_count"] += 1
        if len(neighbors) < top_k:
            stats["low_candidate_count"] += 1
        scores.extend(float(item["score"]) for item in neighbors)
        rows.append(
            {
                "source_row_index": source.row_index,
                "source_image_id": source.image_id,
                "self_score": 1.0,
                "neighbors": [
                    {key: value for key, value in item.items() if key != "profile_signature"}
                    for item in neighbors
                ],
            }
        )
    stats["candidate_generation_axes"] = sorted(candidate_generation_axes)
    return rows, scores, stats


def _write_npz(path: Path, profiles: list[SampleProfile], topk_rows: list[dict[str, Any]], top_k: int) -> None:
    indptr = [0]
    indices: list[int] = []
    scores: list[float] = []
    for row in topk_rows:
        for neighbor in row["neighbors"]:
            indices.append(int(neighbor["target_row_index"]))
            scores.append(float(neighbor["score"]))
        indptr.append(len(indices))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        row_indices=np.asarray([profile.row_index for profile in profiles], dtype=np.int64),
        image_ids=np.asarray([profile.image_id for profile in profiles]),
        top_k=np.asarray([top_k], dtype=np.int64),
        schema_version=np.asarray([SCHEMA_VERSION]),
        self_entry_policy=np.asarray(["diagonal_forced_to_1_at_runtime"]),
        symmetrization_strategy=np.asarray(["max(score_ij,score_ji)"]),
    )


def _write_semantic_manifest(path: Path, profiles: list[SampleProfile]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "image_id"])
        writer.writeheader()
        for profile in profiles:
            writer.writerow({"row_index": profile.row_index, "image_id": profile.image_id})


def _summary(
    manifest_path: Path,
    output_dir: Path,
    profiles: list[SampleProfile],
    topk_rows: list[dict[str, Any]],
    scores: list[float],
    top_k: int,
    include_axes: set[str],
    exclude_axes: set[str],
    allow_raw_unconfirmed: bool,
    profile_cap: int,
    high_frequency_token_threshold: int,
    stats: dict[str, Any],
    mapping_rules_path: Path | None,
    text_prompts_path: Path | None,
) -> dict[str, Any]:
    axis_counts = Counter(token.axis for profile in profiles for token in profile.tokens)
    raw_excluded = Counter(axis for profile in profiles for axis in profile.excluded_raw_unconfirmed_axes)
    low_information_count = 0
    weak_signal_count = 0
    profile_distribution = Counter(profile.signature for profile in profiles)
    for profile in profiles:
        axes = {token.axis for token in profile.tokens if token.axis != "modality"}
        strong_axes = axes & HIGH_VALUE_AXES
        if len(axes) <= 2 and not strong_axes:
            low_information_count += 1
        if not strong_axes:
            weak_signal_count += 1
    neighbor_counts = [len(row["neighbors"]) for row in topk_rows]
    signal_strength = "strong" if scores and mean(scores) >= 0.5 and weak_signal_count < len(profiles) / 2 else "weak"
    if not scores:
        signal_strength = "empty"
    elif weak_signal_count and weak_signal_count < len(profiles) / 2:
        signal_strength = "mixed"
    warnings: list[str] = []
    if allow_raw_unconfirmed:
        warnings.append(
            "RISK: allow_raw_unconfirmed=true; raw_unconfirmed labels were allowed into similarity."
        )
    if weak_signal_count:
        warnings.append(
            "weak_semantic_signal_detected: many rows rely on modality/view/laterality only."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "sample_count": len(profiles),
        "top_k": top_k,
        "mapping_rules_used": mapping_rules_path is not None and mapping_rules_path.exists(),
        "text_prompts_used": text_prompts_path is not None and text_prompts_path.exists(),
        "semantic_soft_label_format": "sparse_topk",
        "dense_matrix_generated": False,
        "candidate_generation": "inverted_index_with_deterministic_posting_cap",
        "candidate_generation_axes": stats.get("candidate_generation_axes", []),
        "high_frequency_token_threshold": high_frequency_token_threshold,
        "skipped_high_frequency_token_count": int(stats["skipped_high_frequency_token_count"]),
        "high_frequency_token_skipped_count": int(stats["skipped_high_frequency_token_count"]),
        "low_information_token_skipped_count": int(stats["low_information_token_skipped_count"]),
        "symmetrization_strategy": "max(score_ij,score_ji)",
        "self_entry_policy": "diagonal_forced_to_1_at_runtime",
        "missing_edge_policy": "soft_label_0",
        "allow_raw_unconfirmed": allow_raw_unconfirmed,
        "raw_unconfirmed_risk": "RISK_ENABLED" if allow_raw_unconfirmed else "disabled",
        "raw_unconfirmed_axes_excluded": dict(sorted(raw_excluded.items())),
        "include_axes": sorted(include_axes),
        "exclude_axes": sorted(exclude_axes),
        "axis_weights": dict(DEFAULT_AXIS_WEIGHTS),
        "axis_token_counts": dict(sorted(axis_counts.items())),
        "profile_cap": profile_cap,
        "unique_profile_count": len(profile_distribution),
        "largest_profile_size": max(profile_distribution.values()) if profile_distribution else 0,
        "low_information_profile_count": low_information_count,
        "weak_semantic_signal_count": weak_signal_count,
        "semantic_signal_strength": signal_strength,
        "mean_neighbor_count": mean(neighbor_counts) if neighbor_counts else 0.0,
        "empty_neighbor_count": int(stats["empty_neighbor_count"]),
        "low_candidate_count": int(stats["low_candidate_count"]),
        "score_min": min(scores) if scores else 0.0,
        "score_mean": mean(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "warnings": warnings,
    }


def build_sparse_semantic_soft_labels(
    manifest_path: Path,
    output_dir: Path,
    top_k: int = 20,
    include_axes: set[str] | None = None,
    exclude_axes: set[str] | None = None,
    allow_raw_unconfirmed: bool = False,
    profile_cap: int = DEFAULT_PROFILE_CAP,
    max_posting_candidates: int = DEFAULT_MAX_POSTING_CANDIDATES,
    high_frequency_token_threshold: int = DEFAULT_HIGH_FREQUENCY_TOKEN_THRESHOLD,
    audit_examples: int = DEFAULT_AUDIT_EXAMPLES,
    mapping_rules_path: Path | None = None,
    text_prompts_path: Path | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if profile_cap <= 0:
        raise ValueError("profile_cap must be positive")
    if max_posting_candidates <= 0:
        raise ValueError("max_posting_candidates must be positive")
    if high_frequency_token_threshold <= 0:
        raise ValueError("high_frequency_token_threshold must be positive")
    rows, _fieldnames = _read_csv(manifest_path)
    include = set(include_axes or DEFAULT_INCLUDE_AXES)
    exclude = set(exclude_axes or set())
    weights = dict(DEFAULT_AXIS_WEIGHTS)
    profiles = _build_profiles(
        rows,
        include_axes=include,
        exclude_axes=exclude,
        allow_raw_unconfirmed=allow_raw_unconfirmed,
        weights=weights,
    )
    topk_rows, scores, stats = _build_topk(
        profiles,
        top_k=top_k,
        profile_cap=profile_cap,
        max_posting_candidates=max_posting_candidates,
        high_frequency_token_threshold=high_frequency_token_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "stage1_semantic_soft_labels_topk.npz"
    jsonl_path = output_dir / "stage1_semantic_soft_labels_topk.jsonl"
    summary_path = output_dir / "semantic_soft_label_summary.json"
    audit_path = output_dir / "semantic_soft_label_audit_examples.jsonl"
    semantic_manifest_path = output_dir / "stage1_semantic_soft_label_manifest.csv"
    _write_npz(npz_path, profiles, topk_rows, top_k)
    _write_jsonl(jsonl_path, topk_rows)
    _write_semantic_manifest(semantic_manifest_path, profiles)
    audit_rows = []
    for row in topk_rows[: max(0, audit_examples)]:
        profile = profiles[int(row["source_row_index"])]
        audit_rows.append(
            {
                "source_row_index": row["source_row_index"],
                "source_image_id": row["source_image_id"],
                "profile_signature": profile.signature,
                "excluded_raw_unconfirmed_axes": list(profile.excluded_raw_unconfirmed_axes),
                "neighbor_count": len(row["neighbors"]),
                "neighbors_preview": row["neighbors"][:5],
            }
        )
    _write_jsonl(audit_path, audit_rows)
    summary = _summary(
        manifest_path=manifest_path,
        output_dir=output_dir,
        profiles=profiles,
        topk_rows=topk_rows,
        scores=scores,
        top_k=top_k,
        include_axes=include,
        exclude_axes=exclude,
        allow_raw_unconfirmed=allow_raw_unconfirmed,
        profile_cap=profile_cap,
        high_frequency_token_threshold=high_frequency_token_threshold,
        stats=stats,
        mapping_rules_path=mapping_rules_path,
        text_prompts_path=text_prompts_path,
    )
    summary["outputs"] = {
        "topk_npz": str(npz_path),
        "topk_jsonl": str(jsonl_path),
        "semantic_manifest": str(semantic_manifest_path),
        "summary_json": str(summary_path),
        "audit_examples_jsonl": str(audit_path),
    }
    summary["optional_inputs"] = {
        "mapping_rules_path": str(mapping_rules_path) if mapping_rules_path else None,
        "text_prompts_path": str(text_prompts_path) if text_prompts_path else None,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_sparse_semantic_soft_labels(
        manifest_path=args.manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        top_k=int(args.top_k),
        include_axes=_parse_axis_set(args.include_axes, DEFAULT_INCLUDE_AXES),
        exclude_axes=_parse_axis_set(args.exclude_axes),
        allow_raw_unconfirmed=bool(args.allow_raw_unconfirmed),
        profile_cap=int(args.profile_cap),
        max_posting_candidates=int(args.max_posting_candidates),
        high_frequency_token_threshold=int(args.high_frequency_token_threshold),
        audit_examples=int(args.audit_examples),
        mapping_rules_path=args.mapping_rules.expanduser().resolve() if args.mapping_rules else None,
        text_prompts_path=args.text_prompts.expanduser().resolve() if args.text_prompts else None,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
