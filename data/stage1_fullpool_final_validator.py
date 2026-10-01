"""Independent final Stage 1 bundle validation layer (Work7 hotfix 1).

This validator is the single formal gate between a materialized final bundle
and ``formal_pretraining_authorization``.  It performs the Stage1 Entry
Contract V6.1 check set (identity/split, image/E1, Stage0/E2, text, embedding,
Clinical Graph, P0-B concept, prototype, semantic soft-label V2, P0-A loss
policy, final exclusion) while consuming the Work456 frozen contracts and
runtime interfaces directly.  It never re-materializes or re-derives semantic
truth, and it never writes into the immutable bundle root.

Hotfix 1 changes:

* ``validate_p0b_semantic_asset_contract`` performs real
  ``sha256_file(path) == declared_sha256`` checks (not just format checks).
* final exclusion proof distinguishes case-scoped rejection ids from
  image-scoped ids.
* the bundle SHA manifest is recomputed per file, and the manifest /
  mapping / inventory / bundle hashes are all closed against the receipt.
* the runtime manifest is the V6 manifest (``manifest_stage1_v6.csv``) that
  the formal launcher consumes; identity checks use the V6 ``split`` column.

Hotfix 2 control-plane closure:

* the final validator is unified with the actual V6 Entry validator: it
  resolves the complete ``V6_STANDARD_FILENAMES`` runtime bundle through the
  bundle-bound V6 asset index (real per-asset SHA256 re-verification), calls
  ``validate_stage1_v6_bundle`` on the final root itself, and never announces
  ``PASS_FORMAL`` unless the V6 Entry bundle validation is also
  ``PASS_FORMAL``;
* the bundle SHA closure now covers the V6 asset index (the carrier of the V6
  asset references/hashes consumed by the runtime and Entry Contract) together
  with manifest / mapping / inventory, with no circular hash;
* final exclusion verification is an explicit set intersection: the
  materializer collects rejected/discarded ids unconditionally, so any
  overlap with the manifest is a hard contamination failure.

Hotfix 3 (micro):

* the bundle SHA closure moved into the single ``_bundle_closure_sha``
  authority (materializer-frozen algorithm); the new public
  ``verify_bundle_self_integrity`` wrapper recomputes the real closure from
  the on-disk files, fail-closed, and is consumed by the launcher
  authorization guard (no second SHA algorithm anywhere).

A tiny synthetic final-bundle fixture is permitted for tests only.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from breast_pretrain.data.stage1_fullpool_binding_schema import (
    canonical_json_bytes,
    is_sha256,
    sha256_bytes,
    sha256_file,
)
from breast_pretrain.data.stage1_fullpool_final_materializer import (
    BUNDLE_INVENTORY_FILENAME,
    BUNDLE_SHA_MANIFEST_FILENAME,
    FINAL_MANIFEST_COLUMNS,
    FINAL_MANIFEST_FILENAME,
    MATERIALIZATION_RECEIPT_FILENAME,
    ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME,
    V6_ASSET_INDEX_FILENAME,
    V6_ASSET_INDEX_SCHEMA_VERSION,
)
from breast_pretrain.data.stage1_v6_contract import (
    V6_STANDARD_FILENAMES,
    _ACCEPTED as _V6_ACCEPTED_ROUTES,
    validate_stage1_v6_bundle,
)

FINAL_VALIDATION_SCHEMA_VERSION = "stage1_final_bundle_validation_v1"
FINAL_VALIDATOR_VERSION = "work7_final_validator_v1"

# Unification with the V6 Entry contract: the final accepted-route authority
# is exactly the V6 accepted set (AUTOMATIC_ACCEPTED alone is not accepted by
# the runtime Entry Contract and must not pass the Work7 control plane).
_ACCEPTED_REPORT_ROUTES = _V6_ACCEPTED_ROUTES

_TEXT_SOURCE_TYPES = frozenset({"real", "generated"})
_GAZE_QUALITY = "usable"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool_flag(value: Any) -> bool:
    return value is True or _text(value).lower() in {"1", "true", "yes"}


def _append(errors: list[str], message: str) -> None:
    if len(errors) < 300:
        errors.append(message)


def _require_file(path: Path, label: str, errors: list[str]) -> bool:
    if not path.is_file():
        _append(errors, f"missing required {label}: {path}")
        return False
    return True


def _read_manifest(root: Path, errors: list[str]) -> list[dict[str, str]]:
    path = root / FINAL_MANIFEST_FILENAME
    if not _require_file(path, "final runtime manifest (manifest_stage1_v6.csv)", errors):
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    if missing := sorted(set(FINAL_MANIFEST_COLUMNS) - columns):
        _append(errors, "final runtime manifest misses columns: " + ", ".join(missing))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON authority must be an object: {path}")
    return value


def _verify_receipt_self_sha(receipt: Mapping[str, Any], sha_field: str, errors: list[str]) -> None:
    expected = _text(receipt.get(sha_field))
    recomputed = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != sha_field})
    )
    if expected and expected != recomputed:
        _append(errors, f"{sha_field} does not match the receipt contents.")


def _bundle_closure_sha(
    manifest_sha: str,
    mapping_sha: str,
    inventory_sha: str,
    v6_asset_index_sha: str,
) -> str:
    """Materializer-frozen bundle SHA closure (single authoritative algorithm).

    ``bundle_sha256 = sha256(canonical{manifest, mapping, inventory,
    v6_asset_index})``.  The V6 asset index carries the V6 asset
    references/hashes consumed by the runtime and Entry Contract; including
    its own file hash closes the bundle over them without circularity.  Every
    consumer (validator, launcher guard) must reuse this exact function --
    never a second SHA algorithm.
    """
    return sha256_bytes(
        canonical_json_bytes(
            {
                "manifest": manifest_sha,
                "mapping": mapping_sha,
                "inventory": inventory_sha,
                "v6_asset_index": v6_asset_index_sha,
            }
        )
    )


def _validate_bundle_self_integrity(root: Path, errors: list[str]) -> dict[str, Any]:
    receipt_path = root / MATERIALIZATION_RECEIPT_FILENAME
    inventory_path = root / BUNDLE_INVENTORY_FILENAME
    sha_manifest_path = root / BUNDLE_SHA_MANIFEST_FILENAME
    mapping_path = root / ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME
    index_path = root / V6_ASSET_INDEX_FILENAME
    if not _require_file(receipt_path, "materialization receipt", errors):
        return {}
    if not _require_file(inventory_path, "bundle inventory", errors):
        return {}
    receipt = _read_json(receipt_path)
    inventory = _read_json(inventory_path)
    _verify_receipt_self_sha(receipt, "materialization_receipt_sha256", errors)
    if receipt.get("bundle_schema_version") != "stage1_final_bundle_v1":
        _append(errors, "final bundle receipt has unsupported bundle_schema_version.")
    for label, path, field in (
        ("manifest", root / FINAL_MANIFEST_FILENAME, "manifest_sha256"),
        ("mapping", mapping_path, "mapping_sha256"),
        ("inventory", inventory_path, "inventory_sha256"),
        ("sha manifest", sha_manifest_path, "sha_manifest_sha256"),
        ("v6 asset index", index_path, "v6_asset_index_sha256"),
    ):
        if _require_file(path, f"bundle {label}", errors):
            actual = sha256_file(str(path))
            if receipt.get(field) != actual:
                _append(errors, f"final bundle receipt {field} does not match the {label} file.")
    # Bundle SHA closure: bundle_sha256 = sha256(canonical{manifest, mapping,
    # inventory, v6_asset_index}).  The V6 asset index carries the V6 asset
    # references/hashes consumed by the runtime and Entry Contract; including
    # its own file hash here closes the bundle over them without circularity.
    recomputed_bundle_sha = _bundle_closure_sha(
        receipt.get("manifest_sha256", ""),
        receipt.get("mapping_sha256", ""),
        receipt.get("inventory_sha256", ""),
        receipt.get("v6_asset_index_sha256", ""),
    )
    if receipt.get("bundle_sha256") != recomputed_bundle_sha:
        _append(
            errors,
            "final bundle receipt bundle_sha256 does not close over manifest/mapping/inventory/v6_asset_index.",
        )
    # Recompute every file in the bundle SHA manifest.  The manifest covers
    # exactly the immutable bundle-local data files (manifest / mapping /
    # inventory / v6 asset index) and deliberately excludes itself and the
    # receipt (no circular dependency).
    if _require_file(sha_manifest_path, "bundle sha manifest", errors):
        expected_names = {
            FINAL_MANIFEST_FILENAME,
            ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME,
            BUNDLE_INVENTORY_FILENAME,
            V6_ASSET_INDEX_FILENAME,
        }
        listed: dict[str, str] = {}
        for line in sha_manifest_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) != 2 or not is_sha256(parts[0]) or not parts[1].strip():
                _append(errors, "bundle sha manifest contains an invalid line.")
                continue
            declared_sha, name = parts[0], parts[1].strip()
            if name in listed:
                _append(errors, f"bundle sha manifest duplicates file entry: {name}")
            listed[name] = declared_sha
            file_path = root / name
            if not _require_file(file_path, f"bundle sha manifest file {name}", errors):
                continue
            actual = sha256_file(str(file_path))
            if actual != declared_sha:
                _append(errors, f"bundle sha manifest mismatch for {name}: declared={declared_sha}, actual={actual}.")
        if set(listed) != expected_names:
            missing = sorted(expected_names - set(listed))
            extra = sorted(set(listed) - expected_names)
            _append(
                errors,
                "bundle sha manifest must cover exactly the immutable data files; "
                f"missing={missing}, extra={extra}.",
            )
    return {"receipt": receipt, "inventory": inventory}


def verify_bundle_self_integrity(
    bundle_root: str | Path,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Public fail-closed wrapper around the bundle self-integrity checks.

    Hotfix 3: the launcher authorization guard must not trust receipt fields
    alone -- it recomputes the real bundle SHA closure from the on-disk files
    with this exact validator helper (single SHA algorithm, never a second
    copy).  Runs the same checks as ``validate_final_bundle`` (receipt
    self-hash, per-file SHA256 of manifest / mapping / inventory / v6 asset
    index against both the receipt fields and the bundle SHA manifest, and the
    materializer-frozen closure) and returns ``actual_bundle_sha256`` computed
    from the real on-disk file hashes.  Status is ``PASS`` only when every
    check passes; any exception raised by the checks is converted into a FAIL
    (the guard never crashes on a malformed bundle).
    """
    root = Path(bundle_root).expanduser().resolve()
    error_list = list(errors) if errors is not None else []
    try:
        result = _validate_bundle_self_integrity(root, error_list)
    except Exception as exc:  # noqa: BLE001 - fail closed, never crash the guard
        error_list.append(f"bundle self-integrity raised {type(exc).__name__}: {exc}")
        result = {}
    actual_bundle_sha256 = ""
    data_files = (
        root / FINAL_MANIFEST_FILENAME,
        root / ROW_IMAGE_CASE_REPORT_MAPPING_FILENAME,
        root / BUNDLE_INVENTORY_FILENAME,
        root / V6_ASSET_INDEX_FILENAME,
    )
    if all(path.is_file() for path in data_files):
        actual_bundle_sha256 = _bundle_closure_sha(*(sha256_file(str(path)) for path in data_files))
    return {
        "status": "PASS" if not error_list else "FAIL",
        "errors": error_list,
        "receipt": result.get("receipt") or {},
        "inventory": result.get("inventory") or {},
        "actual_bundle_sha256": actual_bundle_sha256,
    }


# ---------------------------------------------------------------------------
# A. Identity / split
# ---------------------------------------------------------------------------


def _validate_identity_split(rows: list[Mapping[str, Any]], errors: list[str]) -> dict[str, int]:
    image_ids: set[str] = set()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_study_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    mammography_laterality_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_keys: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        image_id = _text(row.get("image_id"))
        if image_id in image_ids:
            _append(errors, f"final manifest row {index}: duplicate image_id={image_id!r}.")
        image_ids.add(image_id)
        patient_id = _text(row.get("patient_id"))
        split = _text(row.get("split"))
        if patient_id and split:
            patient_splits[patient_id].add(split)
            study_id = _text(row.get("study_id"))
            if study_id:
                patient_study_splits[(patient_id, study_id)].add(split)
            laterality = _text(row.get("laterality"))
            modality = _text(row.get("modality")).lower()
            if laterality and modality in {"mammo", "mammography", "ffdm", "dbt", "cesm"}:
                mammography_laterality_splits[(patient_id, laterality)].add(split)
        case_key = tuple(_text(row.get(field)) for field in ("dataset_id", "case_id", "report_unit_id"))
        if all(case_key):
            case_keys[case_key].add(_text(row.get("report_id")))
            case_keys[case_key].add(split)
    for patient_id, splits in sorted(patient_splits.items()):
        if len(splits) > 1:
            _append(errors, f"patient_id={patient_id!r} crosses splits: {sorted(splits)}.")
    for key, splits in sorted(patient_study_splits.items()):
        if len(splits) > 1:
            _append(errors, f"patient_id+study_id={key!r} crosses splits: {sorted(splits)}.")
    for key, splits in sorted(mammography_laterality_splits.items()):
        if len(splits) > 1:
            _append(errors, f"mammography patient_id+laterality={key!r} crosses splits: {sorted(splits)}.")
    for key, values in sorted(case_keys.items()):
        if len(values) > 2:
            _append(errors, f"case/report group={key!r} has conflicting report/split membership: {sorted(values)}.")
    return {"image_count": len(image_ids)}


# ---------------------------------------------------------------------------
# B. Image / E1  C. Stage0 / E2
# ---------------------------------------------------------------------------


def _validate_image_e1_stage0_gaze(
    rows: list[Mapping[str, Any]], inventory: Mapping[str, Any], errors: list[str]
) -> dict[str, Any]:
    patch_token_orders: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not _bool_flag(row.get("stage0_usable")):
            _append(errors, f"final manifest row {index}: stage0_usable must be 1 in the formal bundle.")
        if not _bool_flag(row.get("gaze_prior_available")):
            _append(errors, f"final manifest row {index}: gaze_prior_available must be 1.")
        if _text(row.get("gaze_prior_quality")) != _GAZE_QUALITY:
            _append(errors, f"final manifest row {index}: gaze_prior_quality must be {_GAZE_QUALITY!r}.")
        if not _bool_flag(row.get("gaze_training_enabled")):
            _append(errors, f"final manifest row {index}: gaze_training_enabled must be 1.")
        token_order = _text(row.get("patch_token_order_version"))
        if not token_order:
            _append(errors, f"final manifest row {index}: patch_token_order_version is required.")
        patch_token_orders.add(token_order)
        for field in ("source_image_sha256", "patch_geometry_key", "valid_content_mask_key", "gaze_prior_key", "gaze_to_patch_projection_key"):
            if not _text(row.get(field)):
                _append(errors, f"final manifest row {index}: {field} is required.")
    e1_audit = inventory.get("e1_audit") if isinstance(inventory, Mapping) else None
    if isinstance(e1_audit, Mapping) and _text(e1_audit.get("status")) != "PASS":
        _append(errors, "E1 image/patch authority audit is not PASS.")
    stage0_audit = inventory.get("stage0_audit") if isinstance(inventory, Mapping) else None
    if isinstance(stage0_audit, Mapping) and _text(stage0_audit.get("status")) != "PASS":
        _append(errors, "Stage0/E2 authority audit is not PASS.")
    if len(patch_token_orders) > 1:
        _append(errors, "final manifest mixes patch_token_order_version values: " + ", ".join(sorted(patch_token_orders)))
    return {"patch_token_order_count": len(patch_token_orders)}


# ---------------------------------------------------------------------------
# D. Text  E. Embedding  F. Clinical Graph
# ---------------------------------------------------------------------------


def _validate_text_embedding_graph(
    rows: list[Mapping[str, Any]], errors: list[str]
) -> dict[str, Any]:
    hashes_by_image: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        route = _text(row.get("final_case_route"))
        if route not in _ACCEPTED_REPORT_ROUTES:
            _append(errors, f"final manifest row {index}: final_case_route={route!r} is not final accepted.")
        source_type = _text(row.get("text_source_type"))
        if source_type not in _TEXT_SOURCE_TYPES:
            _append(errors, f"final manifest row {index}: text_source_type={source_type!r}; Structured Prompt is forbidden.")
        report_hash = _text(row.get("effective_report_sha256"))
        if not is_sha256(report_hash):
            _append(errors, f"final manifest row {index}: effective_report_sha256 must be a lowercase SHA256.")
        if report_hash in hashes_by_image and hashes_by_image[report_hash] != report_hash:
            _append(errors, f"final manifest row {index}: effective_report_sha256 lineage is inconsistent.")
        hashes_by_image[row.get("image_id", "")] = report_hash
        for field in ("text_prompt_key", "embedding_key", "graph_key", "concept_schema_version"):
            if not _text(row.get(field)):
                _append(errors, f"final manifest row {index}: {field} is required.")
        for field in ("visual_training_enabled", "text_semantic_enabled", "clinical_graph_enabled", "concept_target_enabled", "semantic_soft_label_enabled"):
            if not _bool_flag(row.get(field)):
                _append(errors, f"final manifest row {index}: required formal flag {field} must be 1.")
    return {"hash_lineage_rows": len(hashes_by_image)}


# ---------------------------------------------------------------------------
# G/H/I. Work456 frozen interface consumption
# ---------------------------------------------------------------------------


def _verify_asset_sha(path_value: Any, declared_sha: Any, label: str, errors: list[str]) -> bool:
    """Real file-content SHA256 verification: sha256_file(path) == declared."""
    if not isinstance(path_value, str) or not path_value.strip():
        _append(errors, f"{label} path is required.")
        return False
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        _append(errors, f"{label} file is missing: {path}")
        return False
    declared = _text(declared_sha)
    if not is_sha256(declared):
        _append(errors, f"{label} declared sha256 is malformed.")
        return False
    actual = sha256_file(str(path))
    if actual != declared:
        _append(errors, f"{label} SHA256 mismatch: declared={declared}, actual file={actual}.")
        return False
    return True


def validate_p0b_semantic_asset_contract(
    bundle_root: Path,
    receipt: Mapping[str, Any],
    errors: list[str],
    *,
    loaders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consume the Work456 frozen contracts without re-materializing assets.

    Every declared receipt/contract authority is verified with a real
    ``sha256_file(path) == declared_sha256`` check.  ``loaders`` is a
    dependency-injection seam for tests: each key is a callable
    ``(asset_path, expected_sha256) -> summary dict`` that internally uses the
    frozen Work456 runtime interface.
    """
    loaders = loaders or {}
    concept_loader = loaders.get("concept_runtime")
    prototype_loader = loaders.get("prototype")
    soft_label_loader = loaders.get("soft_label_v2")

    semantic_receipts = receipt.get("semantic_receipts")
    if not isinstance(semantic_receipts, Mapping):
        _append(errors, "final bundle receipt must record semantic_receipts.")
        return {"status": "BLOCKED_PENDING_ASSETS"}

    summary: dict[str, Any] = {}

    def _dispatch(loader, path_value, sha_value, key, label) -> None:
        if loader is not None:
            summary[key] = loader(path_value, sha_value) if path_value else {
                "status": "MISSING", "reason": f"{label} path is not recorded."
            }
            return
        if _verify_asset_sha(path_value, sha_value, label, errors):
            summary[key] = {"status": "RESOLVABLE", "sha256": _text(sha_value)}
        else:
            summary[key] = {"status": "FAIL"}

    _dispatch(
        concept_loader,
        semantic_receipts.get("concept_target_receipt_path"),
        semantic_receipts.get("concept_target_receipt_sha256"),
        "concept_runtime",
        "concept target receipt",
    )
    _dispatch(
        prototype_loader,
        semantic_receipts.get("prototype_receipt_path"),
        semantic_receipts.get("prototype_receipt_sha256"),
        "prototype",
        "prototype receipt",
    )
    _dispatch(
        soft_label_loader,
        semantic_receipts.get("soft_label_v2_receipt_path"),
        semantic_receipts.get("soft_label_v2_receipt_sha256"),
        "soft_label_v2",
        "semantic soft-label V2 receipt",
    )
    _dispatch(
        None,
        semantic_receipts.get("concept_schema_path"),
        semantic_receipts.get("concept_schema_sha256"),
        "concept_schema",
        "frozen concept schema",
    )
    _dispatch(
        None,
        semantic_receipts.get("semantic_contract_path"),
        semantic_receipts.get("semantic_contract_sha256"),
        "semantic_contract",
        "frozen semantic contract",
    )
    return summary


# ---------------------------------------------------------------------------
# J. P0-A / loss policy
# ---------------------------------------------------------------------------


def validate_loss_policy_contract(resolved_config: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    """Hard-check the frozen loss policy from the resolved formal config.

    L_global and L_graph must be static; omega_sem may scale exactly
    L_visible / L_soft / L_concept / L_cc and nothing else.
    """
    losses = resolved_config.get("losses") if isinstance(resolved_config.get("losses"), Mapping) else {}
    conflict_aware = losses.get("conflict_aware")
    conflict_aware = conflict_aware if isinstance(conflict_aware, Mapping) else {}
    allow_dynamic_graph = _bool_flag(conflict_aware.get("allow_dynamic_graph_consistency_weighting"))
    if allow_dynamic_graph:
        _append(errors, "loss policy: allow_dynamic_graph_consistency_weighting must be false (L_graph static).")
    dynamic_keys = conflict_aware.get("dynamic_omega_sem_targets")
    if dynamic_keys is not None:
        if not isinstance(dynamic_keys, list):
            _append(errors, "loss policy: dynamic_omega_sem_targets must be a list.")
        else:
            normalized = {_text(key) for key in dynamic_keys}
            expected = {"visible_align", "semantic_soft", "concept_loss", "concept_consistency"}
            if normalized != expected:
                _append(
                    errors,
                    "loss policy: omega_sem target set must be exactly "
                    "{visible_align, semantic_soft, concept_loss, concept_consistency}; "
                    f"got {sorted(normalized)}.",
                )
    masking = resolved_config.get("masking") if isinstance(resolved_config.get("masking"), Mapping) else {}
    try:
        floor = float(masking.get("min_visible_salient_fraction"))
    except (TypeError, ValueError):
        _append(errors, "loss policy: masking.min_visible_salient_fraction must be a finite number in (0, 1].")
    else:
        if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
            _append(errors, "loss policy: masking.min_visible_salient_fraction must be a finite number in (0, 1].")
    p0a = resolved_config.get("p0a_contract") if isinstance(resolved_config.get("p0a_contract"), Mapping) else {}
    if p0a and not _bool_flag(p0a.get("visible_salient_floor_enforced")):
        _append(errors, "loss policy: p0a_contract.visible_salient_floor_enforced must be true.")
    return {"loss_policy": "static_global_static_graph" if not errors else "VIOLATION"}


# ---------------------------------------------------------------------------
# K. Final exclusion
# ---------------------------------------------------------------------------


def validate_final_exclusion(
    rows: list[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    errors: list[str],
) -> dict[str, int]:
    exclusion = inventory.get("final_exclusion") if isinstance(inventory, Mapping) else None
    if isinstance(exclusion, Mapping):
        manifest_case_ids = {_text(row.get("case_id")) for row in rows}
        manifest_image_ids = {_text(row.get("image_id")) for row in rows}
        case_scoped = ("final_rejected_case_ids",)
        image_scoped = ("discarded_image_ids", "stage0_review_ids", "stage0_reject_ids", "no_gaze_ids", "gaze_disabled_ids")
        for key in case_scoped + image_scoped:
            excluded_ids = exclusion.get(key)
            if not isinstance(excluded_ids, list):
                _append(errors, f"final bundle inventory final_exclusion.{key} must be a list.")
                continue
            universe = manifest_case_ids if key in case_scoped else manifest_image_ids
            overlap = sorted(set(_text(item) for item in excluded_ids) & universe)
            if overlap:
                _append(errors, f"final manifest intersects {key}: {overlap[:20]}.")
    else:
        _append(errors, "final bundle inventory must record final_exclusion zero-intersection proof.")
    return {"final_exclusion_rows": len(rows)}


# ---------------------------------------------------------------------------
# L. Actual V6 Entry bundle validation (unified with the launcher gate)
# ---------------------------------------------------------------------------

_V6_SHARED_FIELDS = (
    "image_id",
    "case_id",
    "report_unit_id",
    "report_id",
    "split",
    "dataset_id",
    "modality",
    "effective_report_sha256",
)


def _validate_v6_entry_contract(
    root: Path,
    receipt: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    resolved_config_path: str | Path | None,
    errors: list[str],
) -> dict[str, Any]:
    """Resolve the complete V6 runtime bundle and run the actual V6 Entry validator.

    The final root must expose every ``V6_STANDARD_FILENAMES`` asset through
    the bundle-bound V6 asset index (symlinks resolved from the frozen upstream
    V6 root).  Each indexed asset is re-verified with a real file SHA256, then
    ``validate_stage1_v6_bundle`` -- the exact validator the formal launcher
    gate consumes -- must return ``PASS_FORMAL`` on the final root itself.
    """
    v6_root_value = _text(receipt.get("v6_bundle_root"))
    index_sha = _text(receipt.get("v6_asset_index_sha256"))
    if not v6_root_value or not index_sha:
        _append(errors, "V6 entry: final bundle receipt must record v6_bundle_root and v6_asset_index_sha256.")
        return {"status": "BLOCKED"}
    v6_root = Path(v6_root_value).expanduser().resolve()
    index_path = root / V6_ASSET_INDEX_FILENAME
    if not _require_file(index_path, "V6 asset index", errors):
        return {"status": "BLOCKED"}
    actual_index_sha = sha256_file(str(index_path))
    if actual_index_sha != index_sha:
        _append(
            errors,
            f"V6 entry: V6 asset index SHA256 mismatch: declared={index_sha}, actual file={actual_index_sha}.",
        )
    index = _read_json(index_path)
    if index.get("index_schema_version") != V6_ASSET_INDEX_SCHEMA_VERSION:
        _append(errors, "V6 entry: V6 asset index has an unsupported schema version.")
    if _text(index.get("v6_bundle_root")) != str(v6_root):
        _append(errors, "V6 entry: V6 asset index v6_bundle_root disagrees with the bundle receipt.")
    assets = index.get("assets")
    expected_keys = set(V6_STANDARD_FILENAMES) - {"manifest"}
    if not isinstance(assets, Mapping) or set(assets) != expected_keys:
        _append(
            errors,
            "V6 entry: V6 asset index must cover exactly the standard V6 assets except the projected manifest; "
            f"missing={sorted(expected_keys - set(assets or {}))}, extra={sorted(set(assets or {}) - expected_keys)}.",
        )
        assets = assets if isinstance(assets, Mapping) else {}
    for key in sorted(expected_keys):
        asset = assets.get(key)
        expected_filename = V6_STANDARD_FILENAMES[key]
        if not isinstance(asset, Mapping):
            _append(errors, f"V6 entry: V6 asset index misses entry {key}.")
            continue
        if _text(asset.get("filename")) != expected_filename:
            _append(errors, f"V6 entry: V6 asset index filename for {key} must be {expected_filename}.")
            continue
        source = Path(str(asset.get("source_path") or "")).expanduser().resolve()
        declared_sha = _text(asset.get("sha256"))
        if not source.is_file():
            _append(errors, f"V6 entry: indexed V6 asset {key} source file is missing: {source}")
            continue
        if not is_sha256(declared_sha) or sha256_file(str(source)) != declared_sha:
            _append(
                errors,
                f"V6 entry: indexed V6 asset {key} SHA256 mismatch (declared={declared_sha}, "
                f"actual file={sha256_file(str(source))}).",
            )
        bundle_local = root / expected_filename
        if not bundle_local.is_file():
            _append(errors, f"V6 entry: final root lacks V6 asset {expected_filename} (index entry {key}).")
            continue
        if sha256_file(str(bundle_local)) != declared_sha:
            _append(
                errors,
                f"V6 entry: final-root V6 asset {expected_filename} content does not match the indexed SHA256.",
            )
    referenced = index.get("referenced_assets")
    if not isinstance(referenced, list):
        _append(errors, "V6 entry: V6 asset index must record referenced_assets.")
    else:
        seen_referenced: set[str] = set()
        for item in referenced:
            if not isinstance(item, Mapping):
                _append(errors, "V6 entry: V6 asset index referenced_assets entries must be objects.")
                continue
            relative = _text(item.get("filename"))
            source = Path(str(item.get("source_path") or "")).expanduser().resolve()
            declared_sha = _text(item.get("sha256"))
            if not relative or relative in seen_referenced:
                _append(errors, f"V6 entry: V6 asset index has an invalid/duplicate referenced asset {relative!r}.")
                continue
            seen_referenced.add(relative)
            bundle_local = root / relative
            if not source.is_file():
                _append(errors, f"V6 entry: referenced V6 asset source is missing: {source}")
                continue
            if not is_sha256(declared_sha) or sha256_file(str(source)) != declared_sha:
                _append(errors, f"V6 entry: referenced V6 asset {relative} SHA256 mismatch.")
            if not bundle_local.is_file() or sha256_file(str(bundle_local)) != declared_sha:
                _append(
                    errors,
                    f"V6 entry: final-root referenced V6 asset {relative} does not match the indexed SHA256.",
                )
    # The projected manifest must be a deterministic projection of the frozen
    # V6 manifest: same image set, same shared identity/hash fields.
    v6_manifest_path = v6_root / FINAL_MANIFEST_FILENAME
    if _require_file(v6_manifest_path, "frozen V6 bundle manifest", errors):
        with v6_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            v6_rows = list(csv.DictReader(handle))
        v6_by_image = {_text(row.get("image_id")): row for row in v6_rows}
        final_by_image = {_text(row.get("image_id")): row for row in rows}
        if set(v6_by_image) != set(final_by_image):
            _append(
                errors,
                "V6 entry: final manifest image set differs from the frozen V6 bundle manifest "
                f"(final={len(final_by_image)}, v6={len(v6_by_image)}).",
            )
        for image_id in sorted(set(v6_by_image) & set(final_by_image)):
            v6_row = v6_by_image[image_id]
            final_row = final_by_image[image_id]
            for field in _V6_SHARED_FIELDS:
                if _text(v6_row.get(field)) != _text(final_row.get(field)):
                    _append(
                        errors,
                        f"V6 entry: final manifest row image_id={image_id!r} disagrees with the frozen "
                        f"V6 manifest on {field} ({_text(final_row.get(field))!r} != {_text(v6_row.get(field))!r}).",
                    )
    # The actual V6 Entry bundle validation on the final root itself.  The V6
    # contract may raise (e.g. KeyError) on a malformed/mismatched bundle
    # instead of returning errors; that must fail closed, never crash.
    try:
        v6_result = validate_stage1_v6_bundle(root, resolved_config_path)
    except Exception as exc:  # noqa: BLE001 - the V6 contract must fail closed
        v6_result = {"status": "BLOCKED_RAISED", "errors": [f"validate_stage1_v6_bundle raised: {exc!r}"]}
    if v6_result.get("status") != "PASS_FORMAL":
        _append(
            errors,
            "V6 entry: validate_stage1_v6_bundle is not PASS_FORMAL: "
            + " | ".join(v6_result.get("errors", [])[:6]),
        )
    return {
        "status": v6_result.get("status"),
        "v6_bundle_root": str(v6_root),
        "asset_count": len(assets),
        "referenced_asset_count": len(referenced) if isinstance(referenced, list) else 0,
        "v6_result": v6_result,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_final_bundle(
    bundle_root: str | Path,
    *,
    resolved_config_path: str | Path | None = None,
    loaders: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a materialized final bundle; returns a formal validation receipt.

    The validator never mutates the bundle.  Status is ``PASS_FORMAL`` only
    when every hard gate passes.
    """
    root = Path(bundle_root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    sections: dict[str, Any] = {}

    integrity = _validate_bundle_self_integrity(root, errors)
    receipt = integrity.get("receipt") or {}
    inventory = integrity.get("inventory") or {}
    rows = _read_manifest(root, errors)
    if rows:
        sections["identity_split"] = _validate_identity_split(rows, errors)
        sections["image_e1_stage0_gaze"] = _validate_image_e1_stage0_gaze(rows, inventory, errors)
        sections["text_embedding_graph"] = _validate_text_embedding_graph(rows, errors)
        sections["p0b_semantic_assets"] = validate_p0b_semantic_asset_contract(root, receipt, errors, loaders=loaders)
        sections["final_exclusion"] = validate_final_exclusion(rows, inventory, errors)
        sections["v6_entry"] = _validate_v6_entry_contract(root, receipt, rows, resolved_config_path, errors)
    else:
        _append(errors, "final runtime manifest is empty; a formal bundle cannot be validated.")

    if resolved_config_path is not None:
        config_path = Path(resolved_config_path).expanduser().resolve()
        if not _require_file(config_path, "resolved formal config", errors):
            resolved_config: dict[str, Any] = {}
        else:
            try:
                value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                _append(errors, f"resolved formal config YAML is invalid: {exc}")
                value = {}
            resolved_config = value if isinstance(value, Mapping) else {}
        sections["loss_policy"] = validate_loss_policy_contract(resolved_config, errors)

    if not errors:
        status = "PASS_FORMAL"
    elif any("crosses splits" in error or "conflicting report" in error for error in errors):
        status = "BLOCKED_SPLIT_LEAKAGE"
    elif any("intersects" in error for error in errors):
        status = "BLOCKED_FINAL_REJECT_CONTAMINATION"
    elif any("loss policy" in error for error in errors):
        status = "BLOCKED_LOSS_POLICY_MISMATCH"
    elif any("sha256" in error or "does not match" in error or "mismatch" in error for error in errors):
        status = "BLOCKED_HASH_MISMATCH"
    elif any(error.startswith("V6 ") for error in errors):
        status = "BLOCKED_V6_ENTRY"
    elif any("formal P0-B" in error or "concept schema" in error or "prototype" in error or "soft-label" in error for error in errors):
        status = "BLOCKED_CONCEPT_SCHEMA"
    elif any("missing required" in error or "must be 1" in error or "must be usable" in error or "forbidden" in error for error in errors):
        status = "BLOCKED_CONTRACT_VIOLATION"
    else:
        status = "BLOCKED_PENDING_ASSETS"

    return {
        "validation_schema_version": FINAL_VALIDATION_SCHEMA_VERSION,
        "validator_version": FINAL_VALIDATOR_VERSION,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "sections": sections,
        "bundle_root": str(root),
        "bundle_sha256": receipt.get("bundle_sha256"),
        "manifest_row_count": len(rows),
    }


__all__ = [
    "FINAL_VALIDATION_SCHEMA_VERSION",
    "FINAL_VALIDATOR_VERSION",
    "validate_final_bundle",
    "validate_loss_policy_contract",
    "validate_p0b_semantic_asset_contract",
    "verify_bundle_self_integrity",
]
