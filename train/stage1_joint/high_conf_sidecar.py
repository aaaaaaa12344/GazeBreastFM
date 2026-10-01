from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch


class HighConfSidecarLoader:
    """Loads and validates pre-materialized high-confidence patch priors.

    On construction:
      1. Loads high_conf_patch_priors.npz
      2. Loads high_conf_patch_prior_manifest.csv
      3. Validates 519/519 alignment: every manifest entry must have a matching NPZ array
      4. Validates checksums: NPZ array checksum must match manifest
      5. Records zero_prior_count for monitoring

    At batch time:
      lookup(image_ids) -> torch.Tensor [B, N] of high-confidence patch priors
    """

    def __init__(
        self,
        npz_path: str | Path,
        manifest_path: str | Path,
        *,
        expected_count: int | None = None,
        expected_training_manifest_path: str | Path | None = None,
        block_on_checksum_mismatch: bool = True,
    ) -> None:
        npz_path = Path(npz_path).expanduser().resolve()
        manifest_path = Path(manifest_path).expanduser().resolve()

        if not npz_path.is_file():
            raise FileNotFoundError(f"High-conf prior NPZ not found: {npz_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"High-conf prior manifest not found: {manifest_path}")

        self._npz_path = npz_path
        self._manifest_path = manifest_path
        self._arrays: dict[str, np.ndarray] = {}
        self._checksums: dict[str, str] = {}
        self._modalities: dict[str, str] = {}
        self._grids: dict[str, tuple[int, int]] = {}
        self._zero_count = 0
        self._total_loaded = 0
        self._failures: list[str] = []
        self._warnings: list[str] = []

        expected_training_ids: set[str] | None = None
        if expected_training_manifest_path is not None:
            expected_training_manifest = Path(expected_training_manifest_path).expanduser().resolve()
            if not expected_training_manifest.is_file():
                raise FileNotFoundError(f"Training manifest not found: {expected_training_manifest}")
            expected_training_ids = set()
            with open(expected_training_manifest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = row.get("image_id", "").strip()
                    if not image_id:
                        self._failures.append("Training manifest row missing image_id")
                    else:
                        expected_training_ids.add(image_id)

        # Load NPZ
        npz = np.load(str(npz_path))
        for key in npz.files:
            arr = npz[key]
            if not np.isfinite(arr).all():
                self._failures.append(f"NPZ array '{key}' contains non-finite values")
                continue
            self._arrays[key] = arr
            self._checksums[key] = hashlib.sha256(arr.tobytes()).hexdigest()

        # Load manifest
        manifest_entries: list[dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_entries.append(row)

        if expected_count is not None and len(manifest_entries) != expected_count:
            self._failures.append(
                f"Manifest has {len(manifest_entries)} entries, expected {expected_count}"
            )

        # Cross-validate NPZ vs manifest
        manifest_ids: set[str] = set()
        for entry in manifest_entries:
            image_id = entry.get("image_id", "").strip()
            if not image_id:
                self._failures.append("Manifest entry missing image_id")
                continue
            manifest_ids.add(image_id)

            status = entry.get("projection_status", "").strip()
            if status != "pass":
                self._failures.append(f"{image_id}: projection_status={status}, not 'pass'")
                continue

            modality = entry.get("modality", "").strip().lower()
            if not modality or modality not in {"mammography", "mri", "ultrasound"}:
                self._failures.append(f"{image_id}: unknown modality '{modality}'")
                continue

            self._modalities[image_id] = modality
            grid_text = entry.get("patch_grid", "").strip().lower()
            try:
                gh, gw = (int(item) for item in grid_text.replace(",", "x").split("x"))
            except Exception:
                self._failures.append(f"{image_id}: invalid patch_grid '{grid_text}'")
                continue
            self._grids[image_id] = (gh, gw)

            manifest_checksum = entry.get("array_checksum", "").strip()
            if not manifest_checksum:
                self._failures.append(f"{image_id}: missing array_checksum in manifest")
                continue
            transform_checksum = entry.get("transform_spec_checksum", "").strip()
            authoritative_transform_checksum = entry.get("authoritative_transform_checksum", "").strip()
            transform_checksum_match = entry.get("transform_checksum_match", "").strip().lower()
            if not transform_checksum:
                self._failures.append(f"{image_id}: missing transform_spec_checksum in manifest")
                continue
            if not authoritative_transform_checksum:
                self._failures.append(f"{image_id}: missing authoritative_transform_checksum in manifest")
                continue
            if transform_checksum != authoritative_transform_checksum or transform_checksum_match != "true":
                self._failures.append(f"{image_id}: transform checksum mismatch or unverified")
                continue

            if image_id not in self._arrays:
                self._failures.append(f"{image_id}: in manifest but not in NPZ")
                continue

            npz_checksum = self._checksums.get(image_id, "")
            if manifest_checksum != npz_checksum:
                self._failures.append(
                    f"{image_id}: checksum mismatch manifest={manifest_checksum[:16]} "
                    f"npz={npz_checksum[:16]}"
                )
                continue

            arr = self._arrays[image_id]
            if arr.ndim != 1:
                self._failures.append(f"{image_id}: array must be 1D, got shape {arr.shape}")
                continue
            if int(arr.shape[0]) != gh * gw:
                self._failures.append(
                    f"{image_id}: array length {arr.shape[0]} does not match patch_grid {gh}x{gw}"
                )
                continue
            if arr.sum() <= 0:
                self._zero_count += 1
            self._total_loaded += 1

        # Check for NPZ arrays not in manifest
        npz_ids = set(self._arrays.keys())
        missing_from_manifest = npz_ids - manifest_ids
        if missing_from_manifest:
            self._failures.append(
                f"{len(missing_from_manifest)} NPZ arrays not in sidecar manifest: "
                f"{sorted(missing_from_manifest)[:5]}"
            )

        if expected_training_ids is not None:
            if expected_count is not None and len(expected_training_ids) != expected_count:
                self._failures.append(
                    f"Training manifest has {len(expected_training_ids)} image_ids, expected {expected_count}"
                )
            missing_sidecar_ids = expected_training_ids - manifest_ids
            extra_sidecar_ids = manifest_ids - expected_training_ids
            if missing_sidecar_ids:
                self._failures.append(
                    f"{len(missing_sidecar_ids)} training image_ids missing from sidecar manifest: "
                    f"{sorted(missing_sidecar_ids)[:5]}"
                )
            if extra_sidecar_ids:
                self._failures.append(
                    f"{len(extra_sidecar_ids)} sidecar image_ids not present in training manifest: "
                    f"{sorted(extra_sidecar_ids)[:5]}"
                )

        # Block on failures
        if self._failures and block_on_checksum_mismatch:
            raise ValueError(
                f"HighConfSidecarLoader: {len(self._failures)} validation failures:\n  "
                + "\n  ".join(self._failures[:20])
            )

    @property
    def total_loaded(self) -> int:
        return self._total_loaded

    @property
    def zero_prior_count(self) -> int:
        return self._zero_count

    @property
    def failures(self) -> list[str]:
        return list(self._failures)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def lookup(
        self,
        image_ids: list[str],
        num_patches: int | None = None,
        *,
        modalities: list[str] | None = None,
        patch_grid: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Look up high-confidence patch priors for a list of image_ids.

        Returns:
            Tensor of shape [len(image_ids), num_patches] with float32 values.
            Missing entries, shape mismatches, modality mismatches, and grid
            mismatches raise. Formal training must not silently zero-fill priors.
        """
        batch_size = len(image_ids)
        if batch_size == 0:
            return torch.zeros(0, max(1, num_patches or 1), dtype=torch.float32)

        if num_patches is None:
            if patch_grid is not None:
                num_patches = int(patch_grid[0] * patch_grid[1])
            else:
                raise ValueError("num_patches or patch_grid is required for strict high-conf lookup.")

        result = torch.zeros(batch_size, num_patches, dtype=torch.float32)
        for i, img_id in enumerate(image_ids):
            if img_id not in self._arrays:
                raise KeyError(f"high_conf_prior_missing:{img_id}")
            arr = self._arrays[img_id]
            if int(arr.shape[0]) != int(num_patches):
                raise ValueError(
                    f"{img_id}: high-conf prior length {arr.shape[0]} != expected {num_patches}"
                )
            if modalities is not None:
                expected_modality = str(modalities[i]).strip().lower()
                if expected_modality == "mammo":
                    expected_modality = "mammography"
                if expected_modality == "us":
                    expected_modality = "ultrasound"
                actual_modality = self._modalities.get(img_id)
                if actual_modality != expected_modality:
                    raise ValueError(
                        f"{img_id}: sidecar modality {actual_modality!r} != batch modality {expected_modality!r}"
                    )
            if patch_grid is not None and self._grids.get(img_id) != tuple(patch_grid):
                raise ValueError(
                    f"{img_id}: sidecar grid {self._grids.get(img_id)} != batch grid {tuple(patch_grid)}"
                )
            result[i] = torch.from_numpy(arr).float()

        return result

    def summary(self) -> dict[str, object]:
        return {
            "npz_path": str(self._npz_path),
            "manifest_path": str(self._manifest_path),
            "total_loaded": self._total_loaded,
            "zero_prior_count": self._zero_count,
            "nonzero_prior_count": self._total_loaded - self._zero_count,
            "failure_count": len(self._failures),
            "warning_count": len(self._warnings),
        }


__all__ = ["HighConfSidecarLoader"]
