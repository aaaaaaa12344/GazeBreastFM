from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BiradsPriorRecord:
    image_id: str
    schema_version: str
    birads: str
    density: str
    view: str
    laterality: str
    finding: str

    def has_any_known_concept(self) -> bool:
        return any(
            value not in {"", "unknown"}
            for value in (
                self.birads,
                self.density,
                self.view,
                self.laterality,
                self.finding,
            )
        )


class Stage1BiradsPriorIndex:
    EXPECTED_SCHEMA_VERSION = "stage1_birads_prior_v1"

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Stage 1 BI-RADS prior manifest does not exist: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.prior_path_by_image_id: dict[str, Path] = {}
        manifest_dir = self.manifest_path.parent
        for row in rows:
            image_id = str(row.get("image_id") or "").strip()
            prior_path_value = str(row.get("prior_path") or "").strip()
            if not image_id or not prior_path_value:
                continue
            prior_path = Path(prior_path_value).expanduser()
            if not prior_path.is_absolute():
                prior_path = (manifest_dir / prior_path).resolve()
            self.prior_path_by_image_id[image_id] = prior_path

    def _load_record(self, image_id: str, prior_path: Path) -> BiradsPriorRecord:
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        schema_version = str(payload.get("schema_version") or "").strip()
        if schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported BI-RADS prior schema_version for {image_id}: {schema_version!r}"
            )
        concepts = payload.get("concepts", {})
        if not isinstance(concepts, dict):
            raise ValueError(f"concepts must be a mapping in {prior_path}")
        return BiradsPriorRecord(
            image_id=image_id,
            schema_version=schema_version,
            birads=str(concepts.get("birads") or "unknown").strip() or "unknown",
            density=str(concepts.get("density") or "unknown").strip() or "unknown",
            view=str(concepts.get("view") or "unknown").strip() or "unknown",
            laterality=str(concepts.get("laterality") or "unknown").strip() or "unknown",
            finding=str(concepts.get("finding") or "unknown").strip() or "unknown",
        )

    def resolve_batch(
        self,
        image_ids: list[str],
    ) -> tuple[list[BiradsPriorRecord], list[str]]:
        records: list[BiradsPriorRecord] = []
        warnings: list[str] = []
        for image_id in image_ids:
            prior_path = self.prior_path_by_image_id.get(str(image_id).strip())
            if prior_path is None:
                warnings.append(f"birads_prior_missing:{image_id}")
                records.append(
                    BiradsPriorRecord(
                        image_id=str(image_id),
                        schema_version=self.EXPECTED_SCHEMA_VERSION,
                        birads="unknown",
                        density="unknown",
                        view="unknown",
                        laterality="unknown",
                        finding="unknown",
                    )
                )
                continue
            if not prior_path.exists():
                warnings.append(f"birads_prior_file_missing:{image_id}:{prior_path}")
                records.append(
                    BiradsPriorRecord(
                        image_id=str(image_id),
                        schema_version=self.EXPECTED_SCHEMA_VERSION,
                        birads="unknown",
                        density="unknown",
                        view="unknown",
                        laterality="unknown",
                        finding="unknown",
                    )
                )
                continue
            records.append(self._load_record(str(image_id), prior_path))
        return records, warnings
