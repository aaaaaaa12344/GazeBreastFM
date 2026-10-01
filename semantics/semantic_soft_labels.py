from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SemanticSoftLabelBatch:
    matrix: torch.Tensor
    batch_positions: list[int]
    image_ids: list[str]
    warnings: list[str]

    @property
    def valid_count(self) -> int:
        return len(self.batch_positions)


class SemanticSoftLabelIndex:
    def __init__(
        self,
        matrix_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        *,
        label_format: str = "dense",
        topk_path: str | Path | None = None,
        symmetrization_strategy: str = "max",
    ) -> None:
        self.label_format = str(label_format or "dense").strip().lower()
        self.symmetrization_strategy = str(symmetrization_strategy or "max").strip().lower()
        if self.label_format not in {"dense", "sparse_topk"}:
            raise ValueError(f"Unsupported semantic soft-label format: {label_format!r}")
        if self.symmetrization_strategy != "max":
            raise ValueError("Only semantic sparse symmetrization_strategy='max' is supported.")

        self.matrix_path = Path(matrix_path).expanduser().resolve() if matrix_path is not None else None
        self.topk_path = Path(topk_path).expanduser().resolve() if topk_path is not None else None
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        if self.manifest_path is None or not self.manifest_path.exists():
            raise FileNotFoundError(f"Semantic soft-label manifest does not exist: {self.manifest_path}")

        self.matrix: torch.Tensor | None = None
        self.indptr: np.ndarray | None = None
        self.indices: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.row_index_by_id: dict[str, int] = {}
        self.image_id_by_row_index: dict[int, str] = {}
        for expected_row_index, row in enumerate(rows):
            raw_index = str(row.get("row_index") or "").strip()
            if not raw_index:
                raise ValueError(
                    f"Semantic manifest row {expected_row_index} is missing row_index: {self.manifest_path}"
                )
            row_index = int(raw_index)
            if row_index != expected_row_index:
                raise ValueError(
                    "Semantic manifest row_index must be contiguous and aligned with the matrix order: "
                    f"expected {expected_row_index}, got {row_index}."
                )
            image_id = str(row.get("image_id") or row.get("sample_id") or "").strip()
            if not image_id:
                raise ValueError(
                    f"Semantic manifest row {expected_row_index} is missing image_id/sample_id."
                )
            self.row_index_by_id[image_id] = row_index
            self.image_id_by_row_index[row_index] = image_id

        self.sample_count = len(self.row_index_by_id)
        if self.label_format == "dense":
            self._load_dense()
        else:
            self._load_sparse_topk()

    def _load_dense(self) -> None:
        if self.matrix_path is None or not self.matrix_path.exists():
            raise FileNotFoundError(f"Semantic soft-label matrix does not exist: {self.matrix_path}")
        matrix = np.load(self.matrix_path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Semantic soft-label matrix must be square, got {tuple(matrix.shape)}."
            )
        if len(self.row_index_by_id) != int(matrix.shape[0]):
            raise ValueError(
                "Semantic soft-label manifest row count does not match matrix shape: "
                f"{len(self.row_index_by_id)} vs {int(matrix.shape[0])}."
            )
        self.matrix = torch.from_numpy(matrix.astype(np.float32, copy=False))

    def _load_sparse_topk(self) -> None:
        if self.topk_path is None or not self.topk_path.exists():
            raise FileNotFoundError(f"Sparse semantic soft-label top-k file does not exist: {self.topk_path}")
        payload = np.load(self.topk_path, allow_pickle=False)
        required = {"indptr", "indices", "scores", "row_indices", "image_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                f"Sparse semantic soft-label npz is missing required arrays: {', '.join(sorted(missing))}"
            )
        indptr = payload["indptr"].astype(np.int64, copy=False)
        indices = payload["indices"].astype(np.int64, copy=False)
        scores = payload["scores"].astype(np.float32, copy=False)
        row_indices = payload["row_indices"].astype(np.int64, copy=False)
        image_ids = [str(item) for item in payload["image_ids"].tolist()]
        if indptr.ndim != 1 or len(indptr) != self.sample_count + 1:
            raise ValueError(
                "Sparse semantic soft-label indptr length must equal semantic manifest row_count + 1."
            )
        if len(indices) != len(scores):
            raise ValueError("Sparse semantic soft-label indices and scores lengths must match.")
        if list(row_indices) != list(range(self.sample_count)):
            raise ValueError("Sparse semantic soft-label row_indices must be contiguous from 0.")
        manifest_ids = [self.image_id_by_row_index[index] for index in range(self.sample_count)]
        if image_ids != manifest_ids:
            raise ValueError("Sparse semantic soft-label image_ids must align with semantic manifest row order.")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self.sample_count):
            raise ValueError("Sparse semantic soft-label target row index is out of range.")
        self.indptr = indptr
        self.indices = indices
        self.scores = scores

    def get_batch_matrix(self, row_indices: list[int]) -> torch.Tensor:
        if not row_indices:
            return torch.zeros((0, 0), dtype=torch.float32)
        if self.label_format == "dense":
            assert self.matrix is not None
            return self.matrix[row_indices][:, row_indices].clone().to(dtype=torch.float32)

        assert self.indptr is not None and self.indices is not None and self.scores is not None
        batch_size = len(row_indices)
        batch_position_by_row_index = {
            int(row_index): batch_position
            for batch_position, row_index in enumerate(row_indices)
        }
        matrix = torch.zeros((batch_size, batch_size), dtype=torch.float32)
        for source_position, source_row_index in enumerate(row_indices):
            start = int(self.indptr[int(source_row_index)])
            end = int(self.indptr[int(source_row_index) + 1])
            for target_row_index, score in zip(self.indices[start:end], self.scores[start:end]):
                target_position = batch_position_by_row_index.get(int(target_row_index))
                if target_position is None:
                    continue
                current = float(matrix[source_position, target_position].item())
                matrix[source_position, target_position] = max(current, float(score))
        matrix = torch.maximum(matrix, matrix.transpose(0, 1))
        matrix.fill_diagonal_(1.0)
        return matrix.clamp_(0.0, 1.0)

    def resolve_batch(
        self,
        image_ids: list[str],
        row_indices: list[int] | None = None,
    ) -> SemanticSoftLabelBatch:
        input_row_indices = row_indices
        resolved_row_indices: list[int] = []
        batch_positions: list[int] = []
        valid_image_ids: list[str] = []
        warnings: list[str] = []
        for batch_position, image_id in enumerate(image_ids):
            row_index = None
            if input_row_indices is not None:
                candidate = int(input_row_indices[batch_position])
                if 0 <= candidate < self.sample_count:
                    row_index = candidate
                    expected_image_id = self.image_id_by_row_index.get(candidate, "")
                    if expected_image_id and expected_image_id != str(image_id).strip():
                        warnings.append(
                            f"semantic_soft_label_row_image_mismatch:{candidate}:{image_id}"
                        )
                else:
                    warnings.append(f"semantic_soft_label_row_index_out_of_range:{candidate}")
            else:
                row_index = self.row_index_by_id.get(str(image_id).strip())
            if row_index is None:
                warnings.append(f"semantic_soft_label_missing:{image_id}")
                continue
            resolved_row_indices.append(row_index)
            batch_positions.append(batch_position)
            valid_image_ids.append(str(image_id))

        if not resolved_row_indices:
            return SemanticSoftLabelBatch(
                matrix=torch.zeros((0, 0), dtype=torch.float32),
                batch_positions=[],
                image_ids=[],
                warnings=warnings,
            )

        submatrix = self.get_batch_matrix(resolved_row_indices)
        return SemanticSoftLabelBatch(
            matrix=submatrix.to(dtype=torch.float32),
            batch_positions=batch_positions,
            image_ids=valid_image_ids,
            warnings=warnings,
        )
