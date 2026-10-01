from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breast_pretrain.data.datasets import JointPretrainDataset
from breast_pretrain.teachers.base import TEACHER_SOURCE_DETERMINISTIC_FIXTURE
from breast_pretrain.teachers.factory import create_teacher_encoder
from breast_pretrain.teachers.token_utils import infer_patch_grid, resize_teacher_tokens_2d
from breast_pretrain.train.masked_latent_smoke import compute_tensor_checksum, expected_patch_count
from breast_pretrain.utils.config import TeacherConfig


def build_teacher_latent_cache(
    *,
    manifest_path: str | Path,
    image_size: int,
    teacher_latent_dir: str | Path,
    patch_size: int,
    latent_dim: int,
    teacher_config: TeacherConfig | None = None,
    attention_map_dir: str | Path | None = None,
    text_prompt_path: str | Path | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_teacher_dir = Path(teacher_latent_dir).expanduser().resolve()
    resolved_teacher_dir.mkdir(parents=True, exist_ok=True)
    dataset = JointPretrainDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        attention_map_dir=attention_map_dir,
        teacher_latent_dir=resolved_teacher_dir,
        text_prompt_path=text_prompt_path,
        max_samples=max_samples,
        require_attention_prior_paths=False,
    )
    teacher_spec = teacher_config or TeacherConfig(source_type=TEACHER_SOURCE_DETERMINISTIC_FIXTURE)
    teacher = create_teacher_encoder(
        teacher_config=teacher_spec,
        patch_size=patch_size,
        latent_dim=latent_dim,
        device=torch.device("cpu"),
    )
    target_patch_count = expected_patch_count(image_size=image_size, patch_size=patch_size)
    entries: list[dict[str, Any]] = []
    created_count = 0
    reused_count = 0
    for sample in dataset:
        image_id = str(sample["image_id"])
        target_path = resolved_teacher_dir / f"{image_id}.npy"
        if target_path.exists() and not overwrite:
            cached = np.load(target_path, allow_pickle=False)
            checksum = compute_tensor_checksum(torch.from_numpy(np.asarray(cached)))
            entries.append(
                {
                    "image_id": image_id,
                    "path": str(target_path),
                    "shape": list(cached.shape),
                    "checksum": checksum,
                    "source_type": teacher.source_type,
                    "teacher_model_name": teacher.teacher_model_name,
                    "raw_patch_grid": list(infer_patch_grid(int(cached.shape[0]))),
                }
            )
            reused_count += 1
            continue
        image = sample["image"].unsqueeze(0)
        latent_batch = teacher.build_latents(image)
        resized = resize_teacher_tokens_2d(
            latent_batch.tokens.squeeze(0).detach().cpu(),
            expected_patch_count=target_patch_count,
            image_id=image_id,
            teacher_path=target_path,
        )
        array = resized.detach().cpu().numpy().astype(np.float32, copy=False)
        np.save(target_path, array, allow_pickle=False)
        entries.append(
            {
                "image_id": image_id,
                "path": str(target_path),
                "shape": list(array.shape),
                "checksum": compute_tensor_checksum(resized),
                "source_type": teacher.source_type,
                "teacher_model_name": latent_batch.teacher_model_name,
                "raw_patch_grid": list(latent_batch.raw_patch_grid),
            }
        )
        created_count += 1

    manifest_payload = {
        "entries": entries,
        "expected_patch_count": target_patch_count,
        "latent_dim": latent_dim,
    }
    manifest_path_out = resolved_teacher_dir / "teacher_latent_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary = audit_teacher_latent_cache(
        teacher_latent_dir=resolved_teacher_dir,
        expected_patch_count=target_patch_count,
        expected_latent_dim=latent_dim,
    )
    summary.update(
        {
            "teacher_latent_dir": str(resolved_teacher_dir),
            "created_count": created_count,
            "reused_count": reused_count,
            "manifest_path": str(manifest_path_out),
        }
    )
    return summary


def audit_teacher_latent_cache(
    *,
    teacher_latent_dir: str | Path,
    expected_patch_count: int,
    expected_latent_dim: int,
) -> dict[str, Any]:
    resolved_teacher_dir = Path(teacher_latent_dir).expanduser().resolve()
    manifest_path = resolved_teacher_dir / "teacher_latent_manifest.json"
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = list(payload.get("entries", []))
    missing_count = 0
    shape_mismatch_count = 0
    checksum_counter: Counter[str] = Counter()
    for entry in entries:
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.is_absolute():
            path = (resolved_teacher_dir / path).resolve()
        if not path.exists():
            missing_count += 1
            continue
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != (expected_patch_count, expected_latent_dim):
            shape_mismatch_count += 1
        checksum_counter[str(entry.get("checksum", ""))] += 1
    duplicate_checksum_count = sum(count - 1 for count in checksum_counter.values() if count > 1)
    return {
        "entry_count": len(entries),
        "missing_count": missing_count,
        "shape_mismatch_count": shape_mismatch_count,
        "duplicate_checksum_count": duplicate_checksum_count,
        "unique_checksum_count": len([key for key in checksum_counter if key]),
    }
