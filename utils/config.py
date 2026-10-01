from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import yaml


@dataclass(frozen=True)
class LocalSmokeConfig:
    project_root: Path
    fgclip_repo_path: Path
    fgi_repo_path: Path
    panderm_repo_path: Path
    image_manifest_path: Path
    attention_map_dir: Path
    teacher_latent_dir: Path
    text_prompt_path: Path
    image_size: int
    batch_size: int
    num_workers: int
    device: str
    output_dir: Path
    max_samples: int | None = None
    require_attention_prior_paths: bool = False
    require_teacher_latents: bool = False
    teacher: "TeacherConfig | None" = None
    train_smoke: "TrainSmokeConfig | None" = None


@dataclass(frozen=True)
class TrainSmokeConfig:
    patch_size: int
    latent_dim: int
    mask_ratio: float
    max_steps: int
    learning_rate: float
    gaze_loss_mode: str = "soft_attention_plus_high_conf"
    gaze_weight_alpha: float = 1.0
    mask_strategy: str = "random"
    gaze_mask_sampling_alpha: float = 0.7
    high_conf_mask_quota: float = 0.6
    min_random_mask_fraction: float = 0.3
    mask_sampling_temperature: float = 1.0
    gaze_mask_eps: float = 1e-6
    seed: int = 42
    eval_seed: int | None = None
    deterministic_ablation: bool = True
    reuse_initial_model: bool = True
    reuse_patch_mask: bool = True


@dataclass(frozen=True)
class TeacherConfig:
    source_type: str
    backend: str | None = None
    model_name: str | None = None
    expected_patch_count: int | None = None
    expected_patch_grid: tuple[int, int] | None = None
    latent_dim: int | None = None
    batch_size: int | None = None
    device: str | None = None
    local_files_only: bool = False


PATH_KEYS = {
    "project_root",
    "fgclip_repo_path",
    "fgi_repo_path",
    "panderm_repo_path",
    "image_manifest_path",
    "attention_map_dir",
    "teacher_latent_dir",
    "text_prompt_path",
    "output_dir",
}


def _resolve_path(raw_value: Any, base_dir: Path) -> Path:
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _warn_external_repo(path: Path, label: str) -> None:
    if not path.exists():
        warnings.warn(f"{label} does not exist: {path}", stacklevel=2)
        return
    if not path.is_dir():
        warnings.warn(f"{label} is not a directory: {path}", stacklevel=2)


def _as_optional_positive_int(raw_value: Any, field_name: str) -> int | None:
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided.")
    return value


def _as_bool(raw_value: Any, field_name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value, got: {raw_value!r}")


def _as_optional_pair_of_positive_ints(
    raw_value: Any,
    field_name: str,
) -> tuple[int, int] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{field_name} must be a 2-item list/tuple when provided.")

    height = int(raw_value[0])
    width = int(raw_value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{field_name} values must be positive integers.")
    return height, width


def _load_train_smoke_config(raw_value: Any) -> TrainSmokeConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("train_smoke must be a mapping when provided.")

    missing_keys = [
        key
        for key in ("patch_size", "latent_dim", "mask_ratio", "max_steps", "learning_rate")
        if key not in raw_value
    ]
    if missing_keys:
        raise KeyError(f"Missing required train_smoke keys: {', '.join(missing_keys)}")

    return TrainSmokeConfig(
        patch_size=int(raw_value["patch_size"]),
        latent_dim=int(raw_value["latent_dim"]),
        mask_ratio=float(raw_value["mask_ratio"]),
        max_steps=int(raw_value["max_steps"]),
        learning_rate=float(raw_value["learning_rate"]),
        gaze_loss_mode=str(
            raw_value.get("gaze_loss_mode", "soft_attention_plus_high_conf")
        ),
        gaze_weight_alpha=float(raw_value.get("gaze_weight_alpha", 1.0)),
        mask_strategy=str(raw_value.get("mask_strategy", "random")),
        gaze_mask_sampling_alpha=float(raw_value.get("gaze_mask_sampling_alpha", 0.7)),
        high_conf_mask_quota=float(raw_value.get("high_conf_mask_quota", 0.6)),
        min_random_mask_fraction=float(raw_value.get("min_random_mask_fraction", 0.3)),
        mask_sampling_temperature=float(raw_value.get("mask_sampling_temperature", 1.0)),
        gaze_mask_eps=float(raw_value.get("gaze_mask_eps", 1e-6)),
        seed=int(raw_value.get("seed", 42)),
        eval_seed=_as_optional_positive_int(
            raw_value.get("eval_seed"),
            "train_smoke.eval_seed",
        ),
        deterministic_ablation=_as_bool(
            raw_value.get("deterministic_ablation", True),
            "train_smoke.deterministic_ablation",
        ),
        reuse_initial_model=_as_bool(
            raw_value.get("reuse_initial_model", True),
            "train_smoke.reuse_initial_model",
        ),
        reuse_patch_mask=_as_bool(
            raw_value.get("reuse_patch_mask", True),
            "train_smoke.reuse_patch_mask",
        ),
    )


def _load_teacher_config(raw_value: Any) -> TeacherConfig:
    if raw_value is None:
        return TeacherConfig(
            source_type="deterministic_fixture_teacher",
            backend=None,
            model_name=None,
            expected_patch_count=None,
            expected_patch_grid=None,
            latent_dim=None,
            batch_size=None,
            device=None,
            local_files_only=False,
        )
    if not isinstance(raw_value, dict):
        raise ValueError("teacher must be a mapping when provided.")

    source_type = str(
        raw_value.get("source_type", "deterministic_fixture_teacher")
    ).strip()
    if not source_type:
        raise ValueError("teacher.source_type must be a non-empty string.")

    backend = raw_value.get("backend")
    raw_model_name = raw_value.get("model_name")
    raw_model_name_or_path = raw_value.get("model_name_or_path")
    model_name = None
    if raw_model_name is not None:
        model_name = str(raw_model_name).strip()
    if raw_model_name_or_path is not None:
        model_name_or_path = str(raw_model_name_or_path).strip()
        if model_name and model_name_or_path and model_name != model_name_or_path:
            raise ValueError(
                "teacher.model_name and teacher.model_name_or_path must match when both are provided."
            )
        model_name = model_name_or_path or model_name

    return TeacherConfig(
        source_type=source_type,
        backend=str(backend).strip() if backend is not None else None,
        model_name=model_name or None,
        expected_patch_count=_as_optional_positive_int(
            raw_value.get("expected_patch_count"),
            "teacher.expected_patch_count",
        ),
        expected_patch_grid=_as_optional_pair_of_positive_ints(
            raw_value.get("expected_patch_grid"),
            "teacher.expected_patch_grid",
        ),
        latent_dim=_as_optional_positive_int(
            raw_value.get("latent_dim"),
            "teacher.latent_dim",
        ),
        batch_size=_as_optional_positive_int(
            raw_value.get("batch_size"),
            "teacher.batch_size",
        ),
        device=(
            str(raw_value.get("device")).strip()
            if raw_value.get("device") is not None
            else None
        ),
        local_files_only=_as_bool(
            raw_value.get("local_files_only", False),
            "teacher.local_files_only",
        ),
    )


def load_config(config_path: str | Path) -> LocalSmokeConfig:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    missing_keys = [
        key
        for key in (
            "project_root",
            "fgclip_repo_path",
            "fgi_repo_path",
            "panderm_repo_path",
            "image_manifest_path",
            "attention_map_dir",
            "teacher_latent_dir",
            "text_prompt_path",
            "image_size",
            "batch_size",
            "num_workers",
            "device",
            "output_dir",
        )
        if key not in raw_config
    ]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")

    project_root = _resolve_path(raw_config["project_root"], config_path.parent)
    resolved: dict[str, Any] = {}

    for key, value in raw_config.items():
        if key == "project_root":
            resolved[key] = project_root
        elif key in PATH_KEYS:
            resolved[key] = _resolve_path(value, project_root)
        else:
            resolved[key] = value

    _warn_external_repo(resolved["fgclip_repo_path"], "FG-CLIP repository")
    _warn_external_repo(resolved["fgi_repo_path"], "FGI repository")
    _warn_external_repo(resolved["panderm_repo_path"], "PanDerm repository")

    return LocalSmokeConfig(
        project_root=resolved["project_root"],
        fgclip_repo_path=resolved["fgclip_repo_path"],
        fgi_repo_path=resolved["fgi_repo_path"],
        panderm_repo_path=resolved["panderm_repo_path"],
        image_manifest_path=resolved["image_manifest_path"],
        attention_map_dir=resolved["attention_map_dir"],
        teacher_latent_dir=resolved["teacher_latent_dir"],
        text_prompt_path=resolved["text_prompt_path"],
        image_size=int(resolved["image_size"]),
        batch_size=int(resolved["batch_size"]),
        num_workers=int(resolved["num_workers"]),
        device=str(resolved["device"]),
        output_dir=resolved["output_dir"],
        max_samples=_as_optional_positive_int(
            raw_config.get("max_samples"),
            "max_samples",
        ),
        require_attention_prior_paths=_as_bool(
            raw_config.get("require_attention_prior_paths", False),
            "require_attention_prior_paths",
        ),
        require_teacher_latents=_as_bool(
            raw_config.get("require_teacher_latents", False),
            "require_teacher_latents",
        ),
        teacher=_load_teacher_config(raw_config.get("teacher")),
        train_smoke=_load_train_smoke_config(raw_config.get("train_smoke")),
    )
