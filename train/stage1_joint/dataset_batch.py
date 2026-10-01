from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from breast_pretrain.data.bucketed_stage1_dataloader import (
    BATCH_POLICY_BUCKET_BY_MODALITY_AND_IMAGE_SIZE,
    build_bucket_coverage_summary,
    build_bucketed_stage1_dataloader,
)
from breast_pretrain.data.collators.joint_pretrain_collator import joint_pretrain_collate_fn
from breast_pretrain.clinical_graph_sidecar.runtime_loader import ClinicalGraphV2SidecarLoader
from breast_pretrain.data.clinical_concept_canonicalization import canonicalize_concept_label
from breast_pretrain.data.stage1_sparse_concept_runtime import load_formal_p0b_batch_targets
from breast_pretrain.data.datasets import JointPretrainDataset
from breast_pretrain.datasets.breast_image_dataset import BreastImageDataset
from breast_pretrain.text.clinical_concepts import (
    birads_to_index,
    density_to_index,
    finding_labels_to_multi_hot,
    normalize_finding_labels,
    normalize_view,
    normalize_laterality,
    normalize_density,
    normalize_birads,
    normalize_cancer_label,
    normalize_benign_malignant_label,
    normalize_mri_sequence,
    normalize_mri_treatment_response,
    laterality_to_index,
    view_to_index,
    benign_malignant_to_index,
    mri_sequence_to_index,
    mri_treatment_response_to_index,
)
from breast_pretrain.train.stage1_joint.types import Stage1JointBatch, Stage1JointTrainerConfig
from breast_pretrain.train.distributed import resolve_distributed_runtime
from breast_pretrain.train.stage1_joint.formal_config_strict_keys import FORMAL_PRODUCTION_TIERS


def resolve_device(requested_device: str) -> torch.device:
    requested = str(requested_device).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {requested}")
    return torch.device(requested or "cpu")


def build_stage1_joint_dataset(
    config: Stage1JointTrainerConfig,
    *,
    split: str = "train",
) -> BreastImageDataset:
    graph_sidecar_loader = None
    clinical_graph = config.clinical_graph
    if (
        clinical_graph is not None
        and str(clinical_graph.version).strip().lower()
        in {"clinical_graph_v2", "tri_modal_clinical_graph_v2", "tri_modal_clinical_graph_v2_5"}
    ):
        if clinical_graph.nodes_path is None or clinical_graph.sidecar_case_concept_vector_path is None:
            raise ValueError(
                "Clinical Graph V2 requires nodes_path and sidecar_case_concept_vector_path for observed-subgraph consumption."
            )
        graph_sidecar_loader = ClinicalGraphV2SidecarLoader(
            nodes_path=clinical_graph.nodes_path,
            sidecar_path=clinical_graph.sidecar_case_concept_vector_path,
            index_path=getattr(clinical_graph, "sidecar_index_path", None),
        )
    dataset = JointPretrainDataset(
        manifest_path=config.data.image_manifest_path,
        image_size=config.data.image_size,
        attention_map_dir=config.data.attention_map_dir,
        teacher_latent_dir=config.data.teacher_latent_dir,
        text_prompt_path=config.data.text_prompt_path,
        max_samples=config.data.max_samples,
        require_attention_prior_paths=config.data.require_attention_prior_paths,
        image_size_by_modality=config.data.image_size_by_modality,
        transform_policy_by_modality=config.data.transform_policy_by_modality,
        patch_size=config.model.patch_size,
        clinical_graph_v2_sidecar_loader=graph_sidecar_loader,
        dataset_entry_v2_enabled=config.data.dataset_entry_v2_enabled,
        image_release_root=config.data.dataset_entry_v2_image_release_root,
        gaze_release_root=config.data.dataset_entry_v2_gaze_release_root,
        canonical_runtime=config.data.canonical_runtime,
        source_runtime_cache=config.data.source_runtime_cache,
        gaze_membership_path=config.data.gaze_membership_path,
        gaze_membership_expected_available=config.data.gaze_membership_expected_available,
        gaze_membership_expected_disabled=config.data.gaze_membership_expected_disabled,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; Stage 1 joint trainer requires at least one sample.")
    requested_split = str(split).strip().lower()
    if requested_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported Stage 1 runtime split: {split!r}")
    # Split membership is frozen in the formal manifest. Runtime consumers only
    # filter it; they never reassign or reshuffle split authority.
    dataset.records = [
        record for record in dataset.records
        if str(record.get("split", "")).strip().lower() == requested_split
    ]
    if not dataset.records:
        raise ValueError(f"Formal Stage 1 manifest contains no rows for split={requested_split!r}.")
    return dataset


def build_stage1_joint_dataloader(
    dataset: BreastImageDataset,
    config: Stage1JointTrainerConfig,
    *,
    epoch: int = 1,
    start_batch_index: int = 0,
) -> DataLoader:
    distributed_runtime = resolve_distributed_runtime(config.data.device)
    rank = int(distributed_runtime["rank"])
    world_size = int(distributed_runtime["world_size"])
    if config.data.batch_policy == BATCH_POLICY_BUCKET_BY_MODALITY_AND_IMAGE_SIZE:
        if not config.data.batch_size_by_modality:
            raise ValueError(
                "batch_size_by_modality is required when "
                "batch_policy=bucket_by_modality_and_image_size."
            )
        return build_bucketed_stage1_dataloader(
            dataset,
            batch_size_by_modality=config.data.batch_size_by_modality,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            seed=config.reproducibility.seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            start_batch_index=int(start_batch_index),
            contract_version=str(config.data.sampler_contract_version),
            strict_formal=str(config.metadata.run_tier).strip() in FORMAL_PRODUCTION_TIERS,
        )
    if config.data.batch_policy not in {"fixed_batch_size", ""}:
        raise ValueError(f"Unsupported Stage 1 batch_policy: {config.data.batch_policy!r}")
    if start_batch_index:
        raise ValueError("start_batch_index resume is only supported by bucketed Stage 1 batching.")
    generator = None
    if config.data.shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.reproducibility.seed) + max(0, int(epoch) - 1))
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(config.data.shuffle),
            seed=int(config.reproducibility.seed),
            drop_last=False,
        )
        sampler.set_epoch(max(0, int(epoch) - 1))
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=bool(config.data.shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=joint_pretrain_collate_fn,
        generator=generator,
    )


def build_stage1_batching_summary(
    dataset: BreastImageDataset,
    config: Stage1JointTrainerConfig,
) -> dict[str, object]:
    if config.data.batch_policy == BATCH_POLICY_BUCKET_BY_MODALITY_AND_IMAGE_SIZE:
        return build_bucket_coverage_summary(
            dataset,
            batch_policy=config.data.batch_policy,
            batch_size_by_modality=config.data.batch_size_by_modality or {},
            contract_version=str(config.data.sampler_contract_version),
        )
    return {
        "batch_policy": config.data.batch_policy,
        "batch_size": int(config.data.batch_size),
        "shuffle": bool(config.data.shuffle),
    }


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _batch_list(batch: dict[str, object], key: str, batch_size: int, default: str = "") -> list[str]:
    value = batch.get(key)
    if value is None:
        return [default] * batch_size
    if isinstance(value, torch.Tensor):
        return [str(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * batch_size


def _batch_float_list(batch: dict[str, object], key: str, batch_size: int) -> list[float]:
    values = _batch_list(batch, key, batch_size)
    result: list[float] = []
    for value in values:
        try:
            result.append(float(str(value).strip()))
        except ValueError:
            result.append(0.0)
    return result


def _batch_raw_value(batch: dict[str, object], key: str, index: int) -> object:
    values = batch.get(key)
    if values is None:
        return ""
    if isinstance(values, torch.Tensor):
        return values[index].detach().cpu().item()
    if isinstance(values, (list, tuple)):
        return values[index]
    return values


def _observed_mask_override(batch: dict[str, object], head_name: str, index: int) -> bool | None:
    values = batch.get(f"{head_name}_observed_mask")
    if values is None:
        return None
    raw = str(_batch_raw_value(batch, f"{head_name}_observed_mask", index))
    if not raw.strip():
        return None
    canonical = canonicalize_concept_label(
        value=_batch_raw_value(batch, head_name, index),
        status=_batch_raw_value(batch, f"{head_name}_status", index),
        observed_mask=raw,
        source=_batch_raw_value(batch, f"{head_name}_source", index),
        confidence=_batch_raw_value(batch, f"{head_name}_confidence", index),
    )
    return canonical.observed_mask


def _prepare_clinical_graph_v2_tensors(
    batch: dict[str, object],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, tuple[str, ...] | None]:
    values = batch.get("clinical_graph_v2_node_values")
    observed_mask = batch.get("clinical_graph_v2_observed_mask")
    node_index = batch.get("clinical_graph_v2_node_index")
    node_ids = batch.get("clinical_graph_v2_node_ids")
    if values is None and observed_mask is None and node_index is None and node_ids is None:
        return None, None, None, None
    if not isinstance(values, torch.Tensor) or not isinstance(observed_mask, torch.Tensor) or not isinstance(node_index, torch.Tensor):
        raise TypeError("Clinical Graph V2 batch tensors must all be torch.Tensor instances.")
    if not isinstance(node_ids, (list, tuple)):
        raise TypeError("Clinical Graph V2 node ids must be a stable sequence of strings.")
    canonical_node_ids = tuple(str(node_id).strip() for node_id in node_ids)
    if values.dtype != torch.float32 or observed_mask.dtype != torch.bool or node_index.dtype != torch.long:
        raise TypeError("Clinical Graph V2 batch tensors must use float32 values, bool masks, and int64 node index.")
    if values.ndim != 2 or observed_mask.ndim != 2 or node_index.ndim != 1:
        raise ValueError("Clinical Graph V2 values/mask must be [B, N] and node index must be [N].")
    if int(values.shape[0]) != batch_size or values.shape != observed_mask.shape or int(values.shape[1]) != int(node_index.numel()):
        raise ValueError("Clinical Graph V2 batch tensor shapes do not match the Stage 1 batch contract.")
    if len(canonical_node_ids) != int(node_index.numel()) or any(not node_id for node_id in canonical_node_ids):
        raise ValueError("Clinical Graph V2 node ids do not match the canonical node index.")
    if len(set(canonical_node_ids)) != len(canonical_node_ids):
        raise ValueError("Clinical Graph V2 node ids must be unique.")
    if not torch.isfinite(values).all():
        raise ValueError("Clinical Graph V2 node values must be finite.")
    expected_index = torch.arange(node_index.numel(), device=node_index.device, dtype=torch.long)
    if not torch.equal(node_index, expected_index):
        raise ValueError("Clinical Graph V2 node index must be the canonical contiguous [0, N) order.")
    normalized_mask = observed_mask.to(device=device, dtype=torch.bool)
    normalized_values = values.to(device=device, dtype=torch.float32) * normalized_mask.to(
        device=device, dtype=torch.float32
    )
    return normalized_values, normalized_mask, node_index.to(device=device, dtype=torch.long), canonical_node_ids


def _gate_legacy_concept_masks_with_v2_evidence(
    *,
    targets: dict[str, torch.Tensor],
    valid_masks: dict[str, torch.Tensor],
    node_values: torch.Tensor | None,
    observed_mask: torch.Tensor | None,
    node_ids: tuple[str, ...] | None,
) -> None:
    """Require matching observed V2 evidence for legacy axis/group head labels.

    V2 deliberately has no one-head-per-node supervision.  Only legacy heads
    with a lossless axis/class mapping are gated here; unmapped or partially
    observed multi-label axes remain unavailable instead of creating negatives.
    """

    if node_values is None or observed_mask is None or node_ids is None:
        return
    if node_values.device != observed_mask.device:
        raise ValueError(
            "Clinical Graph V2 gate device contract requires node_values and "
            "observed_mask to be on the same device."
        )
    gate_device = node_values.device
    normalized_observed_mask = observed_mask.to(
        device=gate_device,
        dtype=torch.bool,
    )
    normalized_node_values = node_values.to(
        device=gate_device,
        dtype=torch.float32,
    )
    node_lookup = {node_id: index for index, node_id in enumerate(node_ids)}
    class_node_ids = {
        "view": ("mammography.view.cc", "mammography.view.mlo"),
        "laterality": (
            "anatomy_location.laterality.left",
            "anatomy_location.laterality.right",
        ),
        "density": (
            "mammography.density.a_almost_entirely_fatty",
            "mammography.density.b_scattered_fibroglandular",
            "mammography.density.c_heterogeneously_dense",
            "mammography.density.d_extremely_dense",
        ),
        "birads": (
            "assessment.birads.0_incomplete",
            "assessment.birads.1_negative",
            "assessment.birads.2_benign",
            "assessment.birads.3_probably_benign",
            "assessment.birads.4_suspicious",
            "",
            "",
            "",
            "assessment.birads.5_highly_suggestive_malignancy",
            "assessment.birads.6_known_biopsy_proven_malignancy",
        ),
    }
    for head_name, valid_mask in valid_masks.items():
        if head_name not in class_node_ids:
            valid_masks[head_name] = torch.zeros_like(
                valid_mask,
                dtype=torch.bool,
                device=gate_device,
            )
    for head_name, candidates in class_node_ids.items():
        target = targets.get(head_name)
        valid_mask = valid_masks.get(head_name)
        if target is None or valid_mask is None:
            continue
        valid_mask_on_device = valid_mask.to(
            device=gate_device,
            dtype=torch.bool,
        )
        gates = torch.zeros_like(
            valid_mask_on_device,
            dtype=torch.bool,
            device=gate_device,
        )
        for class_index, node_id in enumerate(candidates):
            if not node_id or node_id not in node_lookup:
                continue
            node_index = node_lookup[node_id]
            expected_class = target.to(
                device=gate_device,
                dtype=torch.long,
            ) == class_index
            present_and_observed = (
                normalized_observed_mask[:, node_index]
                & (normalized_node_values[:, node_index] > 0.5)
            )
            gates = gates | (expected_class & present_and_observed)
        valid_masks[head_name] = valid_mask_on_device & gates


def _build_case_prompts_and_labels(
    batch: dict[str, object],
    *,
    clinical_graph_v2_node_values: torch.Tensor | None = None,
    clinical_graph_v2_observed_mask: torch.Tensor | None = None,
    clinical_graph_v2_node_ids: tuple[str, ...] | None = None,
) -> tuple[list[str], list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch_size = int(batch["image"].shape[0])
    prompts: list[str] = []
    prompt_warnings: list[str] = []
    concept_targets: dict[str, list[object]] = {
        "view": [],
        "laterality": [],
        "density": [],
        "birads": [],
        "finding": [],
        "cancer_label": [],
        "benign_malignant_label": [],
        "mri_sequence": [],
        "mri_treatment_response": [],
    }
    concept_valid_masks: dict[str, list[bool]] = {
        head_name: [] for head_name in concept_targets
    }
    for index in range(batch_size):
        raw_view = batch["view"][index]
        raw_laterality = batch["laterality"][index]
        raw_density = batch["density"][index]
        raw_finding = batch["finding"][index]
        raw_birads = batch.get("birads", [""] * batch_size)[index]
        raw_cancer_label = batch.get("cancer_label", [""] * batch_size)[index]
        raw_benign_malignant = batch.get("benign_malignant_label", [""] * batch_size)[index]
        raw_mri_sequence = batch.get("mri_sequence", [""] * batch_size)[index]
        raw_mri_treatment_response = batch.get("mri_treatment_response", [""] * batch_size)[index]
        resolved_prompt = str(batch.get("text_prompt", [""] * batch_size)[index]).strip()
        if resolved_prompt:
            prompts.append(resolved_prompt)
        else:
            raise ValueError(
                "V6 Stage 1 batch is missing an Effective Report text_prompt for "
                f"image_id={batch['image_id'][index]}; runtime Structured Prompt construction is forbidden."
            )
        prompt_warning = str(batch.get("text_prompt_warning", [""] * batch_size)[index]).strip()
        if prompt_warning:
            prompt_warnings.append(prompt_warning)

        view_label, inferred_laterality = normalize_view(raw_view)
        laterality_label = normalize_laterality(raw_laterality) or (inferred_laterality or "")
        density_label = normalize_density(raw_density)
        birads_label = normalize_birads(raw_birads)
        finding_labels = normalize_finding_labels(raw_finding)
        cancer_label = normalize_cancer_label(raw_cancer_label)
        benign_malignant_label = normalize_benign_malignant_label(raw_benign_malignant)
        mri_sequence_label = normalize_mri_sequence(raw_mri_sequence)
        mri_treatment_response_label = normalize_mri_treatment_response(raw_mri_treatment_response)

        view_observed = _observed_mask_override(batch, "view", index)
        laterality_observed = _observed_mask_override(batch, "laterality", index)
        density_observed = _observed_mask_override(batch, "density", index)
        birads_observed = _observed_mask_override(batch, "birads", index)
        finding_observed = _observed_mask_override(batch, "finding", index)
        cancer_observed = _observed_mask_override(batch, "cancer_label", index)
        benign_malignant_observed = _observed_mask_override(batch, "benign_malignant_label", index)
        mri_sequence_observed = _observed_mask_override(batch, "mri_sequence", index)
        mri_treatment_response_observed = _observed_mask_override(batch, "mri_treatment_response", index)

        concept_valid_masks["view"].append(bool(view_label) if view_observed is None else (view_observed and bool(view_label)))
        concept_targets["view"].append(view_to_index(view_label) if view_label else 0)
        concept_valid_masks["laterality"].append(
            bool(laterality_label)
            if laterality_observed is None
            else (laterality_observed and bool(laterality_label))
        )
        concept_targets["laterality"].append(
            laterality_to_index(laterality_label) if laterality_label else 0
        )
        concept_valid_masks["density"].append(
            bool(density_label) if density_observed is None else (density_observed and bool(density_label))
        )
        concept_targets["density"].append(density_to_index(density_label) if density_label else 0)
        concept_valid_masks["birads"].append(
            bool(birads_label) if birads_observed is None else (birads_observed and bool(birads_label))
        )
        concept_targets["birads"].append(birads_to_index(birads_label) if birads_label else 0)
        concept_valid_masks["finding"].append(
            bool(finding_labels)
            if finding_observed is None
            else (finding_observed and bool(finding_labels))
        )
        concept_targets["finding"].append(finding_labels_to_multi_hot(finding_labels))
        concept_valid_masks["cancer_label"].append(
            bool(cancer_label) if cancer_observed is None else (cancer_observed and bool(cancer_label))
        )
        concept_targets["cancer_label"].append(float(cancer_label) if cancer_label else 0.0)
        concept_valid_masks["benign_malignant_label"].append(
            bool(benign_malignant_label)
            if benign_malignant_observed is None
            else (benign_malignant_observed and bool(benign_malignant_label))
        )
        concept_targets["benign_malignant_label"].append(
            benign_malignant_to_index(benign_malignant_label)
            if benign_malignant_label
            else 0
        )
        concept_valid_masks["mri_sequence"].append(
            bool(mri_sequence_label)
            if mri_sequence_observed is None
            else (mri_sequence_observed and bool(mri_sequence_label))
        )
        concept_targets["mri_sequence"].append(
            mri_sequence_to_index(mri_sequence_label) if mri_sequence_label else 0
        )
        concept_valid_masks["mri_treatment_response"].append(
            bool(mri_treatment_response_label)
            if mri_treatment_response_observed is None
            else (mri_treatment_response_observed and bool(mri_treatment_response_label))
        )
        concept_targets["mri_treatment_response"].append(
            mri_treatment_response_to_index(mri_treatment_response_label)
            if mri_treatment_response_label
            else 0
        )

    tensor_targets: dict[str, torch.Tensor] = {
        "view": torch.tensor(concept_targets["view"], dtype=torch.long),
        "laterality": torch.tensor(concept_targets["laterality"], dtype=torch.long),
        "density": torch.tensor(concept_targets["density"], dtype=torch.long),
        "birads": torch.tensor(concept_targets["birads"], dtype=torch.long),
        "finding": torch.tensor(concept_targets["finding"], dtype=torch.float32),
        "cancer_label": torch.tensor(concept_targets["cancer_label"], dtype=torch.float32),
        "benign_malignant_label": torch.tensor(
            concept_targets["benign_malignant_label"],
            dtype=torch.long,
        ),
        "mri_sequence": torch.tensor(
            concept_targets["mri_sequence"],
            dtype=torch.long,
        ),
        "mri_treatment_response": torch.tensor(
            concept_targets["mri_treatment_response"],
            dtype=torch.long,
        ),
    }
    tensor_masks = {
        head_name: torch.tensor(head_mask, dtype=torch.bool)
        for head_name, head_mask in concept_valid_masks.items()
    }
    _gate_legacy_concept_masks_with_v2_evidence(
        targets=tensor_targets,
        valid_masks=tensor_masks,
        node_values=clinical_graph_v2_node_values,
        observed_mask=clinical_graph_v2_observed_mask,
        node_ids=clinical_graph_v2_node_ids,
    )
    return prompts, prompt_warnings, tensor_targets, tensor_masks


def prepare_stage1_joint_batch(
    raw_batch: dict[str, Any],
    device: torch.device,
    config: Stage1JointTrainerConfig | None = None,
) -> Stage1JointBatch:
    batch = _to_device(dict(raw_batch), device=device)
    raw_row_indices = batch.get("manifest_row_index")
    if isinstance(raw_row_indices, torch.Tensor):
        manifest_row_indices = [int(item) for item in raw_row_indices.detach().cpu().tolist()]
    else:
        manifest_row_indices = [int(item) for item in raw_row_indices or range(len(batch["image_id"]))]
    batch_size = int(batch["image"].shape[0])
    clinical_graph_v2_node_values, clinical_graph_v2_observed_mask, clinical_graph_v2_node_index, clinical_graph_v2_node_ids = _prepare_clinical_graph_v2_tensors(
        batch,
        batch_size=batch_size,
        device=device,
    )
    if config is not None and config.semantic.formal_p0b is not None:
        prompts = [str(value).strip() for value in _batch_list(batch, "text_prompt", batch_size)]
        if any(not prompt for prompt in prompts):
            raise ValueError("Formal P0-B batch is missing an Effective Report text_prompt.")
        prompt_warnings = [value for value in _batch_list(batch, "text_prompt_warning", batch_size) if value]
        concept_targets, concept_valid_masks = {}, {}
    else:
        prompts, prompt_warnings, concept_targets, concept_valid_masks = _build_case_prompts_and_labels(
            batch,
            clinical_graph_v2_node_values=clinical_graph_v2_node_values,
            clinical_graph_v2_observed_mask=clinical_graph_v2_observed_mask,
            clinical_graph_v2_node_ids=clinical_graph_v2_node_ids,
        )
    concept_value_valid_masks: dict[str, torch.Tensor] | None = None
    if config is not None and config.semantic.formal_p0b is not None:
        raw_hashes = _batch_list(batch, "effective_report_sha256", batch_size)
        if any(not value for value in raw_hashes):
            raise ValueError("Formal P0-B batch lacks effective_report_sha256; metadata target fallback is forbidden.")
        formal_targets, formal_masks, formal_value_masks = load_formal_p0b_batch_targets(
            config.semantic.formal_p0b,
            [str(item) for item in batch["image_id"]],
            raw_hashes,
        )
        concept_targets = {key: torch.as_tensor(value) for key, value in formal_targets.items()}
        concept_valid_masks = {key: torch.as_tensor(value, dtype=torch.bool) for key, value in formal_masks.items()}
        concept_value_valid_masks = {key: torch.as_tensor(value, dtype=torch.bool) for key, value in formal_value_masks.items()}
    modalities_raw = _batch_list(batch, "modality", batch_size)
    modalities = [
        "mammography" if m.strip().lower() in {"mammography", "mammo"} else m
        for m in modalities_raw
    ]

    modality_vocab = ("mammography", "mri", "ultrasound")
    if config is not None and config.model.modality_vocab:
        modality_vocab = config.model.modality_vocab
    _mod_to_idx = {m: i for i, m in enumerate(modality_vocab)}
    modality_ids = torch.tensor(
        [_mod_to_idx.get(m.strip().lower(), 0) for m in modalities],
        dtype=torch.long,
        device=device,
    )

    # Load patch_gaze_weight if available in batch
    patch_gaze_weight = batch.get("patch_gaze_weight")
    if patch_gaze_weight is not None and isinstance(patch_gaze_weight, torch.Tensor):
        patch_gaze_weight = patch_gaze_weight.to(device=device)

    high_conf_patch_prior = batch.get("high_conf_patch_prior")
    if high_conf_patch_prior is not None and isinstance(high_conf_patch_prior, torch.Tensor):
        high_conf_patch_prior = high_conf_patch_prior.to(device=device)

    valid_content_patch_mask = batch.get("valid_content_patch_mask")
    if valid_content_patch_mask is not None and isinstance(valid_content_patch_mask, torch.Tensor):
        valid_content_patch_mask = valid_content_patch_mask.to(device=device, dtype=torch.bool)

    return Stage1JointBatch(
        image=batch["image"],
        attention_map=batch["attention_map"],
        high_conf_mask=batch["high_conf_mask"],
        image_ids=[str(item) for item in batch["image_id"]],
        dataset_ids=_batch_list(batch, "dataset_id", batch_size),
        case_ids=_batch_list(batch, "case_id", batch_size),
        manifest_row_indices=manifest_row_indices,
        teacher_latent_paths=[str(item) for item in batch["teacher_latent_path"]],
        teacher_source_types=[str(item) for item in batch["teacher_latent_source_type"]],
        prompts=prompts,
        prompt_warnings=prompt_warnings,
        concept_targets={
            head_name: target.to(device=device)
            for head_name, target in concept_targets.items()
        },
        concept_valid_masks={
            head_name: mask.to(device=device)
            for head_name, mask in concept_valid_masks.items()
        },
        modalities=modalities,
        modality_ids=modality_ids,
        gaze_supervision_sources=_batch_list(batch, "gaze_supervision_source", batch_size),
        audit_statuses=_batch_list(batch, "audit_status", batch_size),
        attention_map_paths=_batch_list(batch, "attention_map_path", batch_size),
        high_conf_mask_paths=_batch_list(batch, "high_conf_mask_path", batch_size),
        patch_gaze_weight_paths=_batch_list(batch, "patch_gaze_weight_path", batch_size),
        prior_statuses=_batch_list(batch, "prior_status", batch_size),
        coverage_ratios=_batch_float_list(batch, "coverage_ratio", batch_size),
        high_conf_area_ratios=_batch_float_list(batch, "high_conf_area_ratio", batch_size),
        inside_ratios=_batch_float_list(batch, "inside_ratio", batch_size),
        patch_gaze_weight=patch_gaze_weight,
        high_conf_patch_prior=high_conf_patch_prior,
        valid_content_patch_mask=valid_content_patch_mask,
        concept_value_valid_masks=concept_value_valid_masks,
        transform_policies=_batch_list(batch, "transform_policy", batch_size),
        transform_spec_checksums=_batch_list(batch, "transform_spec_checksum", batch_size),
        image_transform_geometry_checksums=_batch_list(
            batch, "image_transform_geometry_checksum", batch_size
        ),
        image_transform_geometry_jsons=_batch_list(
            batch, "image_transform_geometry_json", batch_size
        ),
        attention_transform_spec_checksums=_batch_list(
            batch, "attention_transform_spec_checksum", batch_size
        ),
        attention_transform_geometry_checksums=_batch_list(
            batch, "attention_transform_geometry_checksum", batch_size
        ),
        attention_transform_geometry_jsons=_batch_list(
            batch, "attention_transform_geometry_json", batch_size
        ),
        high_conf_mask_transform_spec_checksums=_batch_list(
            batch, "high_conf_mask_transform_spec_checksum", batch_size
        ),
        high_conf_mask_transform_geometry_checksums=_batch_list(
            batch, "high_conf_mask_transform_geometry_checksum", batch_size
        ),
        high_conf_mask_transform_geometry_jsons=_batch_list(
            batch, "high_conf_mask_transform_geometry_json", batch_size
        ),
        clinical_graph_v2_node_values=clinical_graph_v2_node_values,
        clinical_graph_v2_observed_mask=clinical_graph_v2_observed_mask,
        clinical_graph_v2_node_index=clinical_graph_v2_node_index,
        clinical_graph_v2_node_ids=clinical_graph_v2_node_ids,
    )


def default_summary_name(output_dir: Path) -> Path:
    return output_dir / "stage1_joint_train_summary.json"
