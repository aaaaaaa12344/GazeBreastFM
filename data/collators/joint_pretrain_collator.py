from __future__ import annotations

from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

_OPTIONAL_PATCH_TENSOR_KEYS = ("patch_gaze_weight", "high_conf_patch_prior")
_CLINICAL_GRAPH_V2_VALUE_KEY = "clinical_graph_v2_node_values"
_CLINICAL_GRAPH_V2_MASK_KEY = "clinical_graph_v2_observed_mask"
_CLINICAL_GRAPH_V2_NODE_INDEX_KEY = "clinical_graph_v2_node_index"
_CLINICAL_GRAPH_V2_NODE_IDS_KEY = "clinical_graph_v2_node_ids"
_CLINICAL_GRAPH_V2_CONTRACT_KEYS = (
    _CLINICAL_GRAPH_V2_VALUE_KEY,
    _CLINICAL_GRAPH_V2_MASK_KEY,
    _CLINICAL_GRAPH_V2_NODE_INDEX_KEY,
    _CLINICAL_GRAPH_V2_NODE_IDS_KEY,
)


def _image_size(sample: dict[str, Any]) -> tuple[int, int]:
    image = sample.get("image")
    if not hasattr(image, "shape") or len(image.shape) < 3:
        raise ValueError("Stage 1 sample image must have shape [C, H, W].")
    return int(image.shape[-2]), int(image.shape[-1])


def _stack_optional_patch_tensor(
    batch: list[dict[str, Any]],
    key: str,
) -> torch.Tensor | None:
    values = [sample.get(key) for sample in batch]
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        return None
    reference = tensors[0]
    stacked: list[torch.Tensor] = []
    for index, value in enumerate(values):
        if value is None:
            stacked.append(torch.zeros_like(reference))
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Stage 1 sample {index} field {key} must be a tensor or None, "
                f"got {type(value).__name__}."
            )
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"Stage 1 sample {index} field {key} shape {tuple(value.shape)} "
                f"does not match batch reference shape {tuple(reference.shape)}."
            )
        stacked.append(value)
    return torch.stack(stacked, dim=0)


def _validate_clinical_graph_v2_batch(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, tuple[str, ...]] | None:
    key_presence = {
        key: [key in sample for sample in batch]
        for key in _CLINICAL_GRAPH_V2_CONTRACT_KEYS
    }
    has_any = any(any(presence) for presence in key_presence.values())
    if not has_any:
        return None
    if any(not all(presence) for presence in key_presence.values()):
        raise ValueError(
            "Clinical Graph V2 batch must provide node values, observed mask, node index, and node ids for every sample."
        )

    reference_values = batch[0][_CLINICAL_GRAPH_V2_VALUE_KEY]
    reference_mask = batch[0][_CLINICAL_GRAPH_V2_MASK_KEY]
    reference_index = batch[0][_CLINICAL_GRAPH_V2_NODE_INDEX_KEY]
    reference_node_ids = batch[0][_CLINICAL_GRAPH_V2_NODE_IDS_KEY]
    if not isinstance(reference_values, torch.Tensor) or reference_values.ndim != 1 or reference_values.dtype != torch.float32:
        raise TypeError("Clinical Graph V2 node values must be rank-1 float32 tensors.")
    if not isinstance(reference_mask, torch.Tensor) or reference_mask.ndim != 1 or reference_mask.dtype != torch.bool:
        raise TypeError("Clinical Graph V2 observed masks must be rank-1 bool tensors.")
    if not isinstance(reference_index, torch.Tensor) or reference_index.ndim != 1 or reference_index.dtype != torch.long:
        raise TypeError("Clinical Graph V2 node index must be a rank-1 int64 tensor.")
    if reference_values.shape != reference_mask.shape or reference_values.shape != reference_index.shape:
        raise ValueError("Clinical Graph V2 values, observed mask, and node index must share the same shape.")
    if not isinstance(reference_node_ids, (list, tuple)):
        raise TypeError("Clinical Graph V2 node ids must be a stable sequence of strings.")
    canonical_node_ids = tuple(str(node_id).strip() for node_id in reference_node_ids)
    if len(canonical_node_ids) != int(reference_index.numel()) or any(not node_id for node_id in canonical_node_ids):
        raise ValueError("Clinical Graph V2 node ids must match the canonical node index length.")
    if len(set(canonical_node_ids)) != len(canonical_node_ids):
        raise ValueError("Clinical Graph V2 node ids must be unique.")
    expected_index = torch.arange(reference_index.numel(), dtype=torch.long)
    if not torch.equal(reference_index.cpu(), expected_index):
        raise ValueError("Clinical Graph V2 node index must be the canonical contiguous [0, N) order.")

    for sample_index, sample in enumerate(batch[1:], start=1):
        values = sample[_CLINICAL_GRAPH_V2_VALUE_KEY]
        mask = sample[_CLINICAL_GRAPH_V2_MASK_KEY]
        node_index = sample[_CLINICAL_GRAPH_V2_NODE_INDEX_KEY]
        node_ids = sample[_CLINICAL_GRAPH_V2_NODE_IDS_KEY]
        if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or tuple(values.shape) != tuple(reference_values.shape):
            raise ValueError(f"Clinical Graph V2 sample {sample_index} node values do not match the batch contract.")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(reference_mask.shape):
            raise ValueError(f"Clinical Graph V2 sample {sample_index} observed mask does not match the batch contract.")
        if not isinstance(node_index, torch.Tensor) or node_index.dtype != torch.long or not torch.equal(node_index.cpu(), reference_index.cpu()):
            raise ValueError(f"Clinical Graph V2 sample {sample_index} node index does not match the batch contract.")
        if tuple(str(node_id).strip() for node_id in node_ids) != canonical_node_ids:
            raise ValueError(f"Clinical Graph V2 sample {sample_index} node ids do not match the batch contract.")
    return reference_index, canonical_node_ids


def joint_pretrain_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Stage 1 collator received an empty batch.")
    image_sizes = {_image_size(sample) for sample in batch}
    if len(image_sizes) > 1:
        raise ValueError(
            "Stage 1 batch contains multiple image sizes and cannot be stacked. "
            "Use batch_policy=bucket_by_modality_and_image_size."
        )
    clinical_graph_v2_contract = _validate_clinical_graph_v2_batch(batch)
    collatable_batch = [
        {
            key: value
            for key, value in sample.items()
            if key not in _OPTIONAL_PATCH_TENSOR_KEYS
            and key != _CLINICAL_GRAPH_V2_NODE_INDEX_KEY
            and key != _CLINICAL_GRAPH_V2_NODE_IDS_KEY
        }
        for sample in batch
    ]
    collated = default_collate(collatable_batch)
    for key in _OPTIONAL_PATCH_TENSOR_KEYS:
        stacked = _stack_optional_patch_tensor(batch, key)
        if stacked is not None:
            collated[key] = stacked
    if clinical_graph_v2_contract is not None:
        clinical_graph_v2_node_index, clinical_graph_v2_node_ids = clinical_graph_v2_contract
        collated[_CLINICAL_GRAPH_V2_NODE_INDEX_KEY] = clinical_graph_v2_node_index.clone()
        collated[_CLINICAL_GRAPH_V2_NODE_IDS_KEY] = clinical_graph_v2_node_ids
    return collated
