from __future__ import annotations

import numpy as np
import torch

from breast_pretrain.train.masked_latent_smoke import build_patch_mask_sampling_priors


def build_patch_gaze_weight_array(
    attention_map: np.ndarray,
    high_conf_mask: np.ndarray,
    *,
    patch_size: int,
) -> np.ndarray:
    attention_tensor = torch.from_numpy(np.asarray(attention_map, dtype=np.float32)).view(1, 1, *attention_map.shape)
    high_conf_tensor = torch.from_numpy(np.asarray(high_conf_mask, dtype=np.float32)).view(1, 1, *high_conf_mask.shape)
    patch_scores, high_conf_scores = build_patch_mask_sampling_priors(
        attention_map=attention_tensor,
        high_conf_mask=high_conf_tensor,
        patch_size=patch_size,
    )
    patch_weight = (patch_scores + high_conf_scores).squeeze(0).detach().cpu().numpy()
    total = float(patch_weight.sum())
    if total > 0.0:
        patch_weight = patch_weight / total
    return patch_weight.astype(np.float32, copy=False)
