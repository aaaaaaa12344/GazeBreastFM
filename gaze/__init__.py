"""Gaze-related modules."""

from breast_pretrain.gaze.attention_map import (
    make_high_confidence_mask,
    points_to_soft_attention_map,
    summarize_trajectory_quality,
)
from breast_pretrain.gaze.prior_builder import build_gaze_priors_from_manifest

__all__ = [
    "build_gaze_priors_from_manifest",
    "make_high_confidence_mask",
    "points_to_soft_attention_map",
    "summarize_trajectory_quality",
]
