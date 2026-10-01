"""Loss modules."""

from breast_pretrain.losses.concept_consistency import (
    concept_consistency_regularization_loss,
)
from breast_pretrain.losses.global_alignment import global_image_case_alignment_loss
from breast_pretrain.losses.graph_consistency import (
    compute_graph_consistency_loss,
    compute_graph_encoder_consistency_loss,
    compute_graph_prototype_alignment_loss,
)
from breast_pretrain.losses.reconstruction import (
    masked_teacher_latent_reconstruction_loss,
)
from breast_pretrain.losses.semantic_soft_contrastive import (
    semantic_soft_contrastive_loss,
)
from breast_pretrain.losses.total_loss import total_joint_pretrain_loss
from breast_pretrain.losses.visible_alignment import visible_latent_alignment_loss

__all__ = [
    "compute_graph_consistency_loss",
    "compute_graph_encoder_consistency_loss",
    "compute_graph_prototype_alignment_loss",
    "concept_consistency_regularization_loss",
    "global_image_case_alignment_loss",
    "masked_teacher_latent_reconstruction_loss",
    "semantic_soft_contrastive_loss",
    "total_joint_pretrain_loss",
    "visible_latent_alignment_loss",
]
