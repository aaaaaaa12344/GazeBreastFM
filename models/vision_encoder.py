from __future__ import annotations

from breast_pretrain.models.minimal_student_encoder import MinimalPatchStudentEncoder
from breast_pretrain.models.visual_encoder_factory import LocalTorchCheckpointVisualEncoder


# Backward-compatible alias for old smoke tests. Formal Stage 1 training should
# build encoders through visual_encoder_factory.build_visual_encoder.
VisionEncoder = MinimalPatchStudentEncoder

__all__ = ["LocalTorchCheckpointVisualEncoder", "VisionEncoder"]
