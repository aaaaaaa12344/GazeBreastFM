from __future__ import annotations

import json

from breast_pretrain.teachers import TEACHER_SOURCE_MISSING


def assert_required_teacher_latents(
    require_teacher_latents: bool,
    teacher_latent_summary: dict[str, object],
    context: str,
) -> None:
    if not require_teacher_latents:
        return

    fallback_count = int(teacher_latent_summary.get("fallback_dummy_image_latent_count", 0))
    missing_count = int(
        teacher_latent_summary.get(
            "missing_teacher_latent_count",
            teacher_latent_summary.get(f"{TEACHER_SOURCE_MISSING}_count", 0),
        )
    )
    if fallback_count <= 0 and missing_count <= 0:
        return

    raise RuntimeError(
        "require_teacher_latents=true but teacher latent fallback/missing was detected in "
        f"{context}: "
        + json.dumps(
            {
                "fallback_dummy_image_latent_count": fallback_count,
                "missing_teacher_latent_count": missing_count,
            },
            ensure_ascii=True,
        )
    )
