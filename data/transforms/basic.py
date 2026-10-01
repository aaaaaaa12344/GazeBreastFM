from __future__ import annotations

from breast_pretrain.data.transforms.stage1_transform_spec import ImageSize, normalize_image_size


def describe_joint_pretrain_transform(image_size: ImageSize) -> dict[str, object]:
    height, width = normalize_image_size(image_size)
    return {
        "resize": [int(height), int(width)],
        "normalize_range": [0.0, 1.0],
        "color_mode": "rgb",
    }
