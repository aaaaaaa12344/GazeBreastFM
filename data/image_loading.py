from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DICOM_SUFFIXES: frozenset[str] = frozenset({".dcm", ".dicom"})


def _dicom_context(path: Path, ds: Any | None = None) -> str:
    transfer_syntax_uid = "<unknown>"
    photometric_interpretation = "<unknown>"
    if ds is not None:
        file_meta = getattr(ds, "file_meta", None)
        transfer_syntax_uid = str(
            getattr(file_meta, "TransferSyntaxUID", None) or "<missing>"
        )
        photometric_interpretation = str(
            getattr(ds, "PhotometricInterpretation", None) or "<missing>"
        )
    return (
        f"path={path}; TransferSyntaxUID={transfer_syntax_uid}; "
        f"PhotometricInterpretation={photometric_interpretation}"
    )


def normalize_dicom_pixels_to_uint8(
    pixels: np.ndarray,
    *,
    photometric_interpretation: str,
) -> np.ndarray:
    """Convert raw DICOM pixel_array to uint8 with MONOCHROME1 inversion.

    This is the canonical implementation used by both BreastImageDataset and
    the source intensity audit script.  Any change here affects both paths.
    """
    array = np.asarray(pixels, dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D DICOM pixel array, got shape {array.shape}.")

    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        finite_values = array[finite_mask]
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        if max_value <= min_value:
            normalized = np.zeros(array.shape, dtype=np.uint8)
        else:
            clipped = np.nan_to_num(array, nan=min_value, posinf=max_value, neginf=min_value)
            scaled = (clipped - min_value) / (max_value - min_value)
            normalized = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)

    if str(photometric_interpretation).strip().upper() == "MONOCHROME1":
        normalized = 255 - normalized
    return normalized


def load_dicom_pil_image(path: Path) -> Image.Image:
    """Decode a DICOM file through VOI LUT → uint8 → PIL RGB.

    Returns a PIL RGB Image whose pixel values are byte-identical to what
    BreastImageDataset feeds into the model.
    """
    try:
        import pydicom

        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
    except Exception as exc:
        raise RuntimeError(
            "Failed to import optional DICOM dependency pydicom for "
            f"{_dicom_context(path)}; error={exc!r}"
        ) from exc

    ds = None
    try:
        ds = pydicom.dcmread(path, force=True)
        pixels = ds.pixel_array
        pixels = apply_voi_lut(pixels, ds)
        photometric_interpretation = str(
            getattr(ds, "PhotometricInterpretation", "") or ""
        )
        normalized = normalize_dicom_pixels_to_uint8(
            pixels,
            photometric_interpretation=photometric_interpretation,
        )
        return Image.fromarray(normalized, mode="L").convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode DICOM image: {_dicom_context(path, ds)}; error={exc!r}"
        ) from exc


def load_image_to_uint8_rgb(image_path: Path) -> np.ndarray:
    """Load any supported image as a uint8 RGB numpy array (H, W, 3).

    DICOM files go through the canonical VOI LUT → uint8 → RGB pipeline.
    PNG / JPEG / etc. use PIL.
    """
    suffix = image_path.suffix.lower()
    if suffix in DICOM_SUFFIXES:
        pil_image = load_dicom_pil_image(image_path)
    else:
        pil_image = Image.open(image_path).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def pil_to_raw_tensor(image: Image.Image, mode: str) -> torch.Tensor:
    """Convert PIL image to normalised float32 tensor [C, H, W] with /255."""
    normalized_mode = "RGB" if mode == "rgb" else "L"
    converted = image.convert(normalized_mode)
    array = np.asarray(converted, dtype=np.float32) / 255.0
    if normalized_mode == "RGB":
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array)


__all__ = [
    "DICOM_SUFFIXES",
    "load_dicom_pil_image",
    "load_image_to_uint8_rgb",
    "normalize_dicom_pixels_to_uint8",
    "pil_to_raw_tensor",
]
