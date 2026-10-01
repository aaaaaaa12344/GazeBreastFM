# breast_pretrain.data_entry — Dataset Entry / Stage 0 input preparation tools.
#
# This package is NOT Stage 1 pre-training. It produces:
#   - full-resolution B-mode / ROI PNGs
#   - authoritative image/lesion/case manifests
#   - materialization receipts and QC assets
#
# Downstream consumers (unchanged):
#   - scripts/build_diffeye_image_cache.py  → letterbox 224×224 RGB
#   - diffeye/diffeye_infer/                 → DiffEye inference
#   - pre_train Stage 1                      → formal pretraining (only after contract gate)

# Keep this package import-light: geometry audits must not require Torch merely
# because image materialization modules are available below this namespace.
