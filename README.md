[README.md](https://github.com/user-attachments/files/32889010/README.md)
# Omni Breast Imaging Foundation Model — Stage 1 Pretraining

This repository contains the source code for Stage 1 pretraining of a tri-modal breast imaging foundation model covering **mammography, breast MRI, and breast ultrasound**.

The released code implements a single-stage joint pretraining framework that combines gaze-guided masked visual learning with image–report alignment, semantic supervision, clinical concept learning, and Clinical Graph consistency. The public repository contains the model, data-loading, loss, masking, semantic, graph, validation, checkpoint/resume, and formal training components required by the Stage 1 runtime.

Medical data, patient-level manifests, model weights, private authorization receipts, internal runtime artifacts, and historical experiment files are not distributed with this repository.

## Overview

Stage 1 jointly optimizes several complementary objectives:

- gaze-guided masked visual reconstruction;
- global image–report alignment;
- visible diagnostic evidence–report alignment;
- semantic soft-relation learning;
- clinical concept prediction and concept consistency;
- Clinical Graph consistency.

The formal training path follows a fail-closed design. Required manifests, semantic assets, gaze priors, concept targets, Clinical Graph sidecars, initialization weights, and authorization receipts must be supplied explicitly and must satisfy their corresponding integrity checks before training proceeds.

## Supported Modalities

The Stage 1 runtime supports three breast imaging modalities:

- Mammography
- Magnetic resonance imaging (MRI)
- Ultrasound

Modality-specific image sizes, batch sizes, transforms, and model settings are configured through the Stage 1 YAML configuration.

## Repository Structure

```text
.
├── configs/
│   ├── formal_stage1_v6_1.example.yaml
│   ├── concept_schema.example.yaml
│   ├── semantic_contract.example.yaml
│   └── clinical_graph/
├── scripts/
│   ├── train_stage1_joint_v5.py
│   └── ... formal pretraining support and audit utilities
├── src/
│   └── breast_pretrain/
│       ├── audit/
│       ├── clinical_graph_encoder/
│       ├── clinical_graph_sidecar/
│       ├── data/
│       ├── data_entry/
│       ├── datasets/
│       ├── gaze/
│       ├── losses/
│       ├── models/
│       ├── semantics/
│       ├── teachers/
│       ├── text/
│       ├── train/
│       └── utils/
├── requirements.txt
└── README.md
```

The public release is intentionally limited to the Stage 1 source tree and the configuration assets required to understand and run the released pretraining workflow. Historical data-processing pipelines, private dataset registries, downstream-task code, internal QA tests, and experiment-specific runtime artifacts are maintained separately.

## Environment

Python **3.10 or newer** is recommended.

A CUDA-capable PyTorch environment is required for formal GPU training. Install a PyTorch/torchvision build compatible with the CUDA runtime available on the target system, then install the remaining Python dependencies.

```bash
python -m pip install -r requirements.txt
```

The released source also contains code paths that may require additional packages depending on the selected backbone, image format, or text/teacher backend. In particular, the default Mammo-FM EfficientNet-B5 backbone requires `timm`; DICOM loading requires `pydicom`; and transformer-based text/teacher backends require `transformers`.

Example:

```bash
python -m pip install timm pydicom transformers
```

Exact CUDA, PyTorch, and dependency versions should be selected for the target hardware and local environment.

## Stage 1 Inputs

The formal Stage 1 runtime consumes a **frozen training manifest** together with pre-materialized supervision assets. Raw dataset-folder discovery is not part of the formal training entry path.

A complete formal run may require the following inputs, depending on the enabled configuration:

- frozen Stage 1 image manifest;
- image files referenced by the manifest;
- effective-report prompts and/or frozen report embeddings;
- gaze membership and gaze priors;
- high-confidence patch priors;
- sparse semantic soft labels;
- semantic-unit mappings;
- clinical concept targets and prototypes;
- Clinical Graph case sidecars and graph schema assets;
- pretrained backbone weights;
- formal initialization state;
- integrity and authorization receipts required by the formal runtime.

These assets are intentionally not included in the public repository.

## Configuration

Start from:

```text
configs/formal_stage1_v6_1.example.yaml
```

The example configuration contains placeholders and **cannot authorize a formal run without replacement**. All dataset, model-weight, semantic, gaze, graph, and authorization paths must be supplied by the user.

The configuration supports modality-specific settings such as:

```yaml
image_size_by_modality:
  mammography: [912, 1520]
  mri: [512, 512]
  ultrasound: [512, 512]

batch_size_by_modality:
  mammography: 1
  mri: 4
  ultrasound: 4
```

Project-specific dataset cardinalities are not hard-coded in the public source. Expected counts and integrity values used by formal runs must come from the corresponding frozen configuration or authorization artifacts.

## Training

Inspect the command-line interface first:

```bash
PYTHONPATH=src python scripts/train_stage1_joint_v5.py --help
```

A formal launch uses an explicit configuration and initialization state:

```bash
PYTHONPATH=src python scripts/train_stage1_joint_v5.py \
  --config /path/to/resolved_stage1_config.yaml \
  --init-state /path/to/formal_init_state.pt
```

Optional command-line overrides include the output directory, maximum training steps, maximum sample count, backbone weight path, artifact directory, gaze prior files, and resume checkpoint. See `--help` for the complete interface.

Before trainer construction, the formal control plane validates the configuration and required lineage/integrity contracts. Missing required assets, incompatible authorization state, invalid sampler conditions, or integrity mismatches are rejected rather than silently replaced.

## Multi-GPU Training

The Stage 1 runtime includes Distributed Data Parallel (DDP) support. Formal bucketed sampling is modality- and image-size-aware and validates the rank-level sampling contract before training.

Users should ensure that the resolved configuration matches the intended world size and that all required formal sampler and resume metadata are available when resuming a distributed run.

## Gaze-Guided Masking

The masking path uses pre-materialized gaze supervision to allocate diagnostic evidence between masked reconstruction and visible semantic learning. The formal runtime validates gaze/patch geometry, token counts, valid-content masks, and gaze mass before using gaze-guided objectives.

The public code does not synthesize missing gaze supervision for a formal run.

## Semantic and Clinical Graph Supervision

The semantic branch supports:

- global image–report alignment;
- visible diagnostic evidence alignment;
- sparse semantic soft-label supervision;
- modality-aware clinical concept prediction;
- concept-consistency learning;
- Clinical Graph consistency.

Clinical Graph schema assets distributed with the repository are located under:

```text
configs/clinical_graph/tri_modal_clinical_graph_v2/
```

The graph pathway is used as structured semantic supervision. The released formal configuration explicitly forbids direct image-region-to-graph-node alignment.

## Checkpoints and Resume

Formal checkpoints retain the state required for exact training continuation, including the model, semantic branch, optimizer, scheduler, AMP scaler, mask-regression state, sampler position, rank-specific random-number-generator states, and associated lineage information.

Resume logic validates the checkpoint and its bound runtime/configuration state before restoring training.

The public Stage 1 release does not include historical downstream-export utilities or model checkpoints.

## Reproducibility and Fail-Closed Validation

The formal runtime contains validation for configuration structure, manifest and asset bindings, semantic contracts, Clinical Graph inputs, DDP sampler behavior, checkpoint/resume state, and other Stage 1 entry conditions.

The example configuration is intended to document the required interface. Reproducing a formal run still requires the corresponding frozen data assets and authorization metadata, which are not distributed because they may contain dataset-specific or restricted information.

## Data and Privacy

No medical images, DICOM files, patient-level manifests, private clinical reports, model checkpoints, private storage paths, credentials, or internal infrastructure information are included in this source release.

Users are responsible for obtaining all datasets independently and for complying with the licenses, data-use agreements, institutional approvals, and privacy requirements associated with those datasets.

## Scope of This Release

This repository focuses on **Stage 1 pretraining source code**. It does not include:

- source medical datasets;
- private dataset registries or patient identifiers;
- Stage 1 pretrained weights;
- historical experiment outputs;
- internal runtime receipts;
- internal QA test suites;
- downstream-task experiment code;
- historical checkpoint-export utilities.

## Citation

A formal citation entry can be added here after the associated manuscript or public preprint is available.

```bibtex
@article{breast_foundation_model,
  title   = {Tri-modal Breast Imaging Foundation Model},
  author  = {Anonymous},
  journal = {To appear},
  year    = {2026}
}
```

Replace the placeholder citation with the final bibliographic information before archival release.

## License

Use of this repository is subject to the license distributed with the public GitHub release. Third-party models, datasets, and external assets remain subject to their original licenses and terms of use.
