# Clinical Graph V0 Schema

This directory contains the V0 schema assets for the Breast Multimodal Diagnostic Attribute Graph.

## Files

- `breast_multimodal_nodes_v0.csv`: node vocabulary for shared upper concepts and modality-specific mammography, ultrasound, and MRI diagnostic attributes.
- `breast_multimodal_edges_v0.csv`: static edge vocabulary and initial semantic relations between nodes.
- `breast_multimodal_mapping_rules_v0.yaml`: dataset mapping draft for mammography, ultrasound, MRI, DBT, and CESM sources.

## V0 Boundary

V0 is schema-only. It does not create a graph database, does not add Neo4j, does not read medical images, and does not train a model.

The current project plan is V5.1 single-stage gaze-guided masked-semantic joint pretraining. This graph is only a Stage 1 semantic prior and sidecar-generation specification for:

- structured clinical prompt generation
- semantic soft-label generation
- clinical concept targets
- concept consistency regularization targets

It is not an independent Stage 2 graph pretraining design. It also must not be interpreted as image-region to graph-node direct alignment.

## Validation

Run:

```text
python scripts/validate_clinical_graph_schema.py
```

The validator checks required CSV columns, YAML parseability, unique `node_id`, edge endpoint existence, and mapping-rule `node_id` references. It writes:

```text
outputs/clinical_graph_schema_validation/validation_report.json
```

## Why It Is Not Wired Into Trainer Yet

Raw datasets are not Stage 1-ready just because they are useful data assets. Each raw dataset must first pass through Dataset Entry Layer and become a standard Stage 1 bundle. Until the bundle contract and concept target sidecars are generated and audited, the Stage 1 trainer should remain unchanged.
