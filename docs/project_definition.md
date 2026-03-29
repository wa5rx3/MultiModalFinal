# Project Definition

## Working title
Publication-grade multimodal pneumonia detection from chest X-rays and clinical data

## Clinical setting
Emergency department / acute-care style prediction around chest X-ray time.

## Index event
Chest X-ray acquisition time (t0).

## Allowed predictors
Only information available at or before t0.

## Main targets
- T1: radiographic pneumonia
- T2: clinically actionable pneumonia

## Planned comparisons
- image-only model
- clinical-only model
- multimodal fusion (first version: **early fusion**, triage + image — see below)
- missingness-aware multimodal fusion (future)

## Image-only pneumonia (current status)
- **Temporal ED cohort, CheXpert `u_ignore` target, 9,137 labeled studies** — same split as clinical baselines.
- Fine-tuning table: `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet`.
- **Canonical** image fine-tune (DenseNet121, 224px): `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/` (init from `image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt`). Ablations (ImageNet-only, older phase1 runs) → `artifacts/archive/models/from_models_root_2026_03/`.

## Multimodal pneumonia (current status)
- **Same 9,137-study ED temporal split and `u_ignore` target** as clinical and image-only baselines.
- Input table: `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet` (triage features + `image_path`).
- **Canonical** run: early fusion (tabular + image → MLP), DenseNet121 initialized from **`image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt`** — `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/`.
- Older frozen/unfrozen phase1 dirs → `artifacts/archive/models/from_models_root_2026_03/` (and other `archive/models/*` subtrees).
- **Interpretation:** small AUROC point gain vs multilabel-init image-only; **patient-level bootstrap** (§9.8) gives **95% CIs for Δ(multimodal − image)** that **include zero** on AUROC and AUPRC — treat multimodal as **not clearly superior** in this setup. Calibration and Grad-CAM still open. See **`README.md`** and **`docs/current_state.md` §9.8**.

## Current image branch policy
- supervised image pretraining uses subject-level pretraining splits
- ED temporal split integrity is preserved via exclusion policy
- CheXpert multilabel supervision currently uses mask policy:
  - supervise only `0/1`
  - mask out uncertain `-1` and missing labels

## Core methodology principles
- no patient leakage across splits
- no post-t0 features
- reproducible manifests
- temporal validation

## Where to run things
- Ordered commands: [`docs/runbook.md`](runbook.md)
- Full narrative + results: [`docs/current_state.md`](current_state.md)