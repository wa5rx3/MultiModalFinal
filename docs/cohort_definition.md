# Cohort Definition

## Current stage
ED-linked, imaging-time anchored cohort for pneumonia prediction.

## Clinical setting
Emergency department prediction at chest X-ray time `t0`.

## Active index policy
- Start from MIMIC-CXR-JPG manifest.
- Keep frontal studies only (PA/AP), one image per study with PA-priority.
- Link studies to ED stays and keep only studies with exactly one ED match.
- Require `t0` to be present.

## Current core cohort
File: `artifacts/manifests/cxr_final_ed_cohort.parquet`

- Rows (studies): 81,385
- Subjects: 47,404
- ED stays: 79,346
- Missing `t0`: 0

## Key design choices
- `t0` is parsed from `StudyDate + StudyTime`.
- Predictor timing policy is strict: use information available at or before `t0`.
- Temporal evaluation is patient-level using first observed `t0` per subject.

## Pretraining imaging cohort (new)
File: `artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet`

Built by `src/data/build_image_pretraining_split.py` with subject-level splits:
- `pretrain_train`
- `pretrain_internal_val`
- `exclude_ed_validate`
- `exclude_ed_test`
- optional `exclude_ed_train` when policy is `exclude_all_ed`

Policy options:
- `allow_ed_train`: ED train subjects are allowed in pretraining train
- `exclude_all_ed`: all ED subjects excluded from supervised pretraining

Current recommendation for publication-first pipeline:
- Use `exclude_all_ed` as primary.
- Use `allow_ed_train` as sensitivity analysis.

## Data integrity filtering
In manifest construction, rows with missing image files are excluded.

From the current local build:
- Total rows in raw manifest: 377,110
- Missing image paths: 663 (~0.18%)

This ensures final cohorts only include accessible image files.

## ED pneumonia image evaluation subset (temporal `u_ignore`)
For **image-only** and **multimodal (triage + image)** models aligned with clinical baselines:

- **Image-only table:** `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet`
- **Multimodal table (triage + paths):** `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet`
- **Rows:** 9,137 studies (same keys as `cxr_pneumonia_training_table_u_ignore_temporal.parquet`)
- **Splits (`temporal_split`, rows):** train 7,132 / validate 930 / test 1,075
- **Builder:** `src/data/build_image_pneumonia_finetune_table.py` (ED temporal cohort + `cxr_pneumonia_training_table_u_ignore.parquet`)

## Artifact organization (pointer)
Sensitivity-only trained runs (e.g. overlap-subset clinical models) and smoke tests may live under **`artifacts/archive/`**; canonical cohort parquet paths above remain the reference for reproduction. Details: [`docs/data_versions.md`](data_versions.md).