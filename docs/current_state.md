# Current Project State

_Last updated: 2026-03-25 — **Terminal snapshot:** CheXpert-stratified JSON exports for image + multimodal **`stronger_lr_v3`** (`artifacts/evaluation/*_normal_vs_abnormal_negatives_stronger_lr_v3.json`); **prediction-behavior** dirs for clinical **`strong_v2`** (`prediction_behavior_clinical_*_strong_v2/`); rounded summary **`artifacts/evaluation/final_results_table.csv`** + **`final_result_note.txt`**. Earlier 2026-03-25 note: pointer to **`scripts/evaluate_normal_vs_abnormal_negatives.py`** / **`docs/runbook.md` §4.9**. Prior stamp: 2026-03-23 — **Clinical triage-only (§9.2–9.3), image fine-tune (§9.6B), multimodal (§9.7), and selected bootstrap comparisons (§9.8)** updated from **`*_phase1_clean_v1`** runs. Legacy runs are kept for comparison._

## 1. Project goal and framing

Build a publication-oriented multimodal pneumonia detection system using:
- chest X-rays (MIMIC-CXR-JPG)
- clinical data (MIMIC-IV / MIMIC-IV-ED)

Core framing:
- ED-based prediction anchored at imaging time `t0`
- strict time-safe predictors (`<= t0`)
- compare image-only, clinical-only, and multimodal models
- evaluate with patient-level temporal splitting and leakage controls

---

## 2. Data sources

### Imaging
- MIMIC-CXR-JPG 2.1.0
- Root: `D:/mimic_data`

### Clinical
- MIMIC-IV (admissions, labevents, d_labitems)
- MIMIC-IV-ED (edstays, triage)

### QC utilities (local integrity checks)
Scripts under `src/qc/` (run as needed after cohort or label changes):
- `qc_cxr_admission_linkage.py`, `qc_cxr_edstay_linkage.py` — admission / ED linkage sanity
- `qc_imaging_cohort.py`, `qc_t0_parsing.py` — imaging / timestamp checks
- `qc_label_balance_by_split.py` — label distribution by split

### Repository layout (after cleanup)

- **`src/`** — pipeline and training code (`data/`, `datasets/`, `models/`, `training/`, `qc/`) plus **`src/evaluation/`** (bootstrap CIs, repair prediction CSVs). There is **no** separate `src/features/` or `src/utils/` package.
- **`configs/`** — `paths.local.example.yaml` / gitignored `paths.local.yaml`; **`configs/experiments/*.yaml`** documents headline hyperparameters (training scripts still use **argparse** only).
- **`tools/audits/`** — ad-hoc audit scripts (not invoked by the main pipeline).
- **`artifacts/manifests/`** — cohort and training-table parquet + companion `*_report.json`.
- **`artifacts/tables/`** — wide lab feature tables and clinical+labs merges.
- **`artifacts/logs/`** — merge/extract reports; **`artifacts/logs/audits/`** for consistency audits when present.
- **`artifacts/models/`** — primary trained runs; see **`artifacts/models/README.md`**. **Overlap-only clinical baselines** and other sensitivity runs live under **`artifacts/archive/models/`** (may be gitignored locally).
- **`artifacts/runs/registry.json`** — index of headline model runs (paths to `metrics.json` / `summary.json`); extend with git SHA and exact commands as you formalize provenance.

**Practical rerun guide:** [`docs/runbook.md`](runbook.md). **Artifact map:** [`docs/data_versions.md`](data_versions.md).

---

## 3. Cohort construction

### 3.1 Raw imaging manifest
- File: `artifacts/manifests/mimic_cxr_manifest.parquet`
- Rows: 377,110
- Studies: 227,835
- Subjects: 65,379
- Missing paths: 663

### 3.2 Primary imaging cohort (frontal, study-level)
- File: `artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet`
- Rows: 217,922
- One row per study
- Includes:
  - `t0` parsed from StudyDate + StudyTime
  - view flags (`is_pa`, `is_ap`)

### 3.3 Final ED-linked cohort
- File: `artifacts/manifests/cxr_final_ed_cohort.parquet`
- Rows: 81,385
- Subjects: 47,404
- ED stays: 79,346
- Each study linked to exactly one ED stay
- Missing `t0`: 0

---

## 4. Labels (CheXpert pneumonia, T1)

- File: `artifacts/manifests/cxr_pneumonia_labels.parquet`

Distribution:
- positive: 4,281
- negative: 4,856
- uncertain: 7,826
- missing: 64,422

Notes:
- merge successful (81384/81385 rows matched)
- missing labels are true blanks, not merge errors
- current CheXpert file in this environment does not include `dicom_id`
- label builder now supports strict preferred merge (`subject_id`, `study_id`, `dicom_id`) and explicit fallback
- current production run uses explicit fallback merge keys: `subject_id`, `study_id`
- fallback mode is recorded in report as `fallback_study_merge_allowed: true`
- conflict QC at fallback key level currently reports:
  - duplicate groups: 0
  - conflicting pneumonia label groups: 0

---

## 5. Training tables

### 5.1 Label-policy tables
- `cxr_pneumonia_training_table_u_ignore.parquet`
  - Rows: 9,137
  - Positive rate: ~0.47
- `cxr_pneumonia_training_table_u_zero.parquet`
  - Rows: 16,963
  - Positive rate: ~0.25

### 5.2 Clinical training table (triage branch, u_ignore)
- `cxr_clinical_pneumonia_training_table_u_ignore.parquet`
- Rows: 9,137

### 5.3 Temporalized training table
- `cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet`
- Split counts:
  - train: 7,132
  - validate: 930
  - test: 1,075

### 5.4 Temporal pneumonia label table (shared eval cohort)
- File: `artifacts/manifests/cxr_pneumonia_training_table_u_ignore_temporal.parquet`
- Rows: 9,137 (same effective labeled set as clinical `u_ignore` branch)
- Split counts (rows): train 7,132 / validate 930 / test 1,075
- Contains: `image_path`, `target`, `temporal_split`, legacy MIMIC `split` dropped in favor of `temporal_split` for evaluation
- Role: canonical row set for **temporal** evaluation across clinical and image branches

### 5.5 ED image pneumonia fine-tuning table
- Builder: `src/data/build_image_pneumonia_finetune_table.py`
- Inputs:
  - `artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet` (provides `temporal_split`, `image_path`, `t0`, view flags)
  - `artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet` (binary `target` from CheXpert pos/neg only)
- Output: `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet`
- Report: `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal_report.json`
- Rows: 9,137; split counts (rows): train 7,132 / validate 930 / test 1,075
- Split counts (subjects): train 6,036 / validate 787 / test 887
- Join: inner merge on `subject_id`, `study_id` with `validate="one_to_one"`
- QC: row keys match `cxr_pneumonia_training_table_u_ignore_temporal.parquet` exactly (same 9,137 studies)

---

## 6. Temporal split

Current policy:
- patient-level temporal split from first observed `t0` per subject
- split labels: train / validate / test
- split then applied to downstream tables using study keys

Status:
- implemented and used in current baseline runs

---

## 7. Clinical triage feature branch (current)

### 7.1 Triage features table
- File: `artifacts/manifests/cxr_ed_triage_features.parquet`
- Rows: 81,385

Feature groups:
- vitals: temperature, heartrate, resprate, o2sat, sbp, dbp
- pain, acuity
- demographics: gender, race
- context: arrival_transport
- chiefcomplaint retained in feature table for future use

Important update:
- `disposition` removed from active clinical baseline features due to post-`t0` leakage risk

**Vitals clipping (in `src/data/build_triage_features.py`):** physiological fields are clipped to plausible ranges before missing flags are derived (e.g. temperature **95.0–105.8 °F**, heartrate 30–220, etc.). Rebuild `cxr_ed_triage_features.parquet` → downstream clinical tables if you change clip bounds.

### 7.2 Model-ready triage table
- File: `artifacts/manifests/cxr_ed_triage_model_table.parquet`

Current preprocessing policy:
- no global imputation at table-build stage
- numeric coercion and categorical cleaning only
- missingness flags retained
- imputation/scaling done inside model pipeline fit on train only

---

## 8. Lab feature branch (new, in progress)

### 8.1 Candidate concept map
- `artifacts/tables/lab_feature_map.json`
- 26 lab concepts mapped to selected itemids

### 8.2 Lab extraction policy
Script: `src/data/extract_labevents_for_cohort.py`

Current extraction behavior:
- reads labevents BigQuery-export shards
- filters to cohort subjects and target itemids
- enforces `charttime <= t0` and `charttime >= t0 - 24h`
- encounter-safe primary matching: `subject_id + hadm_id`
- optional fallback mode exists, but strict mode is default:
  - `--match-mode hadm_only` (default and preferred)
  - `--match-mode hadm_plus_fallback` (sensitivity only)

Current run command (active):
`PYTHONPATH=. python src/data/extract_labevents_for_cohort.py --labevents-dir "D:/mimic_iv/labevents" --cohort "artifacts/manifests/cxr_final_ed_cohort.parquet" --feature-map "artifacts/tables/lab_feature_map.json" --output "artifacts/tables/cohort_labevents_hadm_only.parquet" --report "artifacts/logs/cohort_labevents_hadm_only_report.json" --lookback-hours 24 --match-mode hadm_only`

### 8.3 Hadm-only extraction run (current main run)
- Output table: `artifacts/tables/cohort_labevents_hadm_only.parquet`
- Report: `artifacts/logs/cohort_labevents_hadm_only_report.json`
- Current run stats:
  - `match_mode`: `hadm_only`
  - `after_time_fallback`: 0
  - final rows: 100,662
  - final studies: 6,605
  - final subjects: 5,661
  - final itemids: 28

### 8.4 Hadm-only lab feature table
- `artifacts/tables/cxr_lab_features_hadm_only.parquet`
- Rows: 6,605 studies
- Concepts: 26 numeric lab features + 26 missingness flags
- Sparsity is high for several concepts (examples):
  - `total_protein` ~99.5% missing
  - `crp` ~97.2% missing
  - blood gas variables (`pco2`, `po2`, `base_excess`) ~95.9% missing

### 8.5 Merged clinical+labs training tables (hadm-only branch)
- Full left-merge table:
  - `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet`
  - rows: 9,137
  - contains temporal split labels
  - rows with any non-missing lab concept: 867 (~9.5%)
- Overlap-only subset:
  - `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap.parquet`
  - rows: 867 (subjects: 839)
  - positive rate: ~0.401
  - this is for sensitivity analysis, not primary headline benchmarking

---

## 9. Baseline results

### 9.1 Historical triage logistic baseline (pre leakage-fix)
- Validation AUROC: 0.621
- Validation AUPRC: 0.616
- Test AUROC: 0.618
- Test AUPRC: 0.561

### 9.2 Leakage-corrected triage logistic baseline (temporal `u_ignore`)
Dataset:
- ED-linked cohort
- T1 pneumonia labels (CheXpert)
- `u_ignore` policy
- temporal patient split

Preprocessing:
- no global imputation in tables
- imputation and scaling in sklearn pipeline (fit on train only)
- disposition removed (post-`t0` leakage risk)
- vitals **clipping** at triage feature build; **`prepare_feature_matrix`** fills triage missing flags + **`is_pa`/`is_ap`** and normalizes categoricals (see §15)

**Current reference run — phase1 clean v1 (2026-03)**  
- Output dir: `artifacts/models/clinical_baseline_u_ignore_temporal_phase1_clean_v1` (`metrics.json`)
- Validation (*n* = 930): AUROC **0.627**, AUPRC **0.627**, Accuracy **0.598**, F1 **0.572**
- Test (*n* = 1075): AUROC **0.605**, AUPRC **0.547**, Accuracy **0.577**, F1 **0.521**

**Earlier committed run** (same table policy era, pre–phase1 code path): `artifacts/models/clinical_baseline_u_ignore_temporal` — validation AUROC 0.619 / AUPRC 0.617; test AUROC **0.595**, AUPRC **0.540**.

Interpretation:
- performance dropped after leakage correction vs historical §9.1; phase1 retrain nudges test AUROC/AUPRC slightly vs the older committed logistic run

### 9.3 Clinical XGBoost baseline (triage-only)
**Current reference run — phase1 clean v1 (2026-03)**  
- Output dir: `artifacts/models/clinical_xgb_u_ignore_temporal_phase1_clean_v1` (`metrics.json`, `config.json`)
- Training: up to 1000 trees, **`eval_metric="aucpr"`**, **`early_stopping_rounds=30`** (CLI), **`best_iteration` = 45**, **`best_score` (val AUPRC) ≈ 0.631**
- Validation (*n* = 930): AUROC **0.629**, AUPRC **0.632**, Accuracy **0.587**, F1 **0.573**
- Test (*n* = 1075): AUROC **0.611**, AUPRC **0.559**, Accuracy **0.584**, F1 **0.535**

**Earlier run** (default output dir, pre–early-stopping-on-estimator snapshot): `artifacts/models/clinical_xgb_u_ignore_temporal` — test AUROC **0.588**, AUPRC **0.546** (terminal log from same cohort).

Observation:
- with phase1 preprocessing + val-AUPRC early stopping, XGB is **slightly ahead** of logistic on test AUROC/AUPRC vs the older committed pair; gains are modest

### 9.4 Clinical+labs baselines (hadm-only full temporal table, primary)
Dataset:
- `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet`
- Rows: 9,137
- Split counts: train 7,132 / validate 930 / test 1,075
- Labs merged by left join (primary analysis keeps full cohort)

Logistic regression (triage + labs):
- Validation: AUROC 0.621, AUPRC 0.613, Accuracy 0.597, F1 0.576
- Test: AUROC 0.591, AUPRC 0.533, Accuracy 0.577, F1 0.513

XGBoost (triage + labs):
- Validation: AUROC 0.602, AUPRC 0.609, Accuracy 0.562, F1 0.552
- Test: AUROC 0.598, AUPRC 0.545, Accuracy 0.567, F1 0.519

Interpretation:
- labs do not show a strong, consistent gain on the full 9,137-row table
- small test AUROC improvement appears for XGBoost vs triage-only XGBoost
- logistic regression with labs is mixed and slightly weaker on test AUROC/AUPRC vs triage-only logistic

### 9.5 Overlap-only sensitivity runs (secondary, not headline)
Dataset:
- `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap.parquet`
- Rows: 867 (train 667 / validate 90 / test 110)

Trained **overlap-only** clinical models (logistic / XGB, triage ± labs) are **archived** under `artifacts/archive/models/overlap_sensitivity/` so they are not confused with the primary temporal 9,137-row baselines in `artifacts/models/`. Overlap **tables** remain in `artifacts/tables/` for reproducibility.

Triage-only baselines on overlap subset:
- Logistic: val AUROC 0.534 / test AUROC 0.629 ; val AUPRC 0.484 / test AUPRC 0.545
- XGBoost: val AUROC 0.489 / test AUROC 0.629 ; val AUPRC 0.456 / test AUPRC 0.538

Triage+labs baselines on overlap subset:
- Logistic + labs: val AUROC 0.555 / test AUROC 0.591 ; val AUPRC 0.515 / test AUPRC 0.484
- XGBoost + labs: val AUROC 0.505 / test AUROC 0.695 ; val AUPRC 0.481 / test AUPRC 0.596

Interpretation:
- overlap-only conclusions are unstable due to small validation/test sample sizes
- use overlap runs as sensitivity checks only, not for primary claims

### 9.6 Image-only pneumonia (DenseNet121, temporal `u_ignore` cohort)
Dataset: `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet` (9,137 rows; splits as in §5.5).

Trainer: `src/training/train_image_pneumonia_finetune.py`  
Dataset class: `src/datasets/cxr_binary_dataset.py`

Shared training settings (phase1 runs): CLI requested 10 epochs with patience 5, batch 16, lr 1e-4, weight decay 1e-4, **image size 224**, seed 42, CUDA. Early stopping on **validation AUPRC** selected earlier best epochs.

**A) ImageNet init only** (no CheXpert multilabel checkpoint)  
- Output dir: `artifacts/models/image_pneumonia_finetune_densenet121_imagenet_only`
- Validation: AUROC **0.697**, AUPRC **0.702**, accuracy **0.653**, F1 **0.616**
- Test: AUROC **0.699**, AUPRC **0.674**, accuracy **0.652**, F1 **0.598**

**B) ImageNet init + backbone from multilabel pretrain (phase1 clean v1)**  
- Loads `features.*` from `artifacts/models/image_multilabel_pretrain_densenet121_main/checkpoints/best.pt` (14-class head not loaded).
- Output dir: `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_phase1_clean_v1`
- Early stopping summary: best epoch **2**, epochs completed **7** (`summary.json`)
- Validation: AUROC **0.735**, AUPRC **0.738**, accuracy **0.675**, F1 **0.668**
- Test: AUROC **0.741**, AUPRC **0.720**, accuracy **0.677**, F1 **0.655**

Legacy comparison:
- Earlier output dir: `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal`
- Test: AUROC **0.743**, AUPRC **0.726**

Interpretation (headline):
- On this cohort and setup, **image-only test AUROC (~0.74 with multilabel-init) is higher than triage-only clinical baselines (~0.61 test AUROC with phase1 XGB, ~0.605 logistic)** under the **same temporal split and 9,137 labeled rows**.
- **Multilabel pretraining improves** fine-tune vs ImageNet-only on test (§9.6); **patient-level bootstrap** supports a **positive** AUROC and AUPRC difference vs ImageNet-only (§9.8).
- Multimodal (triage + frozen multilabel backbone) vs multilabel-init image-only: see **§9.7** and **§9.8** — bootstrap CIs for **Δ(multimodal − image)** cross zero on both AUROC and AUPRC.
- **Model weights** (`.pt`) are typically **gitignored**; reproducibility relies on **config.json**, **history.json**, **summary.json**, **`tabular_preprocessor.joblib`** (train-fitted sklearn pipeline), and re-running with the same table + seeds.

### 9.7 Multimodal pneumonia (triage + image, temporal `u_ignore`)
Dataset: `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet` (9,137 rows; triage features + `image_path` + `target` + `temporal_split`).

Trainer: `src/training/train_multimodal_pneumonia.py`  
Model: `src/models/multimodal_model.py` — DenseNet121 **image** branch + **tabular MLP** (sklearn `ColumnTransformer` on triage columns, **fit on train only**) + **fusion MLP** → binary logit.

**Reference run (frozen image backbone, phase1 clean v1):**
- Image backbone weights: `features.*` from `artifacts/models/image_multilabel_pretrain_densenet121_main/checkpoints/best.pt` (14-way CheXpert head not used).
- Flags: **`--freeze-image-backbone`** (only tabular + fusion trained end-to-end).
- Hyperparameters: requested 10 epochs with patience 5; early stop at epoch 8 (best epoch 3), batch 16, lr 1e-4, weight decay 1e-4, image size 224, seed 42, CUDA; `BCEWithLogitsLoss` with **train-derived `pos_weight`**.
- Output dir: `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_clean_v1`
- **Tabular reproducibility:** each training run writes **`tabular_preprocessor.joblib`** in the output dir (same `ColumnTransformer` as used for train/val/test in that run). `config.json` records `tabular_preprocessor_file`, column lists, and `sklearn_version`; `summary.json` repeats the preprocessor path.

**Metrics (from `summary.json` after reloading `checkpoints/best.pt` by val loss):**

| Split | AUROC | AUPRC | Accuracy | F1 |
|--------|--------|--------|----------|-----|
| Validation (930) | **0.727** | **0.746** | 0.681 | 0.643 |
| Test (1075) | **0.753** | **0.723** | 0.691 | 0.645 |

**Comparison (same test n=1075, point estimates):**
- **Image-only (multilabel-init, phase1 clean v1 §9.6B):** test AUROC **0.741**, AUPRC **0.720**
- **This multimodal (frozen backbone, phase1 clean v1):** test AUROC **0.753**, AUPRC **0.723** → modest point-estimate lift on both metrics. **Bootstrap (§9.8):** ΔAUROC 95% CI crosses zero; ΔAUPRC CI crosses zero — still **not** strong evidence of multimodal superiority in this sample.

**Unfrozen sensitivity run (phase1_unfrozen_v1):**
- Command omitted `--freeze-image-backbone` and used lower LR (`--lr 1e-5`) with the same temporal table and checkpoint init.
- Output dir: `artifacts/archive/models/sensitivity_unfrozen_phase1/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_unfrozen_v1`
- Early stopping summary: best epoch **6**, epochs completed **10**
- Validation: AUROC **0.744**, AUPRC **0.764**, accuracy **0.685**, F1 **0.647**
- Test: AUROC **0.761**, AUPRC **0.737**, accuracy **0.702**, F1 **0.656**

**Unfrozen seed sensitivity runs (new):**
- `phase1_unfrozen_seed43_v1` (seed 43): best epoch **5**; test AUROC **0.751**, AUPRC **0.732**, accuracy **0.687**, F1 **0.652**
- `phase1_unfrozen_seed44_v1` (seed 44): best epoch **3**; test AUROC **0.749**, AUPRC **0.726**, accuracy **0.687**, F1 **0.663**
- Compared with `phase1_unfrozen_v1` (seed 42), these seed reruns show similar AUROC/AUPRC rank-order and keep performance above image-only point estimates.

Example command:
```bash
PYTHONPATH=. python src/training/train_multimodal_pneumonia.py \
  --input-table "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet" \
  --image-backbone-checkpoint "artifacts/models/image_multilabel_pretrain_densenet121_main/checkpoints/best.pt" \
  --output-dir "artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_clean_v1" \
  --epochs 10 --patience 5 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 --image-size 224 --num-workers 4 --seed 42 \
  --freeze-image-backbone
```

### 9.8 Patient-level bootstrap uncertainty (test set)

**Scripts**
- `src/evaluation/bootstrap_eval.py` — resamples **`subject_id`** with replacement (cluster bootstrap), recomputes AUROC/AUPRC per replicate; optional **paired delta** (Model A − Model B) with **inner merge** on `subject_id` (+ `study_id` when present). Skips replicates where only one class appears.
- `src/evaluation/repair_prediction_ids.py` — image fine-tune (and similar) **`test_predictions.csv` / `val_predictions.csv`** files may contain only `target` + `pred_prob` in **table row order**; this tool copies **`subject_id`, `study_id`, `dicom_id`** from `cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet` for a given `temporal_split` so bootstrap and alignment can use patient IDs. **Requires equal row counts** and matching **`target`** when both sides carry it.

**Convention:** repaired files are named e.g. `test_predictions_with_ids.csv` next to the original model dir.

**Outputs:** JSON summaries under **`artifacts/evaluation/`** (local); optional per-replicate CSVs with `--save-bootstrap-csv`.

**Runs executed (n_bootstrap = 2000, seed = 42)** — see JSON for full structure:

| Comparison | Δ AUROC (A − B) mean [2.5%, 97.5%] | p(Δ>0) | Δ AUPRC mean [2.5%, 97.5%] | p(Δ>0) |
|------------|-------------------------------------|--------|------------------------------|--------|
| Multimodal − multilabel image | **0.0055** [−0.0117, **0.0223**] | 0.733 | **−0.0066** [−0.0300, 0.0161] | 0.296 |
| Multilabel image − ImageNet-only | **0.0448** [0.0238, 0.0669] | 1.000 | **0.0532** [0.0248, 0.0826] | 1.000 |
| Multilabel image − clinical logistic | **0.1485** [0.1103, 0.1871] | 1.000 | **0.1846** [0.1422, 0.2285] | 1.000 |
| Multimodal (phase1) − image multilabel-init (phase1) | **0.0129** [−0.0051, 0.0303] | 0.923 | **0.0034** [−0.0197, 0.0259] | 0.614 |
| Multimodal (phase1) − clinical XGB (phase1) | **0.1427** [0.1060, 0.1782] | 1.000 | **0.1629** [0.1206, 0.2043] | 1.000 |
| Image (phase1) − clinical XGB (phase1) | **0.1298** [0.0917, 0.1674] | 1.000 | **0.1595** [0.1159, 0.2035] | 1.000 |
| Image (phase1) − clinical logistic (phase1) | **0.1353** [0.0975, 0.1738] | 1.000 | **0.1712** [0.1281, 0.2137] | 1.000 |
| Multimodal (phase1) − clinical logistic (phase1) | **0.1481** [0.1135, 0.1844] | 1.000 | **0.1746** [0.1332, 0.2142] | 1.000 |
| Multimodal unfrozen (phase1) − image (phase1) | **0.0205** [0.0069, 0.0349] | 0.997 | **0.0171** [−0.0020, 0.0353] | 0.959 |
| Multimodal unfrozen (phase1) − multimodal frozen (phase1) | **0.0076** [−0.0056, 0.0212] | 0.865 | **0.0137** [−0.0030, 0.0312] | 0.941 |
| Multimodal unfrozen seed43 (phase1) − image (phase1) | **0.0103** [−0.0046, 0.0253] | 0.904 | **0.0122** [−0.0079, 0.0314] | 0.896 |
| Multimodal unfrozen seed44 (phase1) − image (phase1) | **0.0086** [−0.0050, 0.0221] | 0.894 | **0.0057** [−0.0118, 0.0233] | 0.736 |

**Files (typical):**
- `artifacts/evaluation/bootstrap_multimodal_vs_image.json`
- `artifacts/evaluation/bootstrap_image_pretrain_vs_imagenet.json`
- `artifacts/evaluation/bootstrap_image_vs_clinical.json`
- `artifacts/evaluation/bootstrap_multimodal_vs_image_phase1_clean_v1.json`
- `artifacts/evaluation/bootstrap_multimodal_vs_xgb_phase1_clean_v1.json`
- `artifacts/evaluation/bootstrap_image_vs_xgb_phase1_clean_v1.json`
- `artifacts/evaluation/bootstrap_image_vs_logistic_phase1_clean_v1.json`
- `artifacts/evaluation/bootstrap_multimodal_vs_logistic_phase1_clean_v1.json`
- `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_vs_image_phase1_v1.json`
- `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_vs_frozen_phase1_v1.json`
- `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_seed43_vs_image_phase1_v1.json`
- `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_seed44_vs_image_phase1_v1.json`

**Caveats:** `p(Δ>0)` is the fraction of bootstrap replicates with positive delta (rough one-sided Monte Carlo proportion), **not** a formal p-value against a pre-specified null unless you design it as such. **Calibration** and **multiple comparisons** are still open.

**Stale pairing:** the row **Multilabel image − clinical logistic** uses clinical predictions from **`clinical_baseline_u_ignore_temporal`** (§9.2 pre–phase1). Use the phase1 rows (**Image − clinical logistic phase1** and **Multimodal − clinical logistic phase1**) for current clinical-code comparisons.

**Additional stale pairing:** legacy row **Multimodal − multilabel image** uses older non-phase1 dirs. Prefer the phase1 row above for current image/multimodal codepath.

### 9.9 Calibration analysis tooling (implemented)

Script:
- `src/evaluation/calibration_analysis.py`

Current capabilities:
- Reads one or more prediction CSVs (`target`, `pred_prob`) and computes:
  - **Brier score**
  - **ECE** (expected calibration error; equal-width bins)
  - **MCE** (maximum calibration error)
- Optional bootstrap CIs for Brier and ECE (`--bootstrap --n-bootstrap 2000 --bootstrap-seed 42`).
- Writes:
  - `calibration_metrics.json`
  - `calibration_summary.csv`
  - per-model bin tables: `*_bins.csv`
  - per-model reliability plots: `*_reliability.png`
  - combined plot: `reliability_diagram_all_models.png`
- Default model map targets phase1 prediction CSVs (clinical logistic, clinical XGB, image-only phase1, multimodal unfrozen phase1).

Status in this snapshot:
- Calibration tooling is implemented and ready; bootstrap-focused discrimination results are already in §9.8.
- Keep calibration interpretation with the same caution on repeated comparisons and model-family mixing (legacy vs phase1 paths).

Calibration run executed (terminal, 2026-03):
- Command: `PYTHONPATH=. python src/evaluation/calibration_analysis.py --bootstrap --n-bootstrap 2000 --bootstrap-seed 42`
- Output dir: `artifacts/evaluation/calibration/`
- Files written:
  - `calibration_metrics.json`, `calibration_summary.csv`
  - `reliability_diagram_all_models.png`
  - per-model: `clinical_logistic_*`, `clinical_xgboost_*`, `image_only_densenet121_*`, `multimodal_unfrozen_*`

Phase1 test-set calibration summary (n=1075 each):

| Model | Brier (95% CI) | ECE (95% CI) | MCE |
|-------|-----------------|--------------|-----|
| Clinical Logistic | 0.2423 [0.2357, 0.2492] | 0.0375 [0.0230, 0.0700] | 0.4888 |
| Clinical XGBoost | 0.2401 [0.2348, 0.2454] | 0.0397 [0.0190, 0.0697] | 0.1900 |
| Image-only DenseNet121 | 0.2095 [0.1979, 0.2210] | 0.0723 [0.0508, 0.1006] | 0.1633 |
| Multimodal Unfrozen | 0.1983 [0.1855, 0.2115] | 0.0458 [0.0333, 0.0753] | 0.1065 |

Interpretation notes:
- On Brier score, multimodal unfrozen is best among the default phase1 models in this run.
- ECE ranking is not identical to Brier (image has strong discrimination but higher ECE in this binning setup).
- Logistic MCE is inflated by sparse/extreme bins; review `*_bins.csv` and per-model reliability plots before making calibration claims.

### 9.10 Decision curve analysis (DCA) (implemented + run)

Script:
- `src/evaluation/decision_curve_analysis.py`

Current capabilities:
- Inputs: one or more prediction CSVs via repeated `--model "NAME" path/to/test_predictions.csv` (requires shared row order and identical `target` values).
- Computes threshold-wise:
  - net benefit
  - standardized net benefit
  - confusion-derived metrics at requested clinical thresholds (`sensitivity`, `specificity`, `ppv`, `npv`, etc.).
- Baselines: `treat_all` and `treat_none`.
- Outputs under `artifacts/evaluation/dca/`:
  - `summary.json`
  - `decision_curve.png`
  - `decision_curve_standardized.png`
  - `decision_curve_all_models.csv`
  - `treat_all_curve.csv`, `treat_none_curve.csv`
  - per-model: `*_curve.csv`, `*_threshold_metrics.csv`

Run executed (terminal, 2026-03):
- Command used phase1 default comparison set:
  - Clinical Logistic
  - Clinical XGBoost
  - Image-only DenseNet121
  - Multimodal Unfrozen
- Requested threshold metrics: `0.1, 0.2, 0.3, 0.5, 0.8`
- Cohort summary: `n=1075`, prevalence `0.4530`, DCA thresholds `0.01–0.99` (99 points)

Selected threshold snapshots (net benefit):

| Threshold | Clinical Logistic | Clinical XGBoost | Image-only | Multimodal Unfrozen |
|-----------|-------------------|------------------|------------|---------------------|
| 0.3 | 0.2206 | 0.2186 | 0.2481 | **0.2582** |
| 0.5 | 0.0298 | 0.0372 | 0.1302 | **0.1553** |
| 0.8 | -0.0214 | 0.0009 | 0.0102 | **0.0177** |

Interpretation notes:
- In this run, image and multimodal dominate clinical models in net benefit across practical thresholds.
- Multimodal unfrozen is highest among the four models at key thresholds (0.3/0.5/0.8).
- Always inspect curves against `treat_all` / `treat_none` before threshold recommendations.

---

## 10. Key decisions so far

- Use ED-linked cohort as main dataset
- Use frontal study-level CXRs
- Define time anchor `t0` from StudyDate/StudyTime
- Use triage as first clinical feature set
- Use CheXpert pneumonia as initial target (T1)
- Use `u_ignore` as primary label policy
- Use patient-level temporal split for evaluation
- Use train-only fitted preprocessing for clinical baselines
- First multimodal model: **early fusion** (concat image embedding + preprocessed triage) with optional **frozen** CheXpert-multilabel backbone (§9.7)

---

## 11. Current limitations / risks

- CheXpert pneumonia has large missing fraction in ED-linked cohort
- `u_ignore` yields relatively small effective labeled sample (9,137)
- Label merge currently relies on explicit study-level fallback due to local CheXpert schema (no `dicom_id` column)
- **Image-only** (§9.6) and **multimodal triage+image** (§9.7) are benchmarked on the temporal ED subset; **labs-in-multimodal** and **missingness-aware fusion** are not the current headline model
- Lab availability in labeled cohort is limited (~9.5% with any lab signal in hadm-only main table)
- **Source code** now uses **early stopping** for image fine-tune / multimodal (**val AUPRC**), multilabel pretrain (**val masked loss**), and triage XGB (**AUPRC on val**, up to 1000 trees). Artifact set is now **mixed** (some legacy canonical dirs + newer `*_phase1_clean_v1` dirs); check §9 + §15 for which run family each metric line uses.
- **Patient-level bootstrap CIs** for headline test-set comparisons are available (§9.8); calibration tooling now exists (§9.9) and should be run on the exact comparison set you finalize for thesis tables.
- For inference, load **`tabular_preprocessor.joblib`** from the same run as **`checkpoints/best.pt`** (match `sklearn` major version when possible)
- **`train_image_multilabel_pretrain.py`** CLI default **`--image-size` is 224** (aligned with the headline main run); still pass **`--image-size 224`** explicitly if you change defaults locally
- Experiment YAML under `configs/experiments/` is **not** loaded by training scripts yet — use as documentation or wire up a small loader later
- **`artifacts/runs/registry.json`** lists headline runs but does not yet record git SHA / full CLI per entry

### 11.1 Target-specificity risk (manual probes)

**Multimodal (small slice):** top-confidence cases from
`artifacts/archive/models/sensitivity_unfrozen_phase1/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_unfrozen_v1/test_predictions.csv`
against full CheXpert labels (`D:/mimic-cxr-2.0.0-chexpert.csv.gz`) on a small, non-random subset:
- top 50 multimodal false positives (`target=0`) by `pred_prob`
- top 50 multimodal true positives (`target=1`) by `pred_prob`

Using non-pneumonia abnormality columns (`Atelectasis`, `Edema`, `Pleural Effusion`, `Consolidation`, `Lung Opacity`):
- FP subset: `any_abnormal` = **0.52** (26/50)
- TP subset: `any_abnormal` = **0.56** (28/50)

**Multimodal (full test, CheXpert-stratified negatives — same merge / `any_abnormal` definition as image block below):**
- Predictions: `artifacts/archive/models/sensitivity_unfrozen_phase1/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_unfrozen_v1/test_predictions.csv`
- After merge + dropping rows where **all** five abnormality columns are NaN: **1075 → 686** (same counts as image fine-tune probe: positives **392**, normal negatives **132**, abnormal negatives **162**)
- Positives vs normal negatives (n **524**): AUROC **0.786**, AUPRC **0.917**
- Positives vs abnormal negatives (n **554**): AUROC **0.637**, AUPRC **0.811**
- Contrast (abnormal minus normal negative stratum): ΔAUROC **−0.150**, ΔAUPRC **−0.106**

**Image fine-tune test set, CheXpert-stratified negatives (same five columns, `any_abnormal` = any column == 1):**
- Predictions: `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_phase1_clean_v1/test_predictions.csv`
- CheXpert: `D:/mimic-cxr-2.0.0-chexpert.csv.gz`, merged on `subject_id`, `study_id`
- Rows dropped where **all** five abnormality columns are NaN: **1075 → 686**
- Counts: pneumonia positives **392**; negatives with `any_abnormal` false **132** (“normal” negatives), with `any_abnormal` true **162** (“abnormal” negatives)
- Positives vs normal negatives (n **524**): AUROC **0.767**, AUPRC **0.909**
- Positives vs abnormal negatives (n **554**): AUROC **0.627**, AUPRC **0.812**
- Contrast (abnormal minus normal negative stratum): ΔAUROC **−0.140**, ΔAUPRC **−0.098**

Interpretation:
- This raises a **target-specificity concern**: high model confidence may track broader radiographic abnormality burden, not purely pneumonia-specific signal.
- **Full-test** image and multimodal stratifications use the **same merged rows and group sizes**; on that slice, the **multimodal** run shows a **slightly larger** drop when negatives are restricted to CheXpert “abnormal” (non-pneumonia) vs “normal” (ΔAUROC **−0.150** vs **−0.140**, ΔAUPRC **−0.106** vs **−0.098**).
- CheXpert noise, label timing, and the all-NaN row filter still limit causal claims.
- The multimodal **top-50** probe remains **not a definitive estimate** (ranked subset only).
- Next useful checks: full test-set confusion groups (TP/FP/TN/FN) with CIs and pre-registered thresholds; repeat stratification for other checkpoints (e.g. frozen phase1 clean) if comparing architectures.

**Reproducible CLI (same definitions as above unless merge keys are overridden):** `scripts/evaluate_normal_vs_abnormal_negatives.py` — `--predictions-csv`, `--chexpert-csv`; optional `--merge-on-dicom` when both sides have unique `dicom_id` rows; optional `--output-json`. Command examples: **`docs/runbook.md` §4.9**.

---

## 11.5 Image pretraining split policy (new)

Script:
- `src/data/build_image_pretraining_split.py`

Purpose:
- Build a supervised image pretraining split on the primary frontal cohort while protecting ED temporal evaluation integrity.

Current implemented policies:
- `allow_ed_train`
  - ED `validate`/`test` subjects excluded from supervised pretraining
  - ED `train` subjects allowed in pretraining train
- `exclude_all_ed`
  - all ED subjects excluded from supervised pretraining
  - non-ED subjects split by subject into `pretrain_train` and `pretrain_internal_val`

Current recommendation:
- primary publication-facing pretraining: `exclude_all_ed`
- sensitivity analysis: `allow_ed_train`

Built-in QC checks now enforce:
- allowed temporal split vocabulary (`train`/`validate`/`test`)
- one temporal split per ED subject
- no ED validate/test leakage into trainable pretraining pool
- policy-consistent exclusion sets

---

## 11.6 Image multilabel pretraining pipeline (new)

### Components
- Table builder: `src/data/build_image_multilabel_pretrain_table.py`
- Dataset: `src/datasets/cxr_multilabel_dataset.py`
- Trainer: `src/training/train_image_multilabel_pretrain.py`

### Current table policy
- Input: `mimic_cxr_primary_frontal_with_pretrain_split.parquet`
- Uses only trainable pretraining rows:
  - `pretrain_train`
  - `pretrain_internal_val`
- CheXpert merge keys:
  - preferred: (`subject_id`, `study_id`, `dicom_id`)
  - explicit fallback allowed: (`subject_id`, `study_id`)
- Conflict QC is enforced at the selected merge key level.
- Label value QC is enforced for expected values `{1, 0, -1, NaN}`.

### Current mask policy for training
- Raw label values are preserved in table (`1 / 0 / -1 / NaN`).
- Supervised mask columns are `TRUE` only for `0/1`.
- `-1` (uncertain) and `NaN` are masked out in the first clean run.
- Dataset uses table-provided `*_mask` columns directly.

### Current trainer behavior
- Backbone: DenseNet121 (ImageNet initialization)
- Loss: masked BCE-with-logits
- Splits used:
  - train: `pretrain_train`
  - internal validation: `pretrain_internal_val`
- **Early stopping:** validation **masked loss**; defaults **`--epochs 10`**, **`--patience 5`**; **`checkpoints/best.pt`** is the best val-loss epoch
- Saved outputs:
  - `config.json`
  - per-epoch checkpoints (`*.pt` — often gitignored locally)
  - `history.json`
  - `summary.json`

### Production multilabel run (“main”)
- Output dir: `artifacts/models/image_multilabel_pretrain_densenet121_main`
- Table: `artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet` (45,837 trainable rows: 41,214 train / 4,623 internal val)
- Hyperparameters: 5 epochs, batch 16, lr 1e-4, weight decay 1e-4, **image size 224**, seed 42, CUDA
- Best **validation** masked loss: **0.29635** (epoch **4**; epoch 5 val loss slightly worse)
- Checkpoints: `checkpoints/best.pt` (local disk; not committed to git by default)

Known gap (next improvement):
- Current trainer tracks masked loss only.
- Add per-label and macro AUROC/AUPRC on internal validation for publication-grade monitoring.

### Quick smoke run (optional)
- Output dir (archived): `artifacts/archive/models/smoke_runs/image_multilabel_pretrain_densenet121_quick/` — fast pipeline checks; not a headline run

---

## 12. Next steps (current execution order)

1. ~~Freeze current clinical branch as v1 (leakage-safe baseline reference).~~ — reference metrics in §9.2–9.5
2. ~~Run and stabilize image multilabel pretraining with frozen table/split policy.~~ — **done** (`exclude_all_ed` policy; main run §11.6)
3. ~~Build image-only pneumonia baseline on the same ED temporal evaluation split and target policy.~~ — **done** (table §5.5; results §9.6)
4. ~~**Build multimodal fusion** (first version: triage + image, temporal split).~~ — **done** (§9.7; frozen multilabel backbone + train-only tabular preprocessing).
5. Run focused ablations and remaining uncertainty / reporting:
   - triage-only vs labs-only vs triage+labs
   - `u_ignore` vs `u_zero`
   - lab lookback sensitivity
   - ~~**bootstrap CIs** for AUROC/AUPRC on test (clinical + image + multimodal)~~ — **done** for key pairs (§9.8); extend or pre-register additional comparisons if needed
   - multimodal variants: **unfrozen** backbone; backbone from **pneumonia fine-tune** ckpt vs multilabel ckpt
   - longer fine-tune / hyperparameter search for image / multimodal branches as needed
   - **calibration** (reliability diagrams) and **multiple-comparison** discipline for many bootstrap tests

---

## 13. Future extensions

- Stronger image pretraining monitoring (per-label AUROC/AUPRC, calibration checks)
- ED-overlap pneumonia fine-tuning/evaluation
- Stronger clinical feature sets and time-aware lab summaries
- Clinically actionable pneumonia target (T2)
- Calibration and subgroup analysis

---

## 14. Historical notes preserved (verbatim legacy context)

The project previously documented these milestone statements, kept here to preserve timeline context:
- "Current split comes from MIMIC -> very small val/test"
- "No temporal split yet"
- "No labs (labevents) included yet"
- "No model training yet"
- "First baseline model: clinical-only logistic regression"

These lines are historical and are superseded by Sections 5–13 above (including image pretrain §11.5–11.6, image-only §9.6, and multimodal §9.7).

---

## 15. Code and training updates (2026-03)

The following reflects **current source** behavior. **Re-run training** to align saved `artifacts/models/**/metrics.json`, `summary.json`, and prediction CSVs with this behavior.

### Data pipeline
- **`docs/runbook.md` §1 step 8:** documented order is **link triage → build triage features → build triage model table** (matches script dependencies).
- **`build_triage_features.py`:** vitals **clipping** + missing flags; rebuild triage → clinical → temporal tables after changing clips.

### Clinical (triage-only)
- **`clinical_baseline.py` / `train_clinical_baseline.py`:** `prepare_feature_matrix` coerces numerics, fills **triage missing-indicator** and **`is_pa`/`is_ap`**, normalizes **categorical strings** before the sklearn `ColumnTransformer`. Trainer validates non-empty splits; saves **`model_bundle.joblib`** and **`model.joblib`**; **`val_predictions.csv` / `test_predictions.csv`** include **`subject_id`** and optional **`study_id` / `dicom_id` / `temporal_split`** via `build_prediction_df`.
- **`clinical_xgb.py` / `train_clinical_xgb.py`:** aligned **tabular preprocessing** (flags, `is_pa`/`is_ap`, categoricals → **pandas `category`** for **`enable_categorical=True`**). **`XGBClassifier`:** `n_estimators=1000`, **`eval_metric="aucpr"`**, **`early_stopping_rounds`** on the estimator (default **30**, overridable with **`--early-stopping-rounds`**), **`fit(..., eval_set=[(X_val, y_val)])`**, **`n_jobs=-1`**. **`metrics.json`** records **`best_iteration`** / **`best_score`** when XGBoost provides them (verified with **XGBoost 3.2** in this project).

### Clinical + labs (secondary)
- **`clinical_xgb_with_labs` / `train_clinical_xgb_with_labs.py`:** still the **older** recipe (300 trees, no early stopping, plain `fit` on train only). **Not** matched to triage-only XGB above; use for appendix/exploratory only unless you port the same early-stopping pattern.

### Image / multimodal
- **`CXRBinaryDataset`:** returns **`subject_id`, `study_id`, `dicom_id`, `image_path`** per sample.
- **`train_image_pneumonia_finetune.py`:** **`val_predictions.csv` / `test_predictions.csv`** built from **batch-aligned IDs** (no row-order repair required for new runs). **Early stopping** on **validation AUPRC** (defaults **`--epochs 10`**, **`--patience 5`**); **`best.pt`** is the best **val AUPRC** checkpoint; **`summary.json`** includes **`best_val_auprc`**, **`best_epoch`**, **`epochs_completed`**.
- **`train_multimodal_pneumonia.py`:** same **val AUPRC** early-stopping pattern; evaluation path collects **IDs + `image_path`** into prediction CSVs.
- **`train_image_multilabel_pretrain.py`:** **early stopping on validation masked loss** (defaults **`--epochs 10`**, **`--patience 5`**); **`best.pt`** by val loss.

### Evaluation
- **`repair_prediction_ids.py`:** still useful for **legacy** image CSVs that lack IDs; **new** image fine-tune runs should already include **`subject_id` / `study_id` / `dicom_id`** (and optionally **`image_path`**).
- **`scripts/evaluate_normal_vs_abnormal_negatives.py`:** optional **target-specificity** metrics (CheXpert non-pneumonia abnormality strata on negatives); **`docs/runbook.md` §4.9**.

### Runs executed locally (2026-03)
- **`train_clinical_baseline.py`** → `artifacts/models/clinical_baseline_u_ignore_temporal_phase1_clean_v1`
- **`train_clinical_xgb.py`** with **`--early-stopping-rounds 30`** → `artifacts/models/clinical_xgb_u_ignore_temporal_phase1_clean_v1`
- An additional **`train_clinical_xgb.py`** run targeted **`artifacts/models/clinical_xgb_u_ignore_temporal`** (older default dir; metrics differ — see §9.3 “Earlier run”).
- **`train_image_pneumonia_finetune.py`** (multilabel-init checkpoint, `--epochs 10 --patience 5`) → `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_phase1_clean_v1`
- **`train_multimodal_pneumonia.py`** (`--freeze-image-backbone`, `--epochs 10 --patience 5`) → `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_phase1_clean_v1`
- **`bootstrap_eval.py`** comparisons executed:
  - multimodal phase1 vs XGB phase1 → `artifacts/evaluation/bootstrap_multimodal_vs_xgb_phase1_clean_v1.json`
  - multimodal phase1 vs image phase1 → `artifacts/evaluation/bootstrap_multimodal_vs_image_phase1_clean_v1.json`
  - image phase1 vs XGB phase1 → `artifacts/evaluation/bootstrap_image_vs_xgb_phase1_clean_v1.json`
  - image phase1 vs logistic phase1 → `artifacts/evaluation/bootstrap_image_vs_logistic_phase1_clean_v1.json`
  - multimodal phase1 vs logistic phase1 → `artifacts/evaluation/bootstrap_multimodal_vs_logistic_phase1_clean_v1.json`
  - multimodal **unfrozen** phase1 vs image phase1 → `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_vs_image_phase1_v1.json`
  - multimodal **unfrozen** phase1 vs multimodal frozen phase1 → `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_vs_frozen_phase1_v1.json`
  - multimodal **unfrozen seed43** vs image phase1 → `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_seed43_vs_image_phase1_v1.json`
  - multimodal **unfrozen seed44** vs image phase1 → `artifacts/archive/evaluation/sensitivity_unfrozen_phase1/bootstrap_multimodal_unfrozen_seed44_vs_image_phase1_v1.json`

### Docs / configs lag
- **`configs/experiments/*.yaml`** may still show **5 epochs** for image/multimodal; training CLIs now default to **10** with patience — treat YAML as stale until updated or a loader is wired.

---

## 16. Training-strength upgrade profile (current source, addendum)

This section records the **new default training behavior now implemented in source** after the undertraining audit. It is additive to historical sections above and should be treated as the current reference for new retrains.

### 16.1 What changed in code

- **Image fine-tune (`train_image_pneumonia_finetune.py`):**
  - defaults: `--epochs 40`, `--patience 10`
  - optimizer now uses **parameter groups** (`--lr-backbone` default `3e-5`, `--lr-head` default `1e-4`; `--lr` overrides both)
  - **ReduceLROnPlateau** on validation AUPRC (`mode=max`, factor `0.5`, patience `3`, min lr `1e-6`)
  - added AMP toggle (`--disable-amp`) and configurable grad clipping (`--max-grad-norm`, default `1.0`)
  - checkpoint payload now includes `scheduler_state_dict`; history includes per-group learning rates

- **Multimodal (`train_multimodal_pneumonia.py`):**
  - defaults: `--epochs 30`, `--patience 8`
  - optimizer now uses **parameter groups** (image backbone vs tabular/fusion) with same lr controls as image fine-tune
  - **ReduceLROnPlateau** on validation AUPRC (same settings as image fine-tune)
  - AMP + grad clipping remain active; `scheduler_state_dict` saved in checkpoints
  - `--freeze-image-backbone` remains available for ablations, but unfrozen training is preferred for headline performance

- **Image multilabel pretrain (`train_image_multilabel_pretrain.py`):**
  - defaults increased to `--epochs 20`, `--patience 6`
  - optimizer now uses backbone/head lr groups
  - **ReduceLROnPlateau** added (`mode=min` for `val_loss`, factor `0.5`, patience `2`, min lr `1e-6`)
  - AMP + grad clipping added; micro-AUROC/AUPRC tracking for masked labels

- **Clinical baselines:**
  - logistic baseline now uses `solver="saga"`, `max_iter=10000`, `n_jobs=-1`
  - XGBoost baseline now uses `n_estimators=2000`, `learning_rate=0.03`, `max_depth=5`, with default `early_stopping_rounds=40`

### 16.2 Recommended retrain order (for valid updated comparisons)

1. `train_image_multilabel_pretrain.py` (strong profile)
2. `train_image_pneumonia_finetune.py` using the new pretrain checkpoint
3. `train_multimodal_pneumonia.py` (unfrozen strong profile)
4. clinical retrains (`train_clinical_baseline.py`, `train_clinical_xgb.py`) for fair comparison snapshots

### 16.3 Evaluation rerun requirement (important)

After retraining, rerun evaluation outputs so results are not mixed across old/new training recipes:

- **Bootstrap CIs / paired deltas:** `src/evaluation/bootstrap_eval.py`
- **Calibration:** `src/evaluation/calibration_analysis.py`
- **Decision curves:** `src/evaluation/decision_curve_analysis.py`

Use fresh output names (e.g., `*_strong_v2`) so previous phase1 artifacts remain intact and audit trail stays clear.

### 16.4 Terminal run status update (2026-03-24)

Observed in local terminal logs:

- **Strong pretrain launch attempt failed before training started**:
  - command target: `train_image_multilabel_pretrain.py` with `--output-dir artifacts/models/image_multilabel_pretrain_densenet121_strong_v2 --epochs 30 ...`
  - error: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`
  - impact: no valid `strong_v2` model metrics should be interpreted from that failed attempt.

- **Fix applied in source**:
  - removed scheduler `verbose=` argument from all upgraded neural trainers (`train_image_multilabel_pretrain.py`, `train_image_pneumonia_finetune.py`, `train_multimodal_pneumonia.py`) for compatibility with installed torch API.

- **CUDA environment recovery completed**:
  - venv torch stack reinstalled to CUDA build.
  - verification output recorded:
    - `torch==2.6.0+cu124`
    - `cuda_available True`
    - `torch.version.cuda 12.4`

Current implication:
- the legacy `strong_v2` analytics command block is no longer the preferred final set for image/multimodal.
- use final image/multimodal runs:
  - `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv`
  - `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv`
- keep clinical comparators on:
  - `artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv`
  - `artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv`
- run analytics with the `stronger_lr_v3` command block in `docs/runbook.md` §4.5.

### 16.5 Completed stronger_lr_v3 runs + analytics (verified from terminal)

The following runs and analyses completed successfully after CUDA recovery and scheduler compatibility fixes.

- **Image fine-tune**
  - Output dir: `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3`
  - Best checkpoint: epoch **2** (early stopped at epoch 12 / patience 10)
  - Validation: AUROC **0.7348**, AUPRC **0.7463**
  - Test: AUROC **0.7456**, AUPRC **0.7245**

- **Multimodal**
  - Output dir: `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3`
  - Best checkpoint: epoch **2** (early stopped at epoch 10 / patience 8)
  - Validation: AUROC **0.7381**, AUPRC **0.7457**
  - Test: AUROC **0.7362**, AUPRC **0.7143**

- **Clinical comparators refreshed (`strong_v2`)**
  - Logistic test: AUROC **0.6063**, AUPRC **0.5480**
  - XGBoost test: AUROC **0.6105**, AUPRC **0.5665**

Bootstrap (*n*=2000, seed 42; patient-level resampling):
- `bootstrap_multimodal_vs_image_stronger_lr_v3.json`
  - ΔAUROC (A-B): **-0.0091** [**-0.0227**, **0.0047**], `p(delta>0)=0.100`
  - ΔAUPRC (A-B): **-0.0099** [**-0.0240**, **0.0040**], `p(delta>0)=0.083`
- `bootstrap_image_vs_xgb_stronger_lr_v3.json`
  - ΔAUROC: **0.1355** [**0.0956**, **0.1739**], `p(delta>0)=1.000`
  - ΔAUPRC: **0.1571** [**0.1124**, **0.2017**], `p(delta>0)=1.000`
- `bootstrap_multimodal_vs_xgb_stronger_lr_v3.json`
  - ΔAUROC: **0.1264** [**0.0902**, **0.1638**], `p(delta>0)=1.000`
  - ΔAUPRC: **0.1471** [**0.1040**, **0.1894**], `p(delta>0)=1.000`

Calibration (`artifacts/evaluation/calibration_stronger_lr_v3/calibration_metrics.json`):
- Clinical Logistic strong_v2: Brier **0.2422**, ECE **0.0375**
- Clinical XGB strong_v2: Brier **0.2405**, ECE **0.0460**
- Image stronger_lr_v3: Brier **0.2063**, ECE **0.0674**
- Multimodal stronger_lr_v3: Brier **0.2069**, ECE **0.0403**

Decision curve analysis:
- command completed with 4 models; outputs saved under `artifacts/evaluation/dca/` (summary + per-model threshold tables + curve CSV/PNG).

Interpretation snapshot:
- On this stronger run set, **image > multimodal** by point estimate on both AUROC/AUPRC, with bootstrap CIs crossing zero for image-vs-multimodal deltas.
- Both image and multimodal remain **substantially above** clinical XGB/logistic on discrimination metrics and bootstrap deltas.

### 16.6 Latest terminal probe: image stronger_lr_v3 target-specificity slice

A new terminal one-off reran the CheXpert-stratified negative analysis on:
- predictions: `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv`
- merge labels: `D:/mimic-cxr-2.0.0-chexpert.csv.gz`
- merge keys: `subject_id`, `study_id`
- abnormality columns: `Atelectasis`, `Edema`, `Pleural Effusion`, `Consolidation`, `Lung Opacity`

Observed counts:
- initial merged rows: **1075**
- after dropping rows with all five abnormality columns NaN: **686**
- positives: **392**
- normal negatives (`any_abnormal=False`): **132**
- abnormal negatives (`any_abnormal=True`): **162**

Observed performance split:
- positives vs normal negatives: AUROC **0.7691**, AUPRC **0.9144**
- positives vs abnormal negatives: AUROC **0.6320**, AUPRC **0.8097**
- abnormal-minus-normal contrast: ΔAUROC **-0.1371**, ΔAUPRC **-0.1047**

Interpretation:
- This reproduces the same qualitative pattern documented earlier: materially lower discrimination when negatives are radiographically abnormal (non-pneumonia), consistent with persistent target-specificity risk.

**Scripted rerun:** the same pipeline (merge, all-NaN drop, `any_abnormal` == any of the five columns equals 1, AUROC/AUPRC on the two negative strata) is implemented in **`scripts/evaluate_normal_vs_abnormal_negatives.py`**; see **`docs/runbook.md` §4.9** for example commands matching this study-level merge.

**Committed JSON + multimodal stratum (terminal 2026-03-25, Git Bash on Windows, exit code 0):** Full floating-point summaries were saved to **`artifacts/evaluation/image_normal_vs_abnormal_negatives_stronger_lr_v3.json`** and **`artifacts/evaluation/multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json`** (CheXpert path in JSON: `D:\mimic-cxr-2.0.0-chexpert.csv.gz`; merge keys `subject_id`, `study_id`). Row counts match the bullets above. **Multimodal** on the same 686-row slice: positives vs normal negatives (*n* = 524) AUROC **0.754851**, AUPRC **0.906287**; vs abnormal negatives (*n* = 554) AUROC **0.599080**, AUPRC **0.791940**; abnormal-minus-normal ΔAUROC **−0.155770**, ΔAUPRC **−0.114347**. On this slice the **multimodal** abnormal-negative drop is **larger** than image-only (ΔAUROC **−0.137074**, ΔAUPRC **−0.104675** in the image JSON), i.e. fusion does not remove the CheXpert-stratified hardness pattern.

### 16.7 Grad-CAM pipeline hardening + verified terminal runs (2026-03-24)

Grad-CAM generation for the stronger image model was upgraded and validated from terminal runs.

Code-level upgrades applied:
- `scripts/generate_gradcam_examples.py`
  - switched confusion modes to true threshold-based definitions:
    - `fp`: predicted positive and target 0
    - `tp`: predicted positive and target 1
    - `fn`: predicted negative and target 1
  - default `--target-layer` set to `features.norm5` (more localized than hooking full `features`).
  - added `--threshold`, `--target-layer`, `--image-size`, `--alpha`.
  - added required-column checks for predictions CSV (`image_path`, `target`, `pred_prob`).
  - writes `selection_summary.json` and per-case raw heatmaps (`heatmaps/*.npy`) in output dir.
  - forced non-interactive matplotlib backend (`Agg`) to avoid Tk runtime dependency issues in headless/venv setups.
- `src/interpretability/gradcam.py`
  - replaced module-level backward hook with activation-tensor gradient hook (`output.register_hook(...)`) to avoid DenseNet inplace/view autograd failures.
  - vectorized CAM aggregation and removed unnecessary `retain_graph=True`.

Terminal errors observed and resolved in sequence:
- `ModuleNotFoundError: No module named 'src'`
  - cause: running as file script (`python scripts/...`) rather than module.
  - resolution: run with module mode (`python -m scripts.generate_gradcam_examples`).
- `FileNotFoundError` for checkpoint path ending in `/best.pt`
  - cause: best checkpoint is stored under `/checkpoints/best.pt`.
  - resolution: use `--checkpoint artifacts/models/.../checkpoints/best.pt`.
- DenseNet autograd hook failure:
  - error: `Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace`.
  - resolution: hook gradients from activation tensor in forward hook (not module full backward hook).
- Matplotlib Tk error:
  - error: `_tkinter.TclError: Can't find a usable init.tcl`.
  - resolution: set matplotlib backend to `Agg`.

Verified working command pattern:
- script launcher: `python -m scripts.generate_gradcam_examples`
- stronger model checkpoint:
  - `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt`
- predictions used:
  - `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/val_predictions.csv`
  - `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv`

Current status:
- Grad-CAM outputs were successfully generated (e.g., `artifacts/interpretability/gradcam_val_fp/`, `artifacts/interpretability/gradcam_val_tp/`), and are now reproducible with the runbook command block.

### 16.8 Prediction behavior report (`scripts/check_prediction_behavior.py`) — terminal runs (2026-03-24–25)

Script: `scripts/check_prediction_behavior.py` — reads a `test_predictions.csv` / `val_predictions.csv` with **`target`** and **`pred_prob`**, prints distribution quantiles, AUROC/AUPRC, thresholded accuracy/precision/recall/F1, confusion counts, and a dummy “always positive” sanity baseline. Writes a folder with **`summary.csv`**, **`predictions_copy.csv`**, **`prediction_histogram.png`**, **`prediction_distribution_by_class.csv`**, **`top_false_positives.csv`**, **`top_false_negatives.csv`**.

**Canonical test-set runs** (threshold **0.5**, *n* = **1075**, prevalence **0.4530232558139535** per `summary.csv`):

| Model | Output directory | AUROC | AUPRC | Acc @0.5 | Prec @0.5 | Rec @0.5 | F1 @0.5 | Pred + rate @0.5 | TN / FP / FN / TP |
|--------|------------------|-------|-------|----------|-----------|----------|---------|------------------|-------------------|
| Clinical logistic `strong_v2` | `artifacts/evaluation/prediction_behavior_clinical_logistic_strong_v2/` | 0.606293 | 0.547974 | 0.576744 | 0.534632 | 0.507187 | 0.520548 | 0.429767 | 373 / 215 / 240 / 247 |
| Clinical XGB `strong_v2` | `artifacts/evaluation/prediction_behavior_clinical_xgb_strong_v2/` | 0.610542 | 0.566522 | 0.578605 | 0.535417 | 0.527721 | 0.531541 | 0.446512 | 365 / 223 / 230 / 257 |
| Image-only `stronger_lr_v3` | `artifacts/evaluation/prediction_behavior_image_stronger_lr_v3/` | 0.745582 | 0.724468 | 0.678140 | 0.640719 | 0.659138 | 0.649798 | 0.466047 | 408 / 180 / 166 / 321 |
| Multimodal `stronger_lr_v3` | `artifacts/evaluation/prediction_behavior_multimodal_stronger_lr_v3/` | 0.736206 | 0.714312 | 0.669767 | 0.644105 | 0.605749 | 0.624339 | 0.426047 | 425 / 163 / 192 / 295 |

**Thesis-style rounded export (three decimals where applicable):** **`artifacts/evaluation/final_results_table.csv`** (AUROC/AUPRC/Acc/Prec/Rec/F1) and one-paragraph narrative **`artifacts/evaluation/final_result_note.txt`** — aligned with the four rows above for slides/tables.

**Runbook:** `docs/runbook.md` §4.8.

**Commands** (from repo root, `PYTHONPATH=.`):

```bash
PYTHONPATH=. python scripts/check_prediction_behavior.py \
  --predictions-csv artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv \
  --threshold 0.5 \
  --output-dir artifacts/evaluation/prediction_behavior_clinical_logistic_strong_v2

PYTHONPATH=. python scripts/check_prediction_behavior.py \
  --predictions-csv artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv \
  --threshold 0.5 \
  --output-dir artifacts/evaluation/prediction_behavior_clinical_xgb_strong_v2

PYTHONPATH=. python scripts/check_prediction_behavior.py \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --threshold 0.5 \
  --output-dir artifacts/evaluation/prediction_behavior_image_stronger_lr_v3

PYTHONPATH=. python scripts/check_prediction_behavior.py \
  --predictions-csv artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --threshold 0.5 \
  --output-dir artifacts/evaluation/prediction_behavior_multimodal_stronger_lr_v3
```

Clinical + image + multimodal numbers above match the respective `summary.csv` files in each output dir (2026-03 local tree); re-running overwrites those dirs.
