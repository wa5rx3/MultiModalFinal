# Data versions and artifact map

_Last aligned with repo layout and committed metrics: 2026-03-22._

## Local MIMIC roots (machine-specific)

Copy `configs/paths.local.example.yaml` → `configs/paths.local.yaml` (gitignored) and set:

| Dataset | Example key | Typical use |
|---------|-------------|-------------|
| MIMIC-CXR-JPG 2.1.0 | `mimic_cxr_root` | Image files + manifest build |
| MIMIC-IV | `mimic_iv_root`, `labevents_dir` | Labs, admissions |
| MIMIC-IV-ED | `mimic_iv_ed_root` | `edstays.csv`, `triage.csv` |

All access must follow PhysioNet credentialing. Do not commit raw exports or credentials.

---

## Canonical derived artifacts (primary pipeline)

**Cohort and splits**

- `artifacts/manifests/cxr_final_ed_cohort.parquet` — 81,385 ED-linked studies
- `artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet` — same + `temporal_split`
- `artifacts/manifests/cxr_pneumonia_labels.parquet` — CheXpert T1 pneumonia (study-level fallback merge; no `dicom_id` in current CheXpert export)

**Temporal `u_ignore` evaluation core (9,137 rows)**

- `artifacts/manifests/cxr_pneumonia_training_table_u_ignore_temporal.parquet` — binary target + `temporal_split` + `image_path`
- `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet` — triage features + same rows (clinical + multimodal)
- `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet` — image-only fine-tune rows

**Image pretraining (exclude-all-ED policy, publication-facing)**

- `artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet` — 217,922 rows with pretrain split columns
- `artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet` — 45,837 trainable multilabel rows (see `config.json` in main pretrain run for train/val counts)

**Labs branch (hadm-only primary)**

- `artifacts/tables/cohort_labevents_hadm_only.parquet` — strict `hadm_only` extract
- `artifacts/tables/cxr_lab_features_hadm_only.parquet` — 6,605 studies with wide lab features
- `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet` — full 9,137-row left-merged table (~9.5% rows with any lab signal)

---

## Canonical model output directories

Committed summaries / configs (checkpoints `.pt` often local-only). **Headline runs** under `artifacts/models/`:

| Run | Directory |
|-----|-----------|
| Triage logistic | `artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/` |
| Triage XGBoost | `artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/` |
| Multilabel pretrain (upstream for image + multimodal) | `artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/` |
| Image pneumonia (multilabel-init) | `artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/` |
| Multimodal | `artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/` |

**Archived** former `artifacts/models/*` (phase1, labs baselines, ImageNet-only, `*_main` pretrain, etc.): **`artifacts/archive/models/from_models_root_2026_03/`**.

See also `artifacts/models/README.md`.

---

## Secondary / non-headline artifacts

- **`u_zero` policy:** `artifacts/manifests/cxr_pneumonia_training_table_u_zero.parquet` — larger row count, different label handling; not the primary temporal benchmark.
- **Broader lab extract (non-hadm-only):** `artifacts/tables/cohort_labevents.parquet` and `artifacts/tables/cxr_lab_features.parquet` if present — use hadm-only paths above for headline labs work.
- **Overlap-only labs subset** (867 rows): tables and logs under `artifacts/tables/` / `artifacts/logs/` with `..._overlap...` in the name — **sensitivity only** (see `docs/current_state.md` §9.5).

---

## Archive policy (sensitivity, smoke, abandoned)

The repo uses **`artifacts/archive/`** for non-headline model runs and similar outputs, e.g.:

- `artifacts/archive/models/from_models_root_2026_03/` — bulk move of superseded top-level `artifacts/models/*` dirs (2026-03).
- `artifacts/archive/evaluation/from_evaluation_root_2026_03/` — legacy bootstrap JSONs + old `calibration/` + `dca/` outputs.
- `artifacts/archive/models/overlap_sensitivity/` — overlap-only **clinical** baselines (trained on the small overlap cohort; not primary claims).
- `artifacts/archive/models/smoke_runs/` — e.g. quick multilabel smoke pretrain.
- `artifacts/archive/models/abandoned_pretrain/` — incomplete / superseded pretrain attempts.

**Note:** `.gitignore` may list `artifacts/archive/` — in that case archives exist **only on local disks** unless you change ignore rules. Document what you keep in git vs locally.

**Sample / QC-only manifests** (if present): often under `artifacts/archive/manifests/samples/` after cleanup — do not treat as production cohorts.

---

## Logs and QC layout

- **`artifacts/logs/`** — pipeline JSON reports (merges, lab builds, etc.).
- **`artifacts/logs/qc/`** — linkage / label-balance QC JSON (when relocated).
- **`artifacts/logs/audits/`** — consistency / profiling audit JSON.

---

## Evaluation outputs (`artifacts/evaluation/`)

- **Bootstrap summaries** — JSON from `src/evaluation/bootstrap_eval.py` (patient-level resampling, optional paired Δ metrics). **Canonical** (current): `bootstrap_*_stronger_lr_v3.json`. Legacy JSONs: **`artifacts/archive/evaluation/from_evaluation_root_2026_03/`**.
- **Calibration (canonical):** `artifacts/evaluation/calibration_stronger_lr_v3/`
- **Decision curve analysis (canonical four-model run):** `artifacts/evaluation/dca/` — from `src/evaluation/decision_curve_analysis.py` (`docs/runbook.md` §4.5).
- **Prediction behavior (canonical test-set summaries @0.5):** `artifacts/evaluation/prediction_behavior_clinical_logistic_strong_v2/`, `artifacts/evaluation/prediction_behavior_clinical_xgb_strong_v2/`, `artifacts/evaluation/prediction_behavior_image_stronger_lr_v3/`, `artifacts/evaluation/prediction_behavior_multimodal_stronger_lr_v3/` — from `scripts/check_prediction_behavior.py` (**`docs/current_state.md` §16.8**).
- **CheXpert-stratified negative metrics (image + multimodal `stronger_lr_v3`, study-level merge):** `artifacts/evaluation/image_normal_vs_abnormal_negatives_stronger_lr_v3.json`, `artifacts/evaluation/multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json` — from `scripts/evaluate_normal_vs_abnormal_negatives.py` (**§16.6** in `docs/current_state.md`, **`docs/runbook.md` §4.9**).
- **Rounded headline table + narrative:** `artifacts/evaluation/final_results_table.csv`, `artifacts/evaluation/final_result_note.txt` (aligned with the four `prediction_behavior_*` `summary.csv` rows).
- **Optional CSVs** — with `--save-bootstrap-csv`, per-replicate files (e.g. `bootstrap_model_a.csv`, `bootstrap_delta_a_minus_b.csv`) are written **next to** the chosen `--output-json` path.

Create **`artifacts/evaluation/`** locally when you first run with `--output-json`; add to git only if you want these summaries versioned.

## Prediction CSV variants (`artifacts/models/.../`)

| File | Typical contents |
|------|------------------|
| `test_predictions.csv` / `val_predictions.csv` | From trainers: often **`target`** + **`pred_prob`** (image fine-tune) or **IDs + target + pred_prob** (clinical, multimodal). |
| `*_with_ids.csv` | From `src/evaluation/repair_prediction_ids.py`: **`subject_id`, `study_id`, `dicom_id`, `target`, `pred_prob`** — same row order and count as the corresponding split of `cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet`. Use for **bootstrap** and **paired** comparisons when the raw image CSV has no IDs. |

---

## Run registry

- **`artifacts/runs/registry.json`** — index of **headline** trained runs (output dirs and metrics/summary paths). Add fields such as `git_commit` and `command` per entry when you tighten reproducibility tracking.

---

## Experiment config snapshots (reference only)

- `configs/experiments/image_pretrain_main.yaml`
- `configs/experiments/image_finetune_u_ignore.yaml`
- `configs/experiments/multimodal_triage_u_ignore.yaml`

These document headline hyperparameters; training CLIs do not load them automatically yet.
