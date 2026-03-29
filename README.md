# Multimodal Pneumonia Detection (MIMIC-CXR + MIMIC-IV / MIMIC-IV-ED)

BSc thesis codebase for publication-oriented, time-safe **multimodal pneumonia detection** from chest X-rays and structured ED/clinical data.

## What this repo contains

- **`src/`** — cohort pipelines (`src/data/`), datasets, models, training CLIs (`src/training/`), QC (`src/qc/`), **evaluation** (`src/evaluation/`: patient-level bootstrap CIs; **`repair_prediction_ids.py`** for legacy ID-less image CSVs — current image/multimodal trainers write IDs in-batch). Ad-hoc **prediction distribution / threshold reports**: **`scripts/check_prediction_behavior.py`** (see **`docs/runbook.md` §4.8**, **`docs/current_state.md` §16.8**).
- **`artifacts/`** — derived tables, training manifests, model runs, and logs. See **`artifacts/models/README.md`** for canonical vs secondary model dirs.
- **`configs/`** — `paths.local.example.yaml` (copy to `paths.local.yaml` for machine-specific roots) and **`configs/experiments/*.yaml`** documenting hyperparameters for headline runs. Training scripts currently use **argparse**; YAML files are the **reference spec** until optional loader wiring.
- **`tools/audits/`** — ad-hoc consistency / profiling scripts (not part of the training pipeline).
- **`pyproject.toml`** — package metadata and Python dependencies (`pip install -e .` or `uv pip install -e .`).

## Documentation (read these first)

| Doc | Role |
|-----|------|
| **[`docs/current_state.md`](docs/current_state.md)** | Single source of truth: cohorts, tables, **results**, limitations, next steps |
| **[`docs/runbook.md`](docs/runbook.md)** | Ordered pipeline and training commands |
| [`docs/data_versions.md`](docs/data_versions.md) | Dataset locations, **canonical vs secondary vs archived** artifacts |
| [`docs/cohort_definition.md`](docs/cohort_definition.md) | Cohort and split policy |
| [`docs/triage_feature_policy.md`](docs/triage_feature_policy.md) | Triage feature allowlist and leakage rules |
| [`docs/project_definition.md`](docs/project_definition.md) | Goals and high-level status |

## Headline results (canonical runs, temporal `u_ignore`, 9,137 studies, test *n* = 1,075)

**Canonical model dirs** under `artifacts/models/` (see **`artifacts/models/README.md`**): **`clinical_*_strong_v2`**, **`image_pneumonia_finetune_*_stronger_lr_v3`**, **`multimodal_*_stronger_lr_v3`**, plus upstream **`image_multilabel_pretrain_densenet121_strong_v2`**. Older runs live in **`artifacts/archive/models/from_models_root_2026_03/`**.

**Point estimates** (test set) from `metrics.json` / `summary.json`:

- **Clinical logistic (`clinical_baseline_u_ignore_temporal_strong_v2`):** AUROC **0.606**, AUPRC **0.548**
- **Clinical XGB (`clinical_xgb_u_ignore_temporal_strong_v2`):** AUROC **0.611**, AUPRC **0.567**
- **Image-only (`image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3`):** AUROC **0.746**, AUPRC **0.724**
- **Multimodal (`multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3`):** AUROC **0.736**, AUPRC **0.714**

**Patient-level bootstrap** (*n* = 2000, seed 42): `artifacts/evaluation/bootstrap_*_stronger_lr_v3.json` (multimodal vs image, image/multimodal vs clinical XGB, image vs XGB). Historical phase1 bootstrap JSONs and figures were moved to **`artifacts/archive/evaluation/from_evaluation_root_2026_03/`**.

Commands: **`docs/runbook.md`**. Extended narrative and caveats: **`docs/current_state.md`** (includes older §9 history; align new writing with canonical dirs above).

## Local setup

1. Python ≥ 3.10, CUDA optional but recommended for image runs.
2. `pip install -e .` from the repo root (or install dependencies from `pyproject.toml`).
3. Copy `configs/paths.local.example.yaml` → `configs/paths.local.yaml` and set your MIMIC roots (file is gitignored).
4. Run steps in **`docs/runbook.md`** with `PYTHONPATH=.` (or install the package editable and use `python -m` if you add module entrypoints later).

## Streamlit dashboard

You can launch a lightweight project dashboard to inspect run metrics and bootstrap outputs:

```bash
streamlit run streamlit_app.py
```

The app reads:
- `artifacts/runs/registry.json`
- model `metrics.json` / `summary.json` referenced in the registry
- `artifacts/evaluation/bootstrap*.json`

## Artifacts and reproducibility

- **Canonical** model outputs live under **`artifacts/models/`** (see `artifacts/models/README.md`).
- **Sensitivity / smoke / abandoned** runs are under **`artifacts/archive/`** (see `.gitignore` — this subtree may be **local-only** in some clones).
- Neural **checkpoints** (`.pt`) are often **gitignored**; committed evidence is typically `config.json`, `history.json`, `summary.json`, and prediction CSVs where present. Multimodal runs also need **`tabular_preprocessor.joblib`** paired with the checkpoint.
- **`artifacts/runs/registry.json`** — index of headline model runs (paths to metrics/summary files); extend with git SHA and commands as needed.
- **`artifacts/evaluation/`** — bootstrap JSON (and optional CSV replicates) from `bootstrap_eval.py`; calibration under **`calibration_stronger_lr_v3/`**; DCA under **`dca/`**; per-model **`prediction_behavior_*`** folders from `scripts/check_prediction_behavior.py`; CheXpert-stratified summaries **`image_normal_vs_abnormal_negatives_stronger_lr_v3.json`** / **`multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json`** from `scripts/evaluate_normal_vs_abnormal_negatives.py`; rounded table + note **`final_results_table.csv`**, **`final_result_note.txt`**. Create the directory when you first save results.
- **Calibration tooling:** `src/evaluation/calibration_analysis.py` — default model map targets the **canonical** `strong_v2` / `stronger_lr_v3` prediction CSVs; latest committed run output: **`artifacts/evaluation/calibration_stronger_lr_v3/`**. Older calibration outputs: **`artifacts/archive/evaluation/from_evaluation_root_2026_03/calibration_phase1_default/`**.
- **Decision curve analysis:** `src/evaluation/decision_curve_analysis.py` (pass `--model` paths explicitly). Previous DCA exports: **`artifacts/archive/evaluation/from_evaluation_root_2026_03/dca_mixed_pre_canonical/`**.

## Compliance

PhysioNet credentialing applies to MIMIC data. Do not commit raw patient-level exports or credentials.
