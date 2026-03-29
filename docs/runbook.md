# Pipeline runbook

Prerequisites: install the project (`pip install -e .`), set `PYTHONPATH=.` to the repo root **or** run from root with `PYTHONPATH=.`. Replace paths below using `configs/paths.local.example.yaml` as a template.

Training scripts use **argparse**. Files under `configs/experiments/` mirror the headline settings (name, paths, hyperparameters) for documentation; they are **not** auto-loaded yet—pass equivalent CLI flags.

**Canonical trained runs** (live under `artifacts/models/`): `clinical_*_strong_v2`, `image_pneumonia_finetune_*_stronger_lr_v3`, `multimodal_*_stronger_lr_v3`, and upstream **`image_multilabel_pretrain_densenet121_strong_v2`**. All other former `artifacts/models/*` outputs were moved to **`artifacts/archive/models/from_models_root_2026_03/`**. Legacy bootstrap + calibration + DCA outputs: **`artifacts/archive/evaluation/from_evaluation_root_2026_03/`**.

---

## 1. Canonical data pipeline (order)

Rough dependency order. Skip steps you have already frozen in `artifacts/manifests/` / `artifacts/tables/`.

| Step | Script | Notes |
|------|--------|--------|
| 1. CXR manifest | `src/data/build_cohort.py` | Requires `--base-root`, `--metadata-root` (MIMIC-CXR JPG + metadata CSVs). |
| 2. Primary frontal cohort | `src/data/build_primary_imaging_cohort.py` | Defaults write to `artifacts/manifests/`. |
| 3. Link CXR → ED stays | `src/data/link_cxr_to_edstays.py` | Requires `--edstays` (e.g. MIMIC-IV-ED `edstays.csv`). |
| 4. Final ED cohort | `src/data/build_final_ed_cohort.py` | |
| 5. Temporal split | `src/data/build_temporal_patient_split.py` | → `cxr_final_ed_cohort_with_temporal_split.parquet` |
| 6. CheXpert pneumonia labels | `src/data/build_pneumonia_labels_from_chexpert.py` | CheXpert path required. |
| 7. Binary training tables | `src/data/build_pneumonia_training_table.py` | `u_ignore` / `u_zero` policies. |
| 8. Triage (link → features → model table) | `src/data/link_cxr_to_triage.py`, then `src/data/build_triage_features.py`, then `src/data/build_triage_model_table.py` | **Run in this order.** Link step requires `--triage` (MIMIC-IV-ED `triage.csv`). Outputs: `cxr_ed_triage_linked.parquet` → `cxr_ed_triage_features.parquet` → `cxr_ed_triage_model_table.parquet`. |
| 9. Clinical + labels | `src/data/build_clinical_pneumonia_training_table.py` | |
| 10. Propagate temporal split | `src/data/apply_temporal_split.py` | Use for tables that need `temporal_split` joined from cohort. |
| 11. Labs (optional branch) | `extract_labevents_for_cohort.py`, `build_lab_features_from_labevents.py`, `build_clinical_labs_pneumonia_training_table.py` | See `docs/current_state.md` §8. |
| 12. Image pretraining split | `src/data/build_image_pretraining_split.py` | Prefer **`--policy exclude_all_ed`** for publication-facing pretraining. |
| 13. Multilabel pretrain table | `src/data/build_image_multilabel_pretrain_table.py` | |
| 14. Image pneumonia finetune table | `src/data/build_image_pneumonia_finetune_table.py` | |

**Example (labs extraction, hadm-only):**

```bash
PYTHONPATH=. python src/data/extract_labevents_for_cohort.py \
  --labevents-dir "D:/mimic_iv/labevents" \
  --cohort "artifacts/manifests/cxr_final_ed_cohort.parquet" \
  --feature-map "artifacts/tables/lab_feature_map.json" \
  --output "artifacts/tables/cohort_labevents_hadm_only.parquet" \
  --report "artifacts/logs/cohort_labevents_hadm_only_report.json" \
  --lookback-hours 24 --match-mode hadm_only
```

---

## 2. Canonical training (headline runs)

All use the **same** temporal `u_ignore` cohort (9,137 rows) unless noted.

### Clinical (triage-only)

```bash
PYTHONPATH=. python src/training/train_clinical_baseline.py
PYTHONPATH=. python src/training/train_clinical_xgb.py
# Optional: override XGB early stopping (default 30 rounds on val AUPRC, up to 1000 trees)
# PYTHONPATH=. python src/training/train_clinical_xgb.py --early-stopping-rounds 50
```

Defaults point at `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet`.

**Outputs:** baseline saves **`model_bundle.joblib`** + **`model.joblib`**; both trainers write **`val_predictions.csv` / `test_predictions.csv`** with **`subject_id`** (and **`study_id` / `dicom_id` / `temporal_split`** when present). XGB **`metrics.json`** includes **`best_iteration`** / **`best_score`** when available (XGBoost 3.x API).

**Explicit retrain to canonical dirs** (same as CLI defaults):

```bash
PYTHONPATH=. python src/training/train_clinical_baseline.py \
  --input artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --output-dir artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2

PYTHONPATH=. python src/training/train_clinical_xgb.py \
  --input artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --output-dir artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2 \
  --early-stopping-rounds 40
```

### Clinical + labs (full 9,137 table, hadm-only features)

```bash
PYTHONPATH=. python src/training/train_clinical_baseline_with_labs.py \
  --input artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet \
  --output-dir artifacts/models/clinical_baseline_with_labs_u_ignore_hadm_only_temporal

PYTHONPATH=. python src/training/train_clinical_xgb_with_labs.py \
  --input artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet \
  --output-dir artifacts/models/clinical_xgb_with_labs_u_ignore_hadm_only_temporal
```

(Adjust `--output-dir` if you do not want to overwrite committed baselines.)

**Note:** **`train_clinical_xgb_with_labs.py`** is still the **legacy** recipe (300 trees, no early stopping on val). Triage-only **`train_clinical_xgb.py`** has the newer **AUPRC + early stopping** path — see **`docs/current_state.md` §15**.

### Image multilabel pretraining (upstream checkpoint)

Canonical checkpoint directory: **`artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/`** (older `*_main` run → archive). Trainer defaults include longer epochs and LR scheduling — see `train_image_multilabel_pretrain.py --help`.

```bash
PYTHONPATH=. python src/training/train_image_multilabel_pretrain.py \
  --input-table artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet \
  --output-dir artifacts/models/image_multilabel_pretrain_densenet121_strong_v2 \
  --batch-size 16 --lr 1e-4 --weight-decay 1e-4 \
  --image-size 224 --num-workers 4 --seed 42
```

Reference YAML: `configs/experiments/image_pretrain_main.yaml`.

### Image-only pneumonia fine-tuning

**Multilabel-init (canonical):**

```bash
PYTHONPATH=. python src/training/train_image_pneumonia_finetune.py \
  --input-table artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet \
  --pretrained-checkpoint artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt \
  --output-dir artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3
```

**Trainer defaults:** longer schedule, **param-group LRs**, **ReduceLROnPlateau**, **AMP**, **early stopping on validation AUPRC**; prediction CSVs include **batch-aligned IDs**.

**ImageNet-only ablation** (optional): use `--pretrained-checkpoint ""` and a **new** `--output-dir` under `artifacts/models/` (older `imagenet_only` run is in **archive**).

Reference YAML: `configs/experiments/image_finetune_u_ignore.yaml`.

### Multimodal (triage + image, canonical)

```bash
PYTHONPATH=. python src/training/train_multimodal_pneumonia.py \
  --input-table artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --image-backbone-checkpoint artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt \
  --output-dir artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3 \
  --batch-size 16 --lr 1e-4 --weight-decay 1e-4 \
  --image-size 224 --num-workers 4 --seed 42
```

Add **`--freeze-image-backbone`** only if you want an ablation (older frozen runs are in **archive**). Defaults match the **stronger_lr_v3** training profile (unfrozen backbone unless flag set).

Reference YAML: `configs/experiments/multimodal_triage_u_ignore.yaml`.

### Strong training profile (current recommended defaults)

Use this block for new publication-facing reruns after the training-strength upgrade.

```bash
PYTHONPATH=. python src/training/train_image_multilabel_pretrain.py \
  --input-table artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet \
  --output-dir artifacts/models/image_multilabel_pretrain_densenet121_strong_v2 \
  --epochs 30 --patience 5 \
  --lr-backbone 3e-5 --lr-head 1e-4 \
  --batch-size 16 --image-size 224

PYTHONPATH=. python src/training/train_image_pneumonia_finetune.py \
  --input-table artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet \
  --pretrained-checkpoint artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt \
  --output-dir artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3 \
  --epochs 40 --patience 10 \
  --lr-backbone 3e-5 --lr-head 1e-4 \
  --batch-size 16 --image-size 224

PYTHONPATH=. python src/training/train_multimodal_pneumonia.py \
  --input-table artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --image-backbone-checkpoint artifacts/models/image_multilabel_pretrain_densenet121_strong_v2/checkpoints/best.pt \
  --output-dir artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3 \
  --epochs 30 --patience 8 \
  --lr-backbone 3e-5 --lr-head 1e-4 \
  --batch-size 16 --image-size 224

PYTHONPATH=. python src/training/train_clinical_baseline.py \
  --input artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --output-dir artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2

PYTHONPATH=. python src/training/train_clinical_xgb.py \
  --input artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet \
  --output-dir artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2 \
  --early-stopping-rounds 40
```

---

## 3. QC and audits

- **QC CLIs:** `src/qc/qc_*.py` — run after cohort or label changes.
- **Ad-hoc audits:** `tools/audits/` — not required for training.
- **Structured audit JSON:** `artifacts/logs/audits/` (if present in your tree).
- **Target-specificity (optional):** merge a run’s `test_predictions.csv` with the MIMIC-CXR CheXpert CSV on `subject_id` / `study_id`, then compare metrics on pneumonia positives vs negatives with / without non-pneumonia abnormality flags (method + headline numbers: `docs/current_state.md` §11.1). For a **single scripted** rerun of that stratification (AUROC/AUPRC + counts + JSON), use **`scripts/evaluate_normal_vs_abnormal_negatives.py`** — commands: **`docs/runbook.md` §4.9** (same definition as §11.1 / §16.6 unless you override merge keys).

---

## 4. Evaluation: prediction IDs + patient-level bootstrap

### 4.1 When you need `repair_prediction_ids.py`

**New runs:** **`train_image_pneumonia_finetune.py`** and **`train_multimodal_pneumonia.py`** write prediction CSVs with **`subject_id` / `study_id` / `dicom_id`** aligned to each batch — use them directly for **`bootstrap_eval.py`**.

**Legacy CSVs:** older image outputs had only **`target`** and **`pred_prob`** in **row order** of the eval split. For those, repair IDs before bootstrap or paired comparison.

Align rows with the canonical table (row count must match; **`target`** is checked when present on both sides):

```bash
PYTHONPATH=. python src/evaluation/repair_prediction_ids.py \
  --predictions artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --input-table artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet \
  --split-value test \
  --output artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions_with_ids.csv
```

Use `--split-value validate` for validation rows (930) when repairing `val_predictions.csv`. Only needed for **legacy** CSVs without IDs; **canonical** trainers write IDs directly. Older runs live under **`artifacts/archive/models/`**.

**Clinical** `test_predictions.csv` files include **`subject_id`** and **`study_id` / `dicom_id` / `temporal_split`** when those columns exist on the input table — no repair needed.

**Multimodal** `test_predictions.csv` from current `train_multimodal_pneumonia.py` includes the same ID columns (batch-aligned).

### 4.2 Bootstrap CIs and paired deltas

`src/evaluation/bootstrap_eval.py`:
- Resamples **patients** (`subject_id`) with replacement; includes **all studies** of each sampled patient per replicate.
- **Single model:** point AUROC/AUPRC plus bootstrap mean and **2.5–97.5%** interval.
- **Two models:** same for B, then **paired delta (A − B)** with **merge on `subject_id` (+ `study_id` if present)**; reports **`p(delta>0)`** (fraction of bootstrap draws where Δ>0).

**Canonical paired comparison** (same pattern for other A/B pairs):

```bash
PYTHONPATH=. python src/evaluation/bootstrap_eval.py \
  --model-a artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model-b artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --n-bootstrap 2000 \
  --seed 42 \
  --output-json artifacts/evaluation/bootstrap_multimodal_vs_image_stronger_lr_v3.json \
  --save-bootstrap-csv
```

Full **canonical** bootstrap + calibration + DCA block: **§4.5** below.

**Legacy** phase1 / ImageNet / unfrozen sensitivity JSONs and their prediction paths live under **`artifacts/archive/evaluation/`** and **`artifacts/archive/models/`** (paths inside those files may still say `artifacts/models/...`; resolve against the archive tree).

**Outputs:** `artifacts/evaluation/bootstrap_*_stronger_lr_v3.json` (current). With `--save-bootstrap-csv`, replicate-level CSVs are written next to each JSON.

See **`docs/current_state.md` §9.8** for historical interpretation notes (some rows refer to pre-archive runs).

### 4.3 Calibration analysis (Brier / ECE / reliability diagrams)

`src/evaluation/calibration_analysis.py` — **defaults** target the four **canonical** `test_predictions.csv` files (`strong_v2` clinical, `stronger_lr_v3` image + multimodal). Override with repeated `--model "Name" path`.

Committed outputs: **`artifacts/evaluation/calibration_stronger_lr_v3/`** (see **§4.5** command). Older phase1 calibration dir: **`artifacts/archive/evaluation/from_evaluation_root_2026_03/calibration_phase1_default/`**.

### 4.4 Decision curve analysis (DCA)

`src/evaluation/decision_curve_analysis.py` — pass **`--model`** for each comparator; use the **§4.5** block for the canonical four-model run. Choose `--output-dir` explicitly (e.g. `artifacts/evaluation/dca` or `artifacts/evaluation/dca_canonical`) so new runs do not overwrite archived figures.

### 4.5 Analytics rerun after stronger_lr_v3 final runs

Rerun analytics whenever model training recipe changes, so bootstrap/calibration/DCA summaries are aligned with current checkpoints.

```bash
PYTHONPATH=. python src/evaluation/bootstrap_eval.py \
  --model-a artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model-b artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --n-bootstrap 2000 --seed 42 \
  --output-json artifacts/evaluation/bootstrap_multimodal_vs_image_stronger_lr_v3.json \
  --save-bootstrap-csv

PYTHONPATH=. python src/evaluation/bootstrap_eval.py \
  --model-a artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model-b artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv \
  --n-bootstrap 2000 --seed 42 \
  --output-json artifacts/evaluation/bootstrap_image_vs_xgb_stronger_lr_v3.json \
  --save-bootstrap-csv

PYTHONPATH=. python src/evaluation/bootstrap_eval.py \
  --model-a artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model-b artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv \
  --n-bootstrap 2000 --seed 42 \
  --output-json artifacts/evaluation/bootstrap_multimodal_vs_xgb_stronger_lr_v3.json \
  --save-bootstrap-csv

PYTHONPATH=. python src/evaluation/calibration_analysis.py \
  --output-dir artifacts/evaluation/calibration_stronger_lr_v3 \
  --n-bins 10 \
  --bootstrap --n-bootstrap 2000 --bootstrap-seed 42 \
  --model "Clinical Logistic strong_v2" artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv \
  --model "Clinical XGB strong_v2" artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv \
  --model "Image stronger_lr_v3" artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model "Multimodal stronger_lr_v3" artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv

PYTHONPATH=. python src/evaluation/decision_curve_analysis.py \
  --output-dir artifacts/evaluation/dca \
  --model "Clinical Logistic strong_v2" artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv \
  --model "Clinical XGBoost strong_v2" artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv \
  --model "Image stronger_lr_v3" artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model "Multimodal stronger_lr_v3" artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --threshold-metrics "0.1,0.2,0.3,0.5,0.8"
```

Latest completed local outputs from this block:
- `artifacts/evaluation/bootstrap_multimodal_vs_image_stronger_lr_v3.json`
- `artifacts/evaluation/bootstrap_image_vs_xgb_stronger_lr_v3.json`
- `artifacts/evaluation/bootstrap_multimodal_vs_xgb_stronger_lr_v3.json`
- `artifacts/evaluation/calibration_stronger_lr_v3/calibration_metrics.json`
- `artifacts/evaluation/calibration_stronger_lr_v3/calibration_summary.csv`
- `artifacts/evaluation/calibration_stronger_lr_v3/reliability_diagram_all_models.png`
- DCA: **regenerate** with the command above → `artifacts/evaluation/dca/` (previous mixed-era DCA exports were moved to **`artifacts/archive/evaluation/from_evaluation_root_2026_03/dca_mixed_pre_canonical/`**)

Related evaluation artifacts from the same **canonical four-model** era (see **`docs/current_state.md` §16.6 / §16.8**, **`docs/data_versions.md`**): **`artifacts/evaluation/prediction_behavior_clinical_logistic_strong_v2/`**, **`prediction_behavior_clinical_xgb_strong_v2/`**, **`prediction_behavior_image_stronger_lr_v3/`**, **`prediction_behavior_multimodal_stronger_lr_v3/`**; CheXpert-stratified JSON **`image_normal_vs_abnormal_negatives_stronger_lr_v3.json`**, **`multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json`**; **`final_results_table.csv`** + **`final_result_note.txt`**.

Optional: **multimodal vs logistic** bootstrap (not in the three JSONs above):

```bash
PYTHONPATH=. python src/evaluation/bootstrap_eval.py \
  --model-a artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --model-b artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv \
  --n-bootstrap 2000 --seed 42 \
  --output-json artifacts/evaluation/bootstrap_multimodal_vs_logistic_stronger_lr_v3.json
```

### 4.6 Troubleshooting note (torch scheduler + CUDA wheel)

If training fails immediately with:
- `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

use current source (scheduler `verbose` removed), then verify environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.version.cuda)"
```

If CUDA is unavailable after package changes and torch shows a CPU build, reinstall CUDA wheels in the active venv:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.version.cuda)"
```

### 4.7 Grad-CAM generation (stronger_lr_v3 image model)

Use module mode so `src.*` imports resolve correctly, and point checkpoint to `checkpoints/best.pt`.

```bash
python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_test_fp \
  --mode fp --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35

python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_test_tp \
  --mode tp --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35

python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_test_fn \
  --mode fn --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35

python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/val_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_val_fp \
  --mode fp --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35

python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/val_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_val_tp \
  --mode tp --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35

python -m scripts.generate_gradcam_examples \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/val_predictions.csv \
  --checkpoint artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/checkpoints/best.pt \
  --output-dir artifacts/interpretability/gradcam_image_stronger_lr_v3_val_fn \
  --mode fn --top-k 20 --threshold 0.5 \
  --target-layer features.norm5 --image-size 224 --alpha 0.35
```

Each output directory includes:
- panel images (`*.png`)
- raw heatmaps (`heatmaps/*.npy`)
- run metadata (`selection_summary.json`)

If a Grad-CAM run fails:
- `ModuleNotFoundError: No module named 'src'` -> use `python -m scripts.generate_gradcam_examples` (not `python scripts/...`).
- `FileNotFoundError ... /best.pt` -> use `.../checkpoints/best.pt`.
- `_tkinter.TclError` -> script now forces `Agg`; if needed, update repo and rerun.
- DenseNet backward hook inplace/view error -> fixed in current `src/interpretability/gradcam.py`; pull latest local changes and rerun.

---

### 4.8 Prediction behavior (distribution, threshold metrics, error lists)

`scripts/check_prediction_behavior.py` summarizes a single predictions CSV (`target`, `pred_prob`): quantiles of `pred_prob`, AUROC/AUPRC, confusion matrix at a chosen threshold, histogram by class, and CSVs of top false positives / false negatives.

**Canonical test-set reports** (threshold `0.5`, *n* = 1075) — image + multimodal **`stronger_lr_v3`** and clinical **`strong_v2`**:

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

**Outputs** (per `--output-dir`):

| File | Contents |
|------|-----------|
| `summary.csv` | One row: *n*, threshold, prevalence, mean/std pred, AUROC, AUPRC, acc/prec/rec/F1, pred positive rate, TN/FP/FN/TP |
| `predictions_copy.csv` | Copy of input predictions |
| `prediction_histogram.png` | Overlapping histograms for target 0 vs 1 |
| `prediction_distribution_by_class.csv` | `describe()`-style stats per `target` |
| `top_false_positives.csv` | Up to 100 FP rows, sorted by `pred_prob` desc |
| `top_false_negatives.csv` | Up to 100 FN rows, sorted by `pred_prob` asc |

Interpreted headline metrics for all four output dirs are recorded in **`docs/current_state.md` §16.8**. A rounded **four-model** summary CSV + short narrative note for tables/slides: **`artifacts/evaluation/final_results_table.csv`**, **`artifacts/evaluation/final_result_note.txt`**.

---

### 4.9 CheXpert-stratified negatives (pneumonia vs “normal” vs “abnormal” negatives)

`scripts/evaluate_normal_vs_abnormal_negatives.py` reproduces the **target-specificity** slice documented in **`docs/current_state.md` §11.1** and **§16.6**: merge a predictions CSV with the MIMIC-CXR CheXpert file, stratify **negative** rows by non-pneumonia abnormality (`Atelectasis`, `Edema`, `Pleural Effusion`, `Consolidation`, `Lung Opacity`), define **`any_abnormal`** as **any column == 1** (CheXpert positives only; `-1`/NaN are not abnormal), drop rows where **all five** abnormality columns are NaN, then report AUROC/AUPRC for (positives + normal negatives) vs (positives + abnormal negatives) and the **abnormal-minus-normal** deltas.

**CLI:** `--predictions-csv` (needs `subject_id`, `study_id`, `target`, `pred_prob`), `--chexpert-csv` (gzip `.csv.gz` is fine for `pandas.read_csv`). Optional **`--merge-on-dicom`** merges on `subject_id`, `study_id`, `dicom_id` when both files carry `dicom_id` and CheXpert is unique per key (default is study-level: `subject_id`, `study_id`). Optional **`--output-json`** writes the full summary (merge keys, row counts, metrics).

**Caveats (unchanged from manual probes):** CheXpert rows are **deduplicated** on merge keys before join (first row kept if duplicates exist); rows with **all-NaN** abnormality fields are dropped from **both** strata (including possible positive rows); **`validate="one_to_one"`** on the merge requires predictions to be **unique** on the chosen merge keys.

**Examples (canonical **`stronger_lr_v3`** test predictions, study-level merge — same as terminal runs logged 2026-03-25):**

```bash
PYTHONPATH=. python scripts/evaluate_normal_vs_abnormal_negatives.py \
  --predictions-csv artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --chexpert-csv "D:/mimic-cxr-2.0.0-chexpert.csv.gz" \
  --output-json artifacts/evaluation/image_normal_vs_abnormal_negatives_stronger_lr_v3.json

PYTHONPATH=. python scripts/evaluate_normal_vs_abnormal_negatives.py \
  --predictions-csv artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv \
  --chexpert-csv "D:/mimic-cxr-2.0.0-chexpert.csv.gz" \
  --output-json artifacts/evaluation/multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json
```

**Committed JSON snapshots** (full floating-point metrics + merge metadata): **`artifacts/evaluation/image_normal_vs_abnormal_negatives_stronger_lr_v3.json`**, **`artifacts/evaluation/multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json`**. Headline numbers are summarized in **`docs/current_state.md` §16.6**.

Replace the `--chexpert-csv` path with your local MIMIC-CXR CheXpert export. Add **`--merge-on-dicom`** only when your CheXpert file includes **`dicom_id`** and you want image-aligned rows.

---

## 5. Canonical artifact pointers

| Purpose | Path |
|---------|------|
| ED cohort + temporal split | `artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet` |
| Shared binary eval table | `artifacts/manifests/cxr_pneumonia_training_table_u_ignore_temporal.parquet` |
| Clinical temporal (triage) | `artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet` |
| Image finetune rows | `artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet` |
| Multilabel pretrain rows | `artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet` |
| Primary labs merge (9,137) | `artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet` |
| CheXpert-stratified negative metrics (JSON) | `artifacts/evaluation/image_normal_vs_abnormal_negatives_stronger_lr_v3.json`, `artifacts/evaluation/multimodal_normal_vs_abnormal_negatives_stronger_lr_v3.json` (**§4.9**; regenerate with `--output-json`) |
| Thesis-style metric table (rounded @0.5) | `artifacts/evaluation/final_results_table.csv` + `artifacts/evaluation/final_result_note.txt` (from **`check_prediction_behavior`** outputs; see **§4.8** / **`current_state.md` §16.8**) |

Sensitivity / archived outputs: see **`docs/data_versions.md`**. Bootstrap summaries: **`artifacts/evaluation/`**.

