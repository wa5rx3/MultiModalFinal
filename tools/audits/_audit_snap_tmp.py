import json
from pathlib import Path
import pandas as pd
base=Path(r'c:/MultiModalFinal')
files=[
'artifacts/manifests/mimic_cxr_manifest.parquet',
'artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet',
'artifacts/manifests/cxr_final_ed_cohort.parquet',
'artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet',
'artifacts/manifests/cxr_ed_triage_features.parquet',
'artifacts/manifests/cxr_ed_triage_model_table.parquet',
'artifacts/manifests/cxr_pneumonia_labels.parquet',
'artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet',
'artifacts/manifests/cxr_pneumonia_training_table_u_zero.parquet',
'artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore.parquet',
'artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet',
'artifacts/tables/cohort_labevents_hadm_only.parquet',
'artifacts/tables/cxr_lab_features_hadm_only.parquet',
'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only.parquet',
'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet',
'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap.parquet',
'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap_temporal.parquet',
'artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet',
'artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet',
]
rows=[]
for rel in files:
  p=base/rel
  if not p.exists():
    rows.append({'file':rel,'exists':False}); continue
  df=pd.read_parquet(p)
  sample=df.head(1).to_dict(orient='records')[0] if len(df)>0 else {}
  # keep only first 8 sample fields
  sm={k:sample[k] for k in list(sample)[:8]}
  rows.append({
    'file':rel,
    'exists':True,
    'rows':len(df),
    'cols':len(df.columns),
    'first_cols':list(df.columns[:12]),
    'sample_head':sm,
    'null_top5':(df.isna().mean().sort_values(ascending=False).head(5)*100).round(2).to_dict()
  })
out=base/'artifacts/logs/audit_table_snapshots.json'
out.write_text(json.dumps(rows,indent=2,default=str))
print(out)
