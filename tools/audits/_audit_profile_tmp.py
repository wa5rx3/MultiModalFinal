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
report={}
for rel in files:
    p=base/rel
    if not p.exists():
        report[rel]={'exists':False}
        continue
    df=pd.read_parquet(p)
    info={'exists':True,'rows':int(len(df)),'cols':list(df.columns),'dtypes':{c:str(t) for c,t in df.dtypes.items()}}
    nulls=(df.isna().mean().sort_values(ascending=False).head(12)*100).round(2)
    info['top_null_pct']=nulls.to_dict()
    key_checks={}
    for key in [['subject_id'],['study_id'],['dicom_id'],['subject_id','study_id'],['subject_id','study_id','dicom_id']]:
        if all(k in df.columns for k in key):
            dup=int(df.duplicated(key).sum())
            key_checks['+'.join(key)]={'dups':dup,'nunique':int(df.drop_duplicates(key).shape[0])}
    info['key_checks']=key_checks
    if 'split' in df.columns:
        info['split_counts']=df['split'].value_counts(dropna=False).to_dict()
    if 'temporal_split' in df.columns:
        info['temporal_split_counts']=df['temporal_split'].value_counts(dropna=False).to_dict()
    info['sample']=df.head(2).to_dict(orient='records')
    report[rel]=info
x={}
for rel,name in [
('artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet','cohort'),
('artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet','labs_temporal')]:
    try:
        df=pd.read_parquet(base/rel)
        if {'subject_id','temporal_split'}.issubset(df.columns):
            g=df.groupby('subject_id')['temporal_split'].nunique()
            x[f'patient_split_leak_subjects_{name}']=int((g>1).sum())
    except Exception as e:
        x[f'error_{name}']=str(e)
out=base/'artifacts/logs/audit_profile_generated.json'
out.write_text(json.dumps({'tables':report,'cross_checks':x},indent=2,default=str))
print(str(out))
