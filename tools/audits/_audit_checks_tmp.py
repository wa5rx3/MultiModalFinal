import json
from pathlib import Path
import pandas as pd
base=Path(r'c:/MultiModalFinal')

# Load key tables
m=pd.read_parquet(base/'artifacts/manifests/mimic_cxr_manifest.parquet')
pf=pd.read_parquet(base/'artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet')
ed=pd.read_parquet(base/'artifacts/manifests/cxr_final_ed_cohort.parquet')
edt=pd.read_parquet(base/'artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet')
tri=pd.read_parquet(base/'artifacts/manifests/cxr_ed_triage_features.parquet')
trim=pd.read_parquet(base/'artifacts/manifests/cxr_ed_triage_model_table.parquet')
lablbl=pd.read_parquet(base/'artifacts/manifests/cxr_pneumonia_labels.parquet')
train_u=pd.read_parquet(base/'artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet')
clin=pd.read_parquet(base/'artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore.parquet')
clint=pd.read_parquet(base/'artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet')
lab_ev=pd.read_parquet(base/'artifacts/tables/cohort_labevents_hadm_only.parquet')
lab_feat=pd.read_parquet(base/'artifacts/tables/cxr_lab_features_hadm_only.parquet')
lab_full=pd.read_parquet(base/'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only.parquet')
lab_full_t=pd.read_parquet(base/'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet')
lab_overlap=pd.read_parquet(base/'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap.parquet')
lab_overlap_t=pd.read_parquet(base/'artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap_temporal.parquet')
pre_split=pd.read_parquet(base/'artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet')
pre_tbl=pd.read_parquet(base/'artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet')

# checks
out={}
out['row_flow']={
'manifest':len(m),'primary_frontal':len(pf),'final_ed':len(ed),'labels':len(lablbl),
'u_ignore':len(train_u),'clinical':len(clin),'clinical_temporal':len(clint),
'lab_events_hadm_only':len(lab_ev),'lab_features_hadm_only':len(lab_feat),
'clinical_labs_full':len(lab_full),'clinical_labs_full_temporal':len(lab_full_t),
'clinical_labs_overlap':len(lab_overlap),'clinical_labs_overlap_temporal':len(lab_overlap_t),
'pretrain_split_manifest':len(pre_split),'multilabel_pretrain_table':len(pre_tbl)
}

# uniqueness and integrity checks
out['uniqueness']={
'primary_study_unique_dups':int(pf.duplicated(['subject_id','study_id']).sum()),
'final_ed_study_unique_dups':int(ed.duplicated(['subject_id','study_id']).sum()),
'triage_model_study_unique_dups':int(trim.duplicated(['subject_id','study_id']).sum()),
'labels_study_unique_dups':int(lablbl.duplicated(['subject_id','study_id']).sum()),
'clinical_temporal_study_unique_dups':int(clint.duplicated(['subject_id','study_id']).sum()),
'lab_features_study_unique_dups':int(lab_feat.duplicated(['subject_id','study_id']).sum()),
'pretrain_table_dicom_dups':int(pre_tbl.duplicated(['dicom_id']).sum()) if 'dicom_id' in pre_tbl else None
}

# split leakage checks
for name,df,col in [('ed_temporal',edt,'temporal_split'),('clinical_temporal',clint,'temporal_split'),('lab_full_temporal',lab_full_t,'temporal_split'),('lab_overlap_temporal',lab_overlap_t,'temporal_split')]:
    g=df.groupby('subject_id')[col].nunique(dropna=True)
    out.setdefault('patient_split_leak',{})[name]=int((g>1).sum())

# lab time safety check
ed_keys=ed[['subject_id','study_id','t0','hadm_id']].copy(); ed_keys['t0']=pd.to_datetime(ed_keys['t0'],errors='coerce')
le=lab_ev.merge(ed_keys,on=['subject_id','study_id','hadm_id'],how='left')
le['charttime']=pd.to_datetime(le['charttime'],errors='coerce')
out['lab_time_safety']={
'missing_t0_after_join':int(le['t0'].isna().sum()),
'post_t0_rows':int((le['charttime']>le['t0']).sum()),
'older_than_24h_rows':int((le['charttime']<le['t0']-pd.Timedelta(hours=24)).sum())
}

# hadm mismatch risk in hadm-only
out['lab_hadm_sanity']={
'null_hadm_in_hadm_only':int(lab_ev['hadm_id'].isna().sum())
}

# label missingness + effective set
out['label_counts']={
'positive':int((lablbl['pneumonia_chexpert_raw']==1).sum()),
'negative':int((lablbl['pneumonia_chexpert_raw']==0).sum()),
'uncertain':int((lablbl['pneumonia_chexpert_raw']==-1).sum()),
'missing':int(lablbl['pneumonia_chexpert_raw'].isna().sum())
}

# overlap coverage in full table
lab_value_cols=[c for c in lab_feat.columns if c not in ['subject_id','study_id'] and not c.endswith('_missing')]
any_lab=lab_full_t[lab_value_cols].notna().any(axis=1)
out['lab_coverage']={
'rows_any_lab':int(any_lab.sum()),
'rows_no_lab':int((~any_lab).sum()),
'any_lab_rate':float(any_lab.mean())
}

# output
p=base/'artifacts/logs/audit_consistency_checks.json'
p.write_text(json.dumps(out,indent=2,default=str))
print(p)
