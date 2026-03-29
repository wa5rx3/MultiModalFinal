import json
from pathlib import Path
import pandas as pd
base=Path(r'c:/MultiModalFinal')
files=[
('manifest','artifacts/manifests/mimic_cxr_manifest.parquet'),
('primary','artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet'),
('ed','artifacts/manifests/cxr_final_ed_cohort.parquet'),
('ed_temp','artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet'),
('labels','artifacts/manifests/cxr_pneumonia_labels.parquet'),
('train_u','artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet'),
('train_u_temp','artifacts/manifests/cxr_pneumonia_training_table_u_ignore_temporal.parquet'),
('clin_temp','artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet'),
('img_ft','artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet'),
('pre_split','artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet'),
('multi_tbl','artifacts/manifests/mimic_cxr_multilabel_pretrain_table.parquet'),
('labs_ev','artifacts/tables/cohort_labevents_hadm_only.parquet'),
('labs_feat','artifacts/tables/cxr_lab_features_hadm_only.parquet'),
('clin_labs_t','artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet'),
('overlap_t','artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap_temporal.parquet'),
]
out={}
for name,rel in files:
    p=base/rel
    if not p.exists():
        out[name]={'exists':False}; continue
    df=pd.read_parquet(p)
    sm=df.head(1).to_dict('orient='records')[0] if len(df) else {}
    sm={k:sm[k] for k in list(sm)[:6]}
    kc={}
    if 'subject_id' in df.columns and 'study_id' in df.columns:
        kc['study_dup']=int(df.duplicated(['subject_id','study_id']).sum())
    if 'dicom_id' in df.columns:
        kc['dicom_dup']=int(df.duplicated(['dicom_id']).sum())
    out[name]={'exists':True,'rows':len(df),'cols':len(df.columns),'sample':sm,'keys':kc}
    if 'temporal_split' in df.columns:
        out[name]['temporal_split']=df['temporal_split'].value_counts().to_dict()
# align train_u_temp vs img_ft
if out.get('train_u_temp',{}).get('exists') and out.get('img_ft',{}).get('exists'):
    a=pd.read_parquet(base/'artifacts/manifests/cxr_pneumonia_training_table_u_ignore_temporal.parquet')
    b=pd.read_parquet(base/'artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet')
    keys=['subject_id','study_id']
    ma=set(map(tuple,a[keys].values.tolist()))
    mb=set(map(tuple,b[keys].values.tolist()))
    out['align_train_temp_vs_img_ft']={'same_rows':len(a)==len(b),'set_equal':ma==mb,'only_in_a':len(ma-mb),'only_in_b':len(mb-ma)}
# patient leak
for nm,rel in [('ed_temp','artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet')]:
    df=pd.read_parquet(base/rel)
    g=df.groupby('subject_id')['temporal_split'].nunique()
    out['patient_leak_ed_temp']=int((g>1).sum())
print(json.dumps(out,indent=2,default=str))
