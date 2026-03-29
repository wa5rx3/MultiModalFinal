import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# CHANGE THIS PATH
# =========================
# PRED_PATH = "artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv"
PRED_PATH = "artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv"
# PRED_PATH = "artifacts/models/multimodal_.../test_predictions.csv"

CHEXPERT_PATH = r"D:\mimic-cxr-2.0.0-chexpert.csv.gz"

# =========================
# LOAD DATA
# =========================
pred = pd.read_csv(PRED_PATH)
full = pd.read_csv(CHEXPERT_PATH)

df = pred.merge(
    full,
    on=["subject_id", "study_id"],
    how="left"
)

print("=== INITIAL ===")
print("rows:", len(df))

# =========================
# DEFINE ABNORMALITY
# =========================
abnormal_cols = [
    "Atelectasis",
    "Edema",
    "Pleural Effusion",
    "Consolidation",
    "Lung Opacity",
]

# remove rows where ALL abnormal labels are missing
mask_valid = ~df[abnormal_cols].isna().all(axis=1)
df = df[mask_valid].copy()

print("\n=== AFTER DROPPING ALL-NaN ABNORMAL ROWS ===")
print("rows:", len(df))

# define abnormal flag
df["any_abnormal"] = (df[abnormal_cols] == 1).any(axis=1)

# =========================
# SPLIT GROUPS
# =========================
pos = df[df["target"] == 1].copy()
normal_neg = df[(df["target"] == 0) & (df["any_abnormal"] == False)].copy()
abnormal_neg = df[(df["target"] == 0) & (df["any_abnormal"] == True)].copy()

print("\n=== COUNTS ===")
print("positives:", len(pos))
print("normal negatives:", len(normal_neg))
print("abnormal negatives:", len(abnormal_neg))

# =========================
# EVAL FUNCTION
# =========================
def eval_subset(pos_df, neg_df, name):
    sub = pd.concat([pos_df, neg_df], ignore_index=True).copy()

    y_true = sub["target"].astype(int).values
    y_prob = sub["pred_prob"].astype(float).values

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    print("\n" + name)
    print("n:", len(sub))
    print("positives:", (sub["target"] == 1).sum())
    print("negatives:", (sub["target"] == 0).sum())
    print("positive_rate:", sub["target"].mean())
    print("AUROC:", auroc)
    print("AUPRC:", auprc)

    return auroc, auprc

# =========================
# RUN EVALUATION
# =========================
auroc_norm, auprc_norm = eval_subset(
    pos, normal_neg, "positives vs normal negatives"
)

auroc_abn, auprc_abn = eval_subset(
    pos, abnormal_neg, "positives vs abnormal negatives"
)

# =========================
# COMPARE
# =========================
print("\n=== PERFORMANCE DROP ===")
print("Delta AUROC:", auroc_abn - auroc_norm)
print("Delta AUPRC:", auprc_abn - auprc_norm)