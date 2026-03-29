# Triage Feature Policy

## Purpose
Define the first clinical feature set for the ED-linked multimodal pneumonia project.

## Current feature groups

### Numeric triage variables
- temperature
- heartrate
- resprate
- o2sat
- sbp
- dbp
- pain
- acuity

Physiological variables are clipped to clinically plausible ranges 
in build_triage_features.py before any downstream use.

### Categorical variables
- gender
- race
- arrival_transport

## Leakage-related update

- `disposition` is removed from the active baseline feature set.
- Rationale: disposition can reflect post-`t0` downstream ED decisions and is therefore leakage-prone for strict imaging-time prediction.
- If disposition is ever reintroduced, it must be justified as time-safe for the exact prediction timestamp and validated with source timestamp semantics.

### Text variable
- chiefcomplaint

## Initial baseline policy

### Clinical-only baseline v1
Use:
- numeric triage variables
- categorical variables
- missingness indicators for numeric variables

Do NOT use:
- chiefcomplaint text yet
- disposition

## Missing data handling
- Numeric variables: median imputation
- Add explicit missingness flags
- Categorical variables: fill missing with "UNKNOWN" if needed
- Text: excluded in v1 baseline

Implementation note:
- No global pre-split imputation should be written into dataset tables.
- Imputation is fit on train only inside model pipelines.

## Multimodal v1 (image + triage)
The first multimodal model uses the **same numeric + categorical triage columns** (and missingness flags) as the clinical baselines, via a sklearn `ColumnTransformer` **fit on the training split only** inside `src/training/train_multimodal_pneumonia.py` — see `docs/current_state.md` §9.7. Text (`chiefcomplaint`) remains out of scope for this v1 fusion run.

## Rationale
This keeps the first clinical baseline simple, time-safe, and interpretable.