# Epileptic Seizure Detection — patient-level split (no leakage)

## Files
```
main.ipynb          run this (top to bottom)
preprocessing.py    prepare_data(csv) -> X, y, g   (fixed split by PATIENT), plus load_raw / make_split / select_features (used by LOSO)
base.py             shared contract + check_params (rejects wrong parameter names)
experiment.py       train(model, X_train, y_train)   evaluate(model, X_val, y_val, X_test, y_test, g_test)
loso.py             run_loso(raw, ModelClass, params=...)   Leave-One-Subject-Out
report.py           report(scores), compare([...]), loso_report(res), loso_compare([res, ...])
plots.py            plot(scores), plot_compare([...]), plot_loso(res)
models/
  logistic.py       LogisticModel
  xgboost_model.py  XGBoostModel
  mamba_model.py    MambaModel   (optional, needs PyTorch)
```

## Kaggle setup
1. Upload this whole folder (or the zip) as a Kaggle Dataset, or copy the files into `/kaggle/working`.
2. Upload `balanced_dataset8.csv` as a Kaggle Dataset.
3. In `main.ipynb`, cell 2, edit `PROJECT_DIR` and `CSV_PATH`.
4. Run top to bottom. Skip the Mamba section if you do not use it.

## The flow (same for every model)
```python
X, y, g = prepare_data(CSV_PATH)

# Optuna OFF: type the parameters
model = XGBoostModel.build(n_estimators=500, max_depth=8)

# Optuna ON: tune, then build with the best parameters
best  = XGBoostModel.best_param(X['train'], y['train'], g['train'], n_trials=20)
model = XGBoostModel.build(**best)

model  = train(model, X['train'], y['train'])
scores = evaluate(model, X['validation'], y['validation'], X['test'], y['test'], g['test'])
report(scores); plot(scores)
```
`g` is the patient id of each row. Optuna uses it to hold out whole patients in its cross-validation.

## Per-patient normalisation (optional)
`prepare_data(CSV_PATH, per_patient_norm=None | 'interictal' | 'all')`
- `None` (default): one global scaler fitted on the training patients.
- `'interictal'`: each recording group (`subject_id`) is scaled with the median/IQR of its own seizure-free
  (label 0) windows, then the same global scaler is applied. Assumes a seizure-free calibration recording
  for every new patient. State this in the paper. Only label-0 windows of the test patients are used for the
  statistics, never their seizure windows, and never for training.
- `'all'`: same, but the statistics use all of the group's windows. No labels are used, but the result
  depends on each patient's seizure fraction (13% to 90% in this CSV), which distorts the baseline.
Report all versions you ran. `main.ipynb` section 2b compares the three with Logistic Regression.

## Annotation extension test (R1-7)
Reviewer 1 asked whether extending the positive class by 10s before onset / 5s after offset
inflates performance, versus using only the true seizure interval. Needs a CSV with an
`in_core_ictal` column (label==1 rows split into core-ictal vs pre/post-ictal-only).
```python
from annotation import load_raw_annotation
from loso import run_loso
from report import loso_compare

r_ext  = load_raw_annotation(CSV_PATH, definition='extended')   # current manuscript definition
r_core = load_raw_annotation(CSV_PATH, definition='core')       # true ictal interval only

res_ext  = run_loso(r_ext,  XGBoostModel, params={}, label='extended (label==1)')
res_core = run_loso(r_core, XGBoostModel, params={}, label='core-ictal only')
loso_compare([res_ext, res_core], baseline=0)     # paired per-patient AUC difference + Wilcoxon
```
`definition='core'` drops the pre/post-ictal-only windows (label==1 but not in_core_ictal) entirely --
they are neither its positive nor its negative class. `feats`/`patient`/`cols` come out in the same
shape as `preprocessing.load_raw()`, so `run_loso` needs no changes. A per-patient warning is printed
if a patient has under 10 positive windows in the `'core'` definition.

## Leave-One-Subject-Out (LOSO)
```python
raw = load_raw(CSV_PATH)                                       # read the CSV once
res = run_loso(raw, XGBoostModel, params={})                   # 20 folds, one per real patient
loso_report(res); plot_loso(res)
# SNN ablation, same classifier and folds, paired statistics:
runs = [run_loso(raw, XGBoostModel, features=f, label=f) for f in ['conventional', 'snn', 'all']]
loso_compare(runs, baseline=0)
```
In every fold the imputer, scaler, model and threshold are fitted WITHOUT the held-out patient
(the threshold comes from `n_val=2` other patients). Tested: rewriting the held-out patient's features and
labels leaves everything the model sees bit-identical.
- Hyperparameters: fixed, or `tune=True` (Optuna inside every fold, very slow). Do NOT reuse the Optuna values
  from the fixed split: they were tuned on patients that are held out in some LOSO folds.
- The per-fold threshold comes from only 2 patients and can be very unstable. Use ROC-AUC and the @0.5
  columns as the headline numbers, or raise `n_val`.
- `hold_out=['chb16', 'chb17']` runs just those folds (quick test). `features='conventional'|'snn'|'all'` is the ablation.
- Not fixable from the CSV: it was class-balanced by random undersampling BEFORE splitting, so balancing is
  not fold-independent, and false alarms per hour cannot be computed.

## Split (patient level, no overlap)
- Test: chb16, chb17 (a/b/c together), chb18, chb19, chb20, chb22
- Validation: chb14, chb15 (threshold selection only)
- Train: everything else (chb21 is folded into chb01, the same patient)

## Adding a new model
Copy `models/logistic.py`, rename the classes, change `search_space` and `_build`, add one line to
`models/__init__.py`. Nothing else changes.

## Things to state honestly in the paper
- The CSV is class-balanced and shuffled, so accuracy/precision overstate real monitoring performance,
  and false alarms per hour cannot be computed.
- Mamba here models the 22 channels as the sequence (the CSV has no time order). If the pure-PyTorch
  fallback is used instead of the official `mamba_ssm`, say so.
- Test set = 6 patients: report the per-patient spread, not only the average.
- The decision threshold is chosen on the validation patients by balanced accuracy, then frozen.
  Validation is only 2 patients, so the threshold can be unstable.
