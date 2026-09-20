# Epileptic Seizure Detection — patient-level split (no leakage)

## Files
```
main.ipynb          run this (top to bottom)
preprocessing.py    prepare_data(csv) -> X, y, g   (split by PATIENT, then normalise, then shuffle train)
base.py             shared contract + check_params (rejects wrong parameter names)
experiment.py       train(model, X_train, y_train)   evaluate(model, X_val, y_val, X_test, y_test, g_test)
report.py           report(scores), compare([scores, ...])
plots.py            plot(scores), plot_compare([scores, ...])
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
