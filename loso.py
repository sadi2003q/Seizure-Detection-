# loso.py
"""Leave-One-Subject-Out (LOSO) evaluation.

For every held-out patient, everything is redone WITHOUT that patient:
  * imputer + scaler are fitted on the training patients only,
  * the model is trained on the training patients only,
  * the decision threshold is chosen on a few OTHER patients (validation patients),
  * optional Optuna tuning (tune=True) sees the training patients only.
"""
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from experiment import _metrics, pick_threshold
from preprocessing import make_split, select_features

N_BOOT = 10000
_SHARED = ('sensitivity', 'specificity', 'accuracy', 'balanced_accuracy', 'precision', 'f1')
SCORES = ['roc_auc', 'pr_auc'] + [f'{m}_{t}' for t in ('at05', 'tuned') for m in _SHARED]


def mean_ci(values, seed=42, n_boot=N_BOOT):
    """Mean, sample SD and a 95% bootstrap CI of the mean, over patients."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(mean=np.nan, std=np.nan, ci_low=np.nan, ci_high=np.nan, min=np.nan, max=np.nan, n=0)
    boots = np.random.default_rng(seed).choice(v, size=(n_boot, len(v))).mean(axis=1)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)),
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def run_loso(raw, family, params=None, features='all', hold_out=None, n_val=2,
             tune=False, n_trials=10, seed=42, label=None):
    """
    raw      : from preprocessing.load_raw(csv)
    family   : a model class, e.g. XGBoostModel
    params   : model parameters (tune=False), or overrides applied AFTER tuning (tune=True)
    features : 'all' | 'conventional' (no spike features) | 'snn' (spike features only) |
               'no_time' / 'no_freq' / 'no_wavelet' / 'no_nonlinear' (drop one conventional
               domain, keep the rest + SNN) | 'time' / 'freq' / 'wavelet' / 'nonlinear'
               (that domain alone, for its own standalone AUC)
    hold_out : list of patients to hold out (default: every patient). Use 2 or 3 to test quickly.
    n_val    : number of other patients used ONLY to pick the threshold in each fold
    tune     : True -> Optuna runs INSIDE every fold on that fold's training patients only (very slow)
    """
    params = dict(params or {})
    patients = sorted(set(raw['patient']))
    held_list = patients if hold_out is None else list(hold_out)
    unknown = [p for p in held_list if p not in patients]
    if unknown:
        raise ValueError(f'unknown patient(s) in hold_out: {unknown}. Available: {patients}')
    cols = select_features(raw, features)

    rows, ys, ps, gs, best_by_patient = [], [], [], [], {}
    for held in tqdm(held_list, desc=f'LOSO {family.name} [{features}]'):
        i = patients.index(held)
        others = [p for p in patients if p != held]
        val = [others[(i + k) % len(others)] for k in range(n_val)]     # deterministic, never `held`
        train = [p for p in others if p not in val]
        assert held not in train and held not in val

        X, y, g = make_split(raw, train, val, [held], seed=seed, verbose=False)
        Xtr, Xva, Xte = (X[k][:, cols] for k in ('train', 'validation', 'test'))

        best = family.best_param(Xtr, y['train'], g['train'], n_trials=n_trials, seed=seed) if tune else {}
        best_by_patient[held] = best
        model = family.build(**{**best, **params}).fit(Xtr, y['train'])

        p_val, p_te, yt = model.predict(Xva), model.predict(Xte), y['test']
        thr = pick_threshold(y['validation'], p_val)
        row = dict(patient=held, n=int(len(yt)), seizure_frac=float(yt.mean()),
                   roc_auc=float(roc_auc_score(yt, p_te)),
                   pr_auc=float(average_precision_score(yt, p_te)),
                   threshold=thr, val_patients=', '.join(val))
        for tag, t in (('at05', 0.5), ('tuned', thr)):
            m = _metrics(yt, p_te, t)
            for k in _SHARED:
                row[f'{k}_{tag}'] = float(m[k])
        rows.append(row)
        ys.append(yt); ps.append(p_te); gs.append(np.full(len(yt), held))

    per_patient = pd.DataFrame(rows)
    summary = pd.DataFrame({c: mean_ci(per_patient[c], seed) for c in SCORES}).T
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    pooled = dict(roc_auc=float(roc_auc_score(y_all, p_all)),
                  pr_auc=float(average_precision_score(y_all, p_all)))
    return dict(name=family.name, features=features, label=label or f'{family.name} | {features}',
                params=params, tuned=tune, n_trials=n_trials, n_val=n_val,
                per_patient=per_patient, summary=summary, pooled=pooled,
                y_true=y_all, p_pred=p_all, patient_of_row=np.concatenate(gs),
                best_params=best_by_patient)
