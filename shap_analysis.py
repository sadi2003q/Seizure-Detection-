# shap_analysis.py
"""R1-11: SHAP analysis on the leak-free LOSO model.

Computes SHAP values for each held-out patient's fold (using that fold's own trained
model, so SHAP never sees a model trained on the patient it explains), then aggregates:
  * top-N features by mean |SHAP value|, with direction of effect
  * example explanations for one correctly-classified seizure and one non-seizure window
  * stability: how often each of the full-data top-N features also appears in each
    fold's own top-N (answers "are the important features consistent across patients?")

Only supports XGBoost-family models (shap.TreeExplainer needs a tree model); pass the
already-built _XGB wrapper's underlying .m, not the Model wrapper itself, is handled
internally.
"""
import numpy as np, pandas as pd
import shap
from tqdm.auto import tqdm
from preprocessing import make_split, select_features


def _fold_patients(patients, held, n_val, seed):
    i = patients.index(held)
    others = [p for p in patients if p != held]
    val = [others[(i + k) % len(others)] for k in range(n_val)]
    train = [p for p in others if p not in val]
    return train, val


def loso_shap(raw, family, params=None, features='all', hold_out=None, n_val=2,
             top_n=10, seed=42):
    """
    raw    : from preprocessing.load_raw(csv)
    family : an XGBoostModel-style class (must expose ._XGB with a .m XGBClassifier)
    Returns a dict with:
      feature_names       : list of the 924 (or fewer, if `features` narrows the set) column names
      shap_by_fold         : {patient: (n_test_windows, n_features) SHAP array}
      mean_abs_shap        : (n_features,) mean |SHAP| across ALL folds' test windows pooled
      top_features         : DataFrame, top_n rows, feature / mean_abs_shap / mean_signed_shap
      stability             : DataFrame, top_n rows, feature / fraction_of_folds_in_that_folds_own_top_n
      example_seizure      : dict (patient, window index, per-feature SHAP for one correct seizure window)
      example_nonseizure   : dict (same, for one correct non-seizure window)
    """
    params = dict(params or {})
    patients = sorted(set(raw['patient']))
    held_list = patients if hold_out is None else list(hold_out)
    cols = select_features(raw, features)
    feature_names = [raw['cols'][i] for i in cols]

    shap_by_fold, fold_top_sets = {}, {}
    all_shap, all_X, all_y, all_patient = [], [], [], []
    example_seizure = example_nonseizure = None

    for held in tqdm(held_list, desc=f'SHAP {family.name} [{features}]'):
        train, val = _fold_patients(patients, held, n_val, seed)
        X, y, g = make_split(raw, train, val, [held], seed=seed, verbose=False)
        Xtr, Xte, yte = X['train'][:, cols], X['test'][:, cols], y['test']

        model = family.build(**params).fit(Xtr, y['train'])
        booster = model.m                       # the underlying xgb.XGBClassifier

        explainer = shap.TreeExplainer(booster, feature_perturbation='tree_path_dependent')
        sv = explainer.shap_values(Xte)
        if isinstance(sv, list):                 # older shap versions: list of length 2 for binary clf
            sv = sv[1]

        shap_by_fold[held] = sv
        fold_mean_abs = np.abs(sv).mean(axis=0)
        fold_top_sets[held] = set(np.argsort(fold_mean_abs)[::-1][:top_n])

        all_shap.append(sv); all_X.append(Xte); all_y.append(yte)
        all_patient.append(np.full(len(yte), held))

        proba = model.predict(Xte)
        pred = (proba >= 0.5).astype(int)
        correct_seiz = np.where((yte == 1) & (pred == 1))[0]
        correct_non  = np.where((yte == 0) & (pred == 0))[0]
        if example_seizure is None and len(correct_seiz):
            i = correct_seiz[0]
            example_seizure = dict(patient=held, index=int(i), probability=float(proba[i]),
                                   shap=dict(zip(feature_names, sv[i])))
        if example_nonseizure is None and len(correct_non):
            i = correct_non[0]
            example_nonseizure = dict(patient=held, index=int(i), probability=float(proba[i]),
                                      shap=dict(zip(feature_names, sv[i])))

    pooled_shap = np.concatenate(all_shap, axis=0)
    mean_abs = np.abs(pooled_shap).mean(axis=0)
    mean_signed = pooled_shap.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    top_features = pd.DataFrame({
        'feature': [feature_names[i] for i in order],
        'mean_abs_shap': mean_abs[order],
        'mean_signed_shap': mean_signed[order],
        'direction': ['higher -> more seizure-like' if mean_signed[i] > 0
                     else 'higher -> more non-seizure-like' for i in order],
    })

    full_top_set = set(order)
    stability_rows = []
    for feat_idx in order:
        frac = np.mean([feat_idx in fold_top_sets[h] for h in held_list])
        stability_rows.append(dict(feature=feature_names[feat_idx],
                                   fraction_of_folds_in_top_n=float(frac)))
    stability = pd.DataFrame(stability_rows)

    return dict(feature_names=feature_names, shap_by_fold=shap_by_fold,
                mean_abs_shap=mean_abs, top_features=top_features, stability=stability,
                example_seizure=example_seizure, example_nonseizure=example_nonseizure,
                held_out_patients=held_list, top_n=top_n)


def shap_report(res):
    """Print the SHAP result in a readable layout."""
    print('=' * 66)
    print(f"  SHAP top-{res['top_n']} features (pooled across {len(res['held_out_patients'])} "
          f"LOSO folds)")
    print('=' * 66)
    print(res['top_features'].round(4).to_string(index=False))
    print(f"\nStability: how often each top-{res['top_n']} feature (by pooled SHAP) also appears "
          f"in that SAME fold's own top-{res['top_n']} list:")
    print(res['stability'].round(2).to_string(index=False))
    mean_stab = res['stability']['fraction_of_folds_in_top_n'].mean()
    print(f"\nMean stability across the top-{res['top_n']} features: {mean_stab:.1%} of folds")
    for tag, ex in [('seizure', res['example_seizure']), ('non-seizure', res['example_nonseizure'])]:
        if ex is None:
            print(f"\nNo correctly-classified {tag} example was found in the checked folds.")
            continue
        top5 = sorted(ex['shap'].items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"\nExample {tag} window (patient {ex['patient']}, predicted probability "
              f"{ex['probability']:.3f}), top 5 contributing features:")
        for feat, val in top5:
            print(f"    {feat:30s} SHAP={val:+.4f}")
