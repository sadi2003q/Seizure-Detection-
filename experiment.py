# experiment.py
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix,
                             precision_score, recall_score)

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def train(model, X_train, y_train):
    """Train a model built in the notebook. Sees ONLY the training data."""
    if model.name == 'Mamba':               # Mamba shows its own per-epoch bar
        return model.fit(X_train, y_train)
    with tqdm(total=1, desc=f'Fitting {model.name}') as bar:
        model.fit(X_train, y_train)
        bar.update(1)
    return model


def _metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(threshold=thr, accuracy=accuracy_score(y, yhat),
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                precision=precision_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def evaluate(model, X_val, y_val, X_test, y_test, g_test=None):
    """Threshold is chosen on VALIDATION, then frozen and applied to TEST."""
    p_val, p_test = model.predict(X_val), model.predict(X_test)
    f1s = [f1_score(y_val, p_val >= t, zero_division=0) for t in GRID]
    thr = float(GRID[int(np.argmax(f1s))])

    per_patient = {}
    if g_test is not None:
        for pid in sorted(set(g_test)):
            m = g_test == pid
            yh = (p_test[m] >= thr).astype(int)
            per_patient[pid] = dict(
                n=int(m.sum()), seizure_frac=float(y_test[m].mean()),
                accuracy=float(accuracy_score(y_test[m], yh)),
                sensitivity=float(recall_score(y_test[m], yh, zero_division=0)),
                auc=float(roc_auc_score(y_test[m], p_test[m])) if len(set(y_test[m])) > 1 else np.nan)

    return dict(name=model.name, params=model.params,
                at_default=_metrics(y_test, p_test, 0.5),
                at_tuned=_metrics(y_test, p_test, thr),
                roc_auc=float(roc_auc_score(y_test, p_test)),
                pr_auc=float(average_precision_score(y_test, p_test)),
                val_f1=float(max(f1s)),
                threshold_at_edge=bool(thr in (GRID[0], GRID[-1])),
                per_patient=per_patient, y_test=y_test, p_test=p_test)
