# report.py
import pandas as pd

try:                                   # works in Jupyter/Kaggle; falls back to print elsewhere
    from IPython.display import display
except Exception:
    display = print


def report(scores):
    """Print one model's full result in a readable layout."""
    a, t = scores['at_default'], scores['at_tuned']
    print('=' * 62)
    print(f"  {scores['name']}  -  results on UNSEEN test patients")
    print('=' * 62)
    print('Parameters used:', scores['params'] or 'defaults')
    rows = [['Accuracy', a['accuracy'], t['accuracy']],
            ['Sensitivity (recall)', a['sensitivity'], t['sensitivity']],
            ['Specificity', a['specificity'], t['specificity']],
            ['Precision', a['precision'], t['precision']],
            ['F1', a['f1'], t['f1']]]
    df = pd.DataFrame(rows, columns=['Metric', '@ 0.50',
                                     f"@ {t['threshold']:.2f} (val-tuned)"]).round(4)
    print(df.to_string(index=False))
    print(f"\nROC-AUC {scores['roc_auc']:.4f} | PR-AUC {scores['pr_auc']:.4f} | "
          f"TP {t['tp']} FP {t['fp']} TN {t['tn']} FN {t['fn']}")
    if scores['per_patient']:
        print('\nPer-patient (tuned threshold):')
        print(pd.DataFrame(scores['per_patient']).T.round(3).to_string())
    print('\n' + verdict(scores))


def verdict(scores):
    """Plain-English summary with honest warnings."""
    t, pp = scores['at_tuned'], scores['per_patient']
    lines = [f"{scores['name']} reaches {t['accuracy']:.1%} accuracy on patients it has never seen "
             f"(sensitivity {t['sensitivity']:.1%}, specificity {t['specificity']:.1%})."]
    if pp:
        accs = [v['accuracy'] for v in pp.values()]
        worst = min(pp, key=lambda k: pp[k]['accuracy'])
        if max(accs) - min(accs) > 0.15:
            lines.append(f"Warning: performance varies a lot between patients "
                         f"({min(accs):.1%} to {max(accs):.1%}); the worst is {worst}. "
                         f"Report the spread, not just the average.")
    if scores['threshold_at_edge']:
        lines.append('Warning: the validation threshold sits at the edge of the search grid.')
    if t['sensitivity'] < 0.90:
        lines.append('Note: sensitivity is below 90%, which matters for a seizure detector.')
    frac = float(scores['y_test'].mean())
    lines.append(f'Reminder: the test set is {frac:.1%} seizure, so always guessing the majority class '
                 f'would score {max(frac, 1 - frac):.1%} accuracy. It is also far richer in seizures than '
                 f'real monitoring, so accuracy and precision overstate real-world performance.')
    return '\n'.join(lines)


def compare(scores_list):
    """Side-by-side table for several models + a plain-English ranking."""
    rows = []
    for s in scores_list:
        t = s['at_tuned']
        rows.append(dict(Model=s['name'], Accuracy=t['accuracy'], Sensitivity=t['sensitivity'],
                         Specificity=t['specificity'], F1=t['f1'], ROC_AUC=s['roc_auc'],
                         PR_AUC=s['pr_auc'], Acc_at_0_5=s['at_default']['accuracy']))
    df = (pd.DataFrame(rows).sort_values('ROC_AUC', ascending=False)
          .round(4).reset_index(drop=True))
    display(df)
    best = df.iloc[0]
    msg = f"Best by ROC-AUC: {best['Model']} ({best['ROC_AUC']:.4f})."
    if len(df) > 1:
        gap = best['ROC_AUC'] - df.iloc[1]['ROC_AUC']
        msg += f" Gap to {df.iloc[1]['Model']}: {gap:.4f}"
        msg += ' (very small; likely within noise across seeds/patients).' if gap < 0.01 else '.'
    print(msg)
    return df


# ═══════════════════════════ LOSO ═══════════════════════════
def _pm(row, digits=3):
    """'mean ± sd [95% CI low, high]  (min-max)'"""
    return (f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}  "
            f"[{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]  "
            f"({row['min']:.2f}-{row['max']:.2f})")


def loso_report(res):
    """Per-patient table, mean ± SD with 95% CI across patients, and a plain-English verdict."""
    pp, sm = res['per_patient'], res['summary']
    n = len(pp)
    print('=' * 74)
    print(f"  LOSO  |  {res['label']}  |  {n} held-out patient(s)")
    print('=' * 74)
    if res['tuned']:
        print(f"Hyperparameters: tuned INSIDE every fold on that fold's training patients only "
              f"(nested Optuna, {res['n_trials']} trials). Overrides: {res['params'] or 'none'}")
    else:
        print(f"Hyperparameters: fixed, not tuned in this run: {res['params'] or 'model defaults'}")
        print("  -> valid only if they were NOT chosen by looking at results on these patients.")
    print(f"Threshold per fold: chosen on {res['n_val']} other patients; the 0.5 columns use no threshold selection.\n")

    show = pp[['patient', 'n', 'seizure_frac', 'roc_auc', 'pr_auc',
               'sensitivity_at05', 'specificity_at05', 'balanced_accuracy_at05',
               'threshold', 'sensitivity_tuned', 'specificity_tuned', 'balanced_accuracy_tuned']].copy()
    show.columns = ['patient', 'n', 'seiz%', 'AUC', 'PR-AUC', 'sens@.5', 'spec@.5', 'balAcc@.5',
                    'thr', 'sens@thr', 'spec@thr', 'balAcc@thr']
    print(show.round(3).to_string(index=False))

    print('\nAcross patients (mean ± SD  [95% bootstrap CI]  (min-max)):')
    labels = {'roc_auc': 'ROC-AUC', 'pr_auc': 'PR-AUC',
              'sensitivity_at05': 'Sensitivity @0.5', 'specificity_at05': 'Specificity @0.5',
              'balanced_accuracy_at05': 'Balanced acc @0.5', 'precision_at05': 'Precision @0.5',
              'f1_at05': 'F1 @0.5', 'sensitivity_tuned': 'Sensitivity @thr',
              'specificity_tuned': 'Specificity @thr', 'balanced_accuracy_tuned': 'Balanced acc @thr'}
    for k, name in labels.items():
        print(f'  {name:<20s} {_pm(sm.loc[k])}')
    print(f"\nPooled over all held-out windows: ROC-AUC {res['pooled']['roc_auc']:.3f}, "
          f"PR-AUC {res['pooled']['pr_auc']:.3f}  (secondary; mixes patients' score scales)")

    auc = sm.loc['roc_auc']
    weak = pp.loc[pp['roc_auc'] < 0.70, 'patient'].tolist()
    strong = pp.loc[pp['roc_auc'] >= 0.85, 'patient'].tolist()
    edge = int(((pp['threshold'] <= 0.01) | (pp['threshold'] >= 0.99)).sum())
    lines = [f"Across {n} unseen patient(s) the mean ROC-AUC is {auc['mean']:.3f} "
             f"(SD {auc['std']:.3f}, 95% CI {auc['ci_low']:.3f} to {auc['ci_high']:.3f}), "
             f"ranging from {auc['min']:.3f} to {auc['max']:.3f}."]
    if weak:
        lines.append(f"{len(weak)} patient(s) have AUC below 0.70 (model barely works for them): {', '.join(weak)}.")
    if strong:
        lines.append(f"{len(strong)} patient(s) have AUC of 0.85 or more: {', '.join(strong)}.")
    if edge:
        lines.append(f"Warning: in {edge} fold(s) the chosen threshold sits at the edge of the grid; "
                     f"prefer the threshold-free AUC and the @0.5 columns.")
    lines.append("Independence: in every fold the imputer, scaler, model and threshold were fitted without "
                 "the held-out patient.")
    lines.append("Limitations to state: the CSV was class-balanced by random undersampling BEFORE splitting, "
                 "so balancing is not fold-independent and false alarms per hour cannot be computed; "
                 "the mean weights every patient equally whatever its size.")
    print('\n' + '\n'.join(lines))


def loso_compare(runs, baseline=0, seed=42):
    """Compare LOSO runs made on the SAME held-out patients (e.g. the SNN ablation).
    Uses PAIRED per-patient AUC differences against runs[baseline]."""
    import numpy as np
    from scipy.stats import wilcoxon
    from loso import mean_ci
    base = runs[baseline]['per_patient'].set_index('patient')['roc_auc']
    rows = []
    for k, r in enumerate(runs):
        a = r['per_patient'].set_index('patient')['roc_auc']
        common = a.index.intersection(base.index)
        s = mean_ci(a[common], seed)
        row = {'Run': r['label'], 'n': len(common),
               'AUC mean ± SD': f"{s['mean']:.3f} ± {s['std']:.3f}",
               '95% CI': f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"}
        if k == baseline:
            row.update({'diff vs baseline': '-', 'diff 95% CI': '-', 'Wilcoxon p': '-', 'better in': '-'})
        else:
            d = (a[common] - base[common]).to_numpy()
            ds = mean_ci(d, seed)
            try:
                p = f'{wilcoxon(d).pvalue:.3f}'
            except ValueError:
                p = 'n/a'
            row.update({'diff vs baseline': f"{ds['mean']:+.3f}",
                        'diff 95% CI': f"[{ds['ci_low']:+.3f}, {ds['ci_high']:+.3f}]",
                        'Wilcoxon p': p, 'better in': f'{int((d > 0).sum())}/{len(d)} patients'})
        rows.append(row)
    df = pd.DataFrame(rows)
    display(df)
    print(f"Baseline = '{runs[baseline]['label']}'. Differences are paired per patient. "
          f"A difference whose 95% CI includes 0 is not evidence of a real effect.")
    return df
