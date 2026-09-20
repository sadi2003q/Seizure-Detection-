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
    lines.append('Reminder: the test set is class-balanced, so precision/accuracy here overstate '
                 'real-world monitoring performance.')
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
