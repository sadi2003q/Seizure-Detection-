# plots.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot(scores):
    """ROC, precision-recall, confusion matrix and per-patient accuracy for one model."""
    y, p, t = scores['y_test'], scores['p_test'], scores['at_tuned']
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))

    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2)
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[0].set(title=f"ROC (AUC {scores['roc_auc']:.3f})",
              xlabel='False positive rate', ylabel='True positive rate')

    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, lw=2)
    ax[1].set(title=f"Precision-Recall (AP {scores['pr_auc']:.3f})",
              xlabel='Recall', ylabel='Precision')

    cm = confusion_matrix(y, (p >= t['threshold']).astype(int))
    ax[2].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            colour = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax[2].text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color=colour)
    ax[2].set(title=f"Confusion @ {t['threshold']:.2f}", xticks=[0, 1], yticks=[0, 1],
              xticklabels=['Non-seiz', 'Seizure'], yticklabels=['Non-seiz', 'Seizure'])

    pp = scores['per_patient']
    if pp:
        ax[3].bar(list(pp), [v['accuracy'] for v in pp.values()], color='steelblue')
        ax[3].axhline(t['accuracy'], color='red', ls='--', label='overall')
        ax[3].legend()
    ax[3].set(title='Accuracy per test patient', ylim=(0, 1))

    plt.suptitle(scores['name'], fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_compare(scores_list):
    """ROC overlay + metric bars for several models."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for s in scores_list:
        fpr, tpr, _ = roc_curve(s['y_test'], s['p_test'])
        ax[0].plot(fpr, tpr, lw=2, label=f"{s['name']} ({s['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[0].legend()
    ax[0].set(title='ROC comparison', xlabel='False positive rate', ylabel='True positive rate')

    names = [s['name'] for s in scores_list]
    w, x = 0.25, np.arange(len(names))
    for k, m in enumerate(['accuracy', 'sensitivity', 'specificity']):
        ax[1].bar(x + (k - 1) * w, [s['at_tuned'][m] for s in scores_list], w, label=m)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylim(0, 1)
    ax[1].legend()
    ax[1].set_title('Metrics @ validation-tuned threshold')
    plt.tight_layout()
    plt.show()


def plot_loso(res):
    """Per-patient AUC with mean and CI, sensitivity/specificity per patient, AUC vs seizure fraction."""
    pp, sm = res['per_patient'], res['summary']
    order = pp.sort_values('roc_auc').reset_index(drop=True)
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))

    ax[0].barh(order['patient'], order['roc_auc'], color='steelblue')
    a = sm.loc['roc_auc']
    ax[0].axvline(a['mean'], color='red', ls='--', label=f"mean {a['mean']:.3f}")
    ax[0].axvspan(a['ci_low'], a['ci_high'], color='red', alpha=0.15, label='95% CI of mean')
    ax[0].axvline(0.5, color='k', ls=':', lw=1)
    ax[0].set(title='ROC-AUC per held-out patient', xlabel='ROC-AUC', xlim=(0.4, 1.0))
    ax[0].legend(loc='lower right')

    ax[1].scatter(order['specificity_at05'], order['sensitivity_at05'], s=60, color='darkorange')
    for _, r in order.iterrows():
        ax[1].annotate(r['patient'], (r['specificity_at05'], r['sensitivity_at05']),
                       fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[1].set(title='Sensitivity vs specificity per patient (threshold 0.5)',
              xlabel='Specificity', ylabel='Sensitivity', xlim=(0, 1.02), ylim=(0, 1.02))

    ax[2].scatter(pp['seizure_frac'], pp['roc_auc'], s=60, color='seagreen')
    for _, r in pp.iterrows():
        ax[2].annotate(r['patient'], (r['seizure_frac'], r['roc_auc']),
                       fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[2].axhline(0.5, color='k', ls=':', lw=1)
    ax[2].set(title='Does AUC depend on the patient\'s seizure fraction?',
              xlabel='Seizure fraction in the held-out patient', ylabel='ROC-AUC')

    plt.suptitle(f"LOSO: {res['label']}", fontweight='bold')
    plt.tight_layout()
    plt.show()
