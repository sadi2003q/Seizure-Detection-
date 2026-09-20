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
            ax[2].text(j, i, cm[i, j], ha='center', va='center', fontsize=13)
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
