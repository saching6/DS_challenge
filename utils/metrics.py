import numpy as np 
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             brier_score_loss)


def evaluate(y_ground_truth, y_pred, model_name):
    """PR-AUC primary. Brier reports whether the probabilities can be trusted."""
    y_ground_truth = np.asarray(y_ground_truth)
    return {'model': model_name,
            'pr_auc': average_precision_score(y_ground_truth, y_pred),
            'roc_auc': roc_auc_score(y_ground_truth, y_pred),
            'brier': brier_score_loss(y_ground_truth, y_pred)}

def plot_curves(y, results, path='curves.png'):
    """ROC and PR curves for one or more models on the same axes.
    `results` maps a model name to its predicted probabilities on `y`.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    base_rate = y.mean()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
 
    for name, p in results.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr,
                    label=f'{name} (AUC {roc_auc_score(y, p):.3f})')
        prec, rec, _ = precision_recall_curve(y, p)
        ax_pr.plot(rec, prec,
                   label=f'{name} (AP {average_precision_score(y, p):.3f})')
 
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax_roc.set(xlabel='false positive rate', ylabel='true positive rate',
               title='ROC')
    ax_roc.legend(loc='lower right', fontsize=9)
 
    ax_pr.axhline(base_rate, ls='--', color='k', alpha=0.4,
                  label=f'base rate {base_rate:.3f}')
    ax_pr.set(xlabel='recall', ylabel='precision', title='Precision-Recall')
    ax_pr.legend(loc='upper right', fontsize=9)
 
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)
    return path


def operating_point(y, p, threshold):
    """What ops is actually promised: how many incidents get flagged, how many
    of those are real, and how many real ones we catch."""
    y = np.asarray(y)
    flagged = p >= threshold
    return {'flagged_pct': 100 * flagged.mean(),
            'precision': y[flagged].mean() if flagged.any() else np.nan,
            'recall': y[flagged].sum() / y.sum(),
            'base_rate': y.mean()}
 

def reliability(y, p, bins=8):
    """Checks the quality of calibration"""
    df = pd.DataFrame({'y': np.asarray(y), 'p': p})
    df['bin'] = pd.qcut(df.p.rank(method='first'), bins, labels=False)
    return df.groupby('bin').agg(n=('y', 'size'),
                                mean_pred=('p', 'mean'),
                                observed=('y', 'mean')).round(4)

def plot_reliability(y, results, bins=8, path='reliability.png'):
    """Reliability (calibration) plot for one or more models."""
    import matplotlib.pyplot as plt
 
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6, 6))
    tables = {name: reliability(y, p, bins) for name, p in results.items()}
    for name, table in tables.items():
        ax.plot(table.mean_pred, table.observed, 'o-', label=name)
 
    hi = max(max(t.mean_pred.max(), t.observed.max()) for t in tables.values())
    hi = float(hi) * 1.05
    ax.plot([0, hi], [0, hi], 'k--', alpha=0.4, label='perfect')
    ax.set(xlim=(0, hi), ylim=(0, hi),
           xlabel='mean predicted probability',
           ylabel='observed fraction major',
           title='Reliability (calibration)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)
    return path