import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, roc_curve)

dfbase = pd.read_csv("data/baseline_predictions.csv")

LABEL_BASELINE = "pred_label"
PROB_BASELINE = "pred"
PLOT_DIR = Path("data").joinpath("plots")


if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    dfbase["pred_naive"] = 0

    acc_baseline = accuracy_score(dfbase["target"], dfbase["pred_label"]) * 100
    acc_naive = accuracy_score(dfbase["target"], dfbase["pred_naive"]) * 100

    rep_baseline = classification_report(dfbase["target"], dfbase["pred_label"])
    rep_naive = classification_report(dfbase["target"], dfbase["pred_naive"])

    roc_base = roc_auc_score(dfbase["target"], dfbase["pred_label"])
    roc_naive = roc_auc_score(dfbase["target"], dfbase["pred_naive"])

    print(f"""Accuracy (Baseline): {acc_baseline:.3f}%""")
    print(f"""Accuracy (Naive): {acc_naive:.3f}%""")

    print(f"""ROC AUC (Baseline): {roc_base:.3f}""")
    print(f"""ROC AUC (Naive): {roc_naive:.3f}""")

    print(f"""Classification Report (Baseline): {rep_baseline}""")
    print(f"""Classification Report (Naive): {rep_naive }""")


    # Generate ROC Curve Comparison

    fpr_baseline, tpr_baseline, _ = roc_curve(dfbase["target"],  dfbase["pred"])
    fpr_naive, tpr_naive, _ = roc_curve(dfbase["target"],  dfbase["pred_naive"])

    roc_base = roc_auc_score(dfbase["target"], dfbase["pred_label"])
    roc_naive = roc_auc_score(dfbase["target"], dfbase["pred_naive"])

    plt.plot(fpr_naive, tpr_naive, label=f"Naive {roc_naive:.3f}", alpha=0.7, color="black", linestyle="--")
    plt.plot(fpr_baseline, tpr_baseline, label=f"Baseline {roc_base:.3f}", color="Blue", alpha=0.7)
    plt.legend(loc="upper left")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.title('ROC Curve Comparison')
    plt.grid()
    fname = PLOT_DIR.joinpath("roc_curve_comparison.png")
    plt.savefig(fname)
    plt.show()
