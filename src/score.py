import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

LABEL = "target"
LABEL_BASELINE = "label_baseline"
PRED_BASELINE = "pred_baseline"
PRED_MODEL = "pred"
LABEL_MODEL = "pred_label"
PLOT_DIR = Path("data").joinpath("plots")

DIR = Path("data")
FNAME_BASE = DIR.joinpath("baseline_predictions_cv.csv")
# FNAME_BASE = DIR.joinpath("data/baseline_predictions.csv")
FNAME_MODEL = DIR.joinpath("prediction.snap.parquet")
FNAME_MODEL = DIR.joinpath("cv_scores.snap.parquet")

dfbase = pd.read_csv(FNAME_BASE).rename(
    columns={"pred": "pred_baseline", "pred_label": "label_baseline"}
)

THRESH = .03
dfpred = pd.read_parquet(FNAME_MODEL)
if "target" in dfpred.columns and PRED_MODEL not in dfpred.columns:
    dfpred = dfpred.rename(columns={"target": PRED_MODEL})
if LABEL_MODEL not in dfpred.columns:
    dfpred[LABEL_MODEL] = np.where(dfpred[PRED_MODEL] >= THRESH, 1, 0)
    
dfscore = dfbase.merge(dfpred[["id"] + [PRED_MODEL, LABEL_MODEL]], on="id", how="inner")


if __name__ == "__main__":

    os.makedirs(PLOT_DIR, exist_ok=True)

    dfscore["pred_naive"] = 0

    acc_naive = accuracy_score(dfscore[LABEL], dfscore["pred_naive"]) * 100
    acc_baseline = accuracy_score(dfscore[LABEL], dfscore[LABEL_BASELINE]) * 100
    acc_model = accuracy_score(dfscore[LABEL], dfscore[LABEL_MODEL]) * 100

    rep_naive = classification_report(dfscore[LABEL], dfscore["pred_naive"])
    rep_baseline = classification_report(dfscore[LABEL], dfscore[LABEL_BASELINE])
    rep_model = classification_report(dfscore[LABEL], dfscore[LABEL_MODEL])

    roc_naive = roc_auc_score(dfscore[LABEL], dfscore["pred_naive"])
    roc_base = roc_auc_score(dfscore[LABEL], dfscore[LABEL_BASELINE])
    roc_model = roc_auc_score(dfscore[LABEL], dfscore[LABEL_MODEL])

    print(f"""Accuracy (Naive): {acc_naive:.3f}%""")
    print(f"""Accuracy (Baseline): {acc_baseline:.3f}%""")
    print(f"""Accuracy (Model): {acc_model:.3f}%""")

    print(f"""ROC AUC (Naive): {roc_naive:.3f}""")
    print(f"""ROC AUC (Baseline): {roc_base:.3f}""")
    print(f"""ROC AUC (Model): {roc_model:.3f}""")

    print(f"""Classification Report (Naive): {rep_naive }""")
    print(f"""Classification Report (Baseline): {rep_baseline}""")
    print(f"""Classification Report (Model): {rep_model}""")

    # Generate ROC Curve Comparison
    fpr_naive, tpr_naive, _ = roc_curve(dfscore[LABEL], dfscore["pred_naive"])
    fpr_baseline, tpr_baseline, _ = roc_curve(dfscore[LABEL], dfscore[PRED_BASELINE])
    fpr, tpr, _ = roc_curve(dfscore[LABEL], dfscore[PRED_MODEL])

    plt.plot(fpr_naive, tpr_naive, label=f"Naive {roc_naive:.3f}", alpha=0.7, color="black", linestyle="--")
    plt.plot(fpr_baseline, tpr_baseline, label=f"Baseline {roc_base:.3f}", color="Blue", alpha=0.7)
    plt.plot(fpr, tpr, label=f"Model {roc_model:.3f}", color="Green", alpha=0.95)

    plt.legend(loc="upper left")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.title("ROC Curve Comparison")
    plt.grid()
    fname = PLOT_DIR.joinpath("roc_curve_comparison.png")
    plt.savefig(fname)
    plt.show()
