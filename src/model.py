from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.process import process_data

FNAME_OUT = Path("data").joinpath("prediction.snap.parquet")
PLOT_DIR = Path("data").joinpath("plots")
LABEL = "target"
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/train.csv')

X_train, y_train = process_data(train, LABEL)
X_test, y_test = process_data(test, LABEL)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test)


params = {
    "boosting_type": "gbdt",
    "metric": "logloss",
    "max_depth": 3,
    "objective": "regression_l1"
    }

model = lgb.train(params, lgb_train, verbose_eval=10, valid_sets=[lgb_train, lgb_test])

# Save Predictions
dfout = test[["id", LABEL]].copy()
dfout["pred"] = model.predict(X_test)
dfout["pred_label"] = np.where(dfout["pred"] >= .5, 1, 0)
dfout.to_parquet(FNAME_OUT)

# Shap Scores
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
fig = plt.gcf()
shap.summary_plot(shap_values, X_train, plot_size=(24, 24), max_display=100)
fig.savefig(PLOT_DIR.joinpath("shap_scores.png"))
