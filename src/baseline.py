import os
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.process import load_train_test, process_baseline_data

gb_params = {"n_estimators": [100, 200, 300], "learning_rate": [0.1, 0.2, 0.3], "max_depth": [3, 5, 7]}

gb_class = GradientBoostingClassifier()
gb_grid = GridSearchCV(gb_class, gb_params, cv=5, n_jobs=-1)


if __name__ == "__main__":
    START = time.time()
    LABEL = "target"
    from pprint import pformat
    os.makedirs("Logs", exist_ok=True)
    logger.add(f"Logs/baseline-model.log", format="{time:YYYY-MM-DD HH:mm:ss} LogLevel {level} {message}", level="INFO")

    logger.info(f"Begin processing data for baseline model")

    train, test = load_train_test()
    X_train = process_baseline_data(train, LABEL)
    X_test = process_baseline_data(test, LABEL)
    submission = pd.read_csv("data/sample_submission.csv")


    # Fit model with sample to reduce time
    N = 10_000
    N = None
    FNAME = "data/baseline_predictions.csv"
    FNAME_IN = "data/baseline_predictions_cv.csv"

    X_train = X_train.head(N)
    y_train = train[LABEL].head(N)

    logger.info(f"Train model with {X_train.shape[0]:,d} Samples")

    gb_grid.fit(X_train, y_train)
    best_params = gb_grid.best_estimator_.get_params()
    logger.info(f"Best Estimator Params:\n {pformat(best_params)}")

    gb_opt = GradientBoostingClassifier(
        criterion="friedman_mse",
        init=None,
        learning_rate=gb_grid.best_estimator_.learning_rate,
        loss="log_loss",
        max_depth=gb_grid.best_estimator_.max_depth,
        max_features=None,
        max_leaf_nodes=None,
        # min_impurity_split=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        # n_estimators=100,
        n_estimators=gb_grid.best_estimator_.n_estimators,
        # presort='auto',
        random_state=None,
        subsample=1.0,
        verbose=1,
        warm_start=False,
    )

    gb_opt.fit(X_train, y_train)
    test_y_gb = gb_opt.predict_proba(X_test)

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    # dfout = test[["id", LABEL]].copy()
    # dfout["pred"] = test_y_gb[:, 1]
    # dfout["pred_label"] = np.where(dfout["pred"] > 0.5, 1, 0)
    # dfout.to_csv(FNAME, index=False, float_format="%.4f")

    dfout = submission.copy()
    dfout["pred"] = test_y_gb[:, 1]
    dfout["pred_label"] = np.where(dfout["pred"] > 0.5, 1, 0)
    dfout.to_csv(FNAME, index=False, float_format="%.4f")
    
    dfin = train[["id", LABEL]].copy()
    dfin["pred"] = gb_opt.predict_proba(pd.read_parquet("data/X_train.snap.parquet"))[:,1]
    dfin["pred_label"] = np.where(dfin["pred"] > 0.5, 1, 0)
    dfin.to_csv(FNAME_IN, index=False, float_format="%.4f")

    logger.info(f"""Classification Report:\n{classification_report(dfin[LABEL], dfin["pred_label"])})""")
    logger.info(f"""Accuracy: {accuracy_score(dfin[LABEL], dfin["pred_label"]) * 100:.2f}%""")
    logger.info(f"""ROC: {roc_auc_score(dfin[LABEL], dfin["pred_label"]):,.4}""")
    logger.info(f"Finished Baseline model in {(time.time() - START) / 60 :,.5} minutes.")
