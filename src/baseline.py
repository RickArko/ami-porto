import time

import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


gb_params = {
    'n_estimators' : [100,200,300],
    'learning_rate' : [.1,.2,.3],
    'max_depth' : [3,5,7]
}

gb_class = GradientBoostingClassifier()
gb_grid = GridSearchCV(gb_class, gb_params, cv = 5, n_jobs=-1)


if __name__ == '__main__':
    START = time.time()
    LABEL = "target"
    import os
    os.makedirs("Logs", exist_ok=True)
    logger.add(f"Logs/baseline-processor.log", 
               format="{time:YYYY-MM-DD HH:mm:ss} LogLevel {level} {message}", level="INFO")
    logger.info(f"Begin processing data for baseline model")

    X_train = pd.read_parquet("data/X_train.snap.parquet")
    X_test = pd.read_parquet("data/X_test.snap.parquet")
    y_train = pd.read_parquet("data/y_train.snap.parquet")[LABEL]
    y_test = pd.read_parquet("data/y_test.snap.parquet")[LABEL]
    submission = pd.read_csv('data/sample_submission.csv')
    
    logger.info(f"Read X_train: {X_train.shape} and X_test: {X_test.shape}")

    gb_grid.fit(X_train, y_train)
    logger.info(f"Best Estimator: {gb_grid.best_estimator_}")

    gb_opt = GradientBoostingClassifier(criterion='friedman_mse',
                                        init=None,
                                        learning_rate=0.1, loss='log_loss', max_depth=3,
                                        max_features=None, max_leaf_nodes=None,
                                        # min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0, n_estimators=100,
                                        # presort='auto',
                                        random_state=None, subsample=1.0, verbose=0,
                                        warm_start=False)
        
    gb_opt.fit(X_train, y_train)
    test_y_gb = gb_opt.predict_proba(X_test)
    gb_out = submission
    gb_out['target'] = test_y_gb

    gb_out['target'] = 1-gb_out['target']
    gb_out.to_csv('data/gb_predictions1.csv', index=False, float_format='%.4f')
    logger.info(f"Finished Baseline model in {time.time() - START:,.1} seconds.")
    