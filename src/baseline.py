import pandas as pd
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
    LABEL = "target"
    X_train = pd.read_parquet("data/X_train.snap.parquet").head(1000)
    X_test = pd.read_parquet("data/X_test.snap.parquet").head(1000)
    y_train = pd.read_parquet("data/y_train.snap.parquet")[LABEL].head(1000)
    y_test = pd.read_parquet("data/y_test.snap.parquet")[LABEL].head(1000)
    submission = pd.read_csv('data/sample_submission.csv')

    gb_grid.fit(X_train, y_train)
    gb_grid.best_estimator_
    gb_opt = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                        learning_rate=0.1, loss='deviance', max_depth=3,
                                        max_features=None, max_leaf_nodes=None, min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0, n_estimators=100,
                                        presort='auto', random_state=None, subsample=1.0, verbose=0,
                                        warm_start=False)
        
    gb_opt.fit(X_train, y_train)
    test_y_gb = gb_opt.predict_proba(X_test)
    gb_out = submission
    gb_out['target'] = test_y_gb

    gb_out['target'] = 1-gb_out['target']
    gb_out.to_csv('data/gb_predictions1.csv', index=False, float_format='%.4f')

    