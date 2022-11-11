import numpy as np
import pandas as pd
from typing import Union


def Gini(y_true: Union[np.array, pd.Series], y_pred: Union[np.array, pd.Series]) -> float:
    """Calculate Gini Coefficient between actuals and predictions.

    Args:
        y_true (Union[np.array, pd.Series]): actuals
        y_pred (Union[np.array, pd.Series]): predictions

    Returns:
        float: Gini Score
    """
    assert y_true.shape == y_pred.shape, f"Predictions {y_pred.shape} don't match Actuals {y_true.shape}"
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true
