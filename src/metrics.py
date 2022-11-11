from typing import Union

import numpy as np
import pandas as pd
from numba import jit

"""This competition uses Normalized Gini Coefficient for Evaluation.

The leaderboard ranges from:
    1.   - .29698
    .
    .
    1000 - .28986
    .
    .
    3000 - .28001
    
A "good" score should be somewhere around .28-.29

reading:

- [implementation](https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703)
- [intuitive-explanation](https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation)

"""


def Gini(y_true: Union[np.array, pd.Series], y_pred: Union[np.array, pd.Series]) -> float:
    """Calculate Normalized Gini Coefficient between actuals and predictions.

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


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
