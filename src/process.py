from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


def get_zero_variance_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of columns with zero variance"""
    return df.columns[df.var() == 0].tolist()


if __name__ == '__main__':

    logger.info(f"Begin processing data for modeling")

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/train.csv')

    columns = train.columns.tolist()
    calc_features = [c for c in columns if "calc" in c]
    
    LABEL = "target"
    X_train = train.drop(calc_features, axis=1)
    X_train = X_train.drop(LABEL, axis=1)
    X_test = test.drop(calc_features, axis=1)
    X_test = X_test.drop(LABEL, axis=1)

    y_train = train[LABEL]
    y_test = test[LABEL]

    X_train.to_parquet("data/X_train.snap.parquet")
    X_test.to_parquet("data/X_test.snap.parquet")
    y_train.to_frame().to_parquet("data/y_train.snap.parquet")
    y_test.to_frame().to_parquet("data/y_test.snap.parquet")

    logger.info(f"Finished processing data for modeling")
