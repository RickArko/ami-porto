from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


def get_zero_variance_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of columns with zero variance"""
    return df.columns[df.var() == 0].tolist()


def process_data(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process input DataFrame return X and Y DataFrames for Modeling.

    Args:
        df (pd.DataFrame): DataFrame (test | train)
        label (str): Target variable

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X, y DataFrames.
    """
    rows, cols = df.shape
    logger.info(f"Process input dataframe with {rows:,d} rows and {cols:,d} columns")
    columns = df.columns.tolist()
    calc_features = [c for c in columns if "calc" in c]

    X = df.drop(calc_features, axis=1)
    X = X.drop(label, axis=1)
    y = df[label]
    
    rows_out, cols_out = X.shape
    logger.info(f"Return training data with {rows_out:,d} rows and {cols_out:,d} columns")
    return X, y


if __name__ == '__main__':

    logger.info(f"Begin processing data for modeling")

    try:
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/train.csv')
    except FileNotFoundError as e:
        msg = f"""Data not found!
                  Please download data from Kaggle and place in src/data folder.
                  See Setup section in README.md for more details.
                """
        raise ValueError(msg)

    DIR = Path("data")
    LABEL = "target"

    PATH_XTRAIN = DIR.joinpath("X_train.snap.parquet")
    PATH_YTRAIN = DIR.joinpath("X_train.snap.parquet")
    PATH_XTEST = DIR.joinpath("X_train.snap.parquet")
    PATH_YTEST = DIR.joinpath("X_train.snap.parquet")

    X_train, y_train = process_data(train, LABEL)
    X_test, y_test = process_data(test, LABEL)

    X_train.to_parquet(PATH_XTRAIN)
    X_test.to_parquet(PATH_XTEST)
    y_train.to_frame().to_parquet(PATH_YTRAIN)
    y_test.to_frame().to_parquet(PATH_YTEST)

    logger.info(f"Finished saving processed data for modeling")
