from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger


def get_zero_variance_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of columns with zero variance"""
    return df.columns[df.var() == 0].tolist()


def process_data(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Process input and return DataFrame for ML Modeling.

    Args:
        df (pd.DataFrame): input DataFrame (train | test)
        label (str): Target variable

    Returns:
        pd.DataFrame: DataFrame for modelling
    """
    desc = "test"
    rows, cols = df.shape
    logger.info(f"Process input dataframe with {rows:,d} rows and {cols:,d} columns")
    columns = df.columns.tolist()
    calc_features = [c for c in columns if "calc" in c]

    X = df.drop(calc_features, axis=1)
    
    if label in columns:
        desc = "train"
        logger.info(f"Drop {label} from training data")
        X = X.drop(label, axis=1)
    
    rows_out, cols_out = X.shape
    logger.info(f"Return {desc} data with {rows_out:,d} rows and {cols_out:,d} columns")
    return X


def load_train_test(data_dir: Path = Path("data")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train = pd.read_csv(data_dir.joinpath("train.csv"))
        test = pd.read_csv(data_dir.joinpath("test.csv"))
    except FileNotFoundError as e:
        msg = f"""Data not found!
                  Please download data from Kaggle and place in src/data folder.
                  See Setup section in README.md for more details.
                """
        raise ValueError(msg)
    return train, test


if __name__ == '__main__':

    logger.info(f"Begin processing data for modeling")
    DIR = Path("data")
    LABEL = "target"
    train, test = load_train_test()

    PATH_XTRAIN = DIR.joinpath("X_train.snap.parquet")
    PATH_YTRAIN = DIR.joinpath("y_train.snap.parquet")
    PATH_XTEST = DIR.joinpath("X_test.snap.parquet")

    X_train = process_data(train, LABEL)
    y_train = train[LABEL]
    X_test = process_data(test, LABEL)

    assert X_train.shape[1] == X_test.shape[1], f"Columns mismatch in train/test data."

    X_train.to_parquet(PATH_XTRAIN)
    X_test.to_parquet(PATH_XTEST)
    y_train.to_frame().to_parquet(PATH_YTRAIN)

    logger.info(f"Finished saving processed data for modeling")
