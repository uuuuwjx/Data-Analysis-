import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_SUMMARY_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df[TARGET_COLUMN].value_counts())
    return df



def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = df.copy()
    summary_df["TotalCharges"] = pd.to_numeric(summary_df["TotalCharges"], errors="coerce")
    return summary_df[NUMERIC_SUMMARY_COLUMNS].describe()



def preprocess(df: pd.DataFrame):
    processed = df.copy()
    processed["TotalCharges"] = pd.to_numeric(processed["TotalCharges"], errors="coerce")
    processed["TotalCharges"] = processed["TotalCharges"].fillna(processed["TotalCharges"].median())
    processed[TARGET_COLUMN] = processed[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    processed = pd.get_dummies(processed, columns=CATEGORICAL_COLUMNS, drop_first=False)
    processed = processed.drop(columns=[ID_COLUMN])

    X = processed.drop(columns=[TARGET_COLUMN])
    y = processed[TARGET_COLUMN]
    return X, y



def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
