import os
from sklearn.model_selection import train_test_split
import pandas as pd

from src.consts import CLEANED_GAMELOG_PATH, SINGLE_GAME_COLUMNS, STRING_FIELDS

SINGLE_GAME_COLUMNS


def split_data(
    df: pd.DataFrame,
    target="win",
    fields_to_drop: list[str] = [],
    test_size=0.2,
    random_state=42,
    keep_gamestats=False,
):
    drop_fields = fields_to_drop
    if not keep_gamestats:
        drop_fields += SINGLE_GAME_COLUMNS

    X = df.drop([target] + drop_fields, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def encode_data(df: pd.DataFrame, binned_fields: list[str] = []):
    included_string_fields = set(STRING_FIELDS) & set(df.columns)
    return pd.get_dummies(df, columns=list(included_string_fields) + binned_fields)


def load_cleaned_gamelog():
    return pd.read_csv(f"{os.getcwd()}/{CLEANED_GAMELOG_PATH}")
