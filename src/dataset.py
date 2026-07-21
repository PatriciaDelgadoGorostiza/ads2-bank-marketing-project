from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw Bank Marketing dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path, sep=";")


def clean_bank_marketing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Bank Marketing dataset."""

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    required_columns = {"y", "duration"}
    missing_columns = required_columns - set(cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if cleaned.duplicated().any():
        raise ValueError(
            "Duplicate rows found. Investigate before continuing."
        )

    target_mapping = {
        "no": 0,
        "yes": 1,
    }

    unexpected_target_values = (
        set(cleaned["y"].dropna().unique())
        - set(target_mapping)
    )

    if unexpected_target_values:
        raise ValueError(
            f"Unexpected target values: {sorted(unexpected_target_values)}"
        )

    cleaned["y_binary"] = (
        cleaned["y"]
        .map(target_mapping)
        .astype("int64")
    )

    cleaned = cleaned.drop(columns=["duration"])

    return cleaned


def save_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned data and create the target directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
