from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.str.strip()

    required_columns = {"customerID", "TotalCharges", "tenure", "Churn"}
    missing_columns = required_columns - set(cleaned.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if cleaned["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found. Investigate before continuing.")

    churn_mapping = {"No": 0, "Yes": 1}
    unexpected_churn_values = set(cleaned["Churn"].dropna().unique()) - set(churn_mapping)
    if unexpected_churn_values:
        raise ValueError(f"Unexpected Churn values: {sorted(unexpected_churn_values)}")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    missing_total_charges = cleaned["TotalCharges"].isna()
    missing_with_nonzero_tenure = missing_total_charges & (cleaned["tenure"] != 0)
    if missing_with_nonzero_tenure.any():
        problem_rows = cleaned.loc[
            missing_with_nonzero_tenure,
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges"],
        ]
        raise ValueError(
            "Found missing TotalCharges for customers with tenure > 0. "
            f"Problem rows: {problem_rows.to_dict(orient='records')}"
        )

    cleaned.loc[missing_total_charges, "TotalCharges"] = 0.0
    cleaned["ChurnBinary"] = cleaned["Churn"].map(churn_mapping).astype("int64")

    return cleaned


def save_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned data and create the target directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
