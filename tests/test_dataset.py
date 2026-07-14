import pandas as pd
import pytest

from src.dataset import clean_telco_data


@pytest.mark.unit
def test_clean_telco_data_converts_total_charges_and_target():
    raw_df = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "TotalCharges": ["29.85", " "],
            "tenure": [1, 0],
            "MonthlyCharges": [29.85, 19.85],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned_df = clean_telco_data(raw_df)

    assert cleaned_df.loc[0, "TotalCharges"] == 29.85
    assert cleaned_df.loc[1, "TotalCharges"] == 0.0
    assert cleaned_df["ChurnBinary"].tolist() == [0, 1]


@pytest.mark.unit
def test_clean_telco_data_rejects_missing_total_charges_for_existing_customer():
    raw_df = pd.DataFrame(
        {
            "customerID": ["A"],
            "TotalCharges": [" "],
            "tenure": [3],
            "MonthlyCharges": [29.85],
            "Churn": ["No"],
        }
    )

    with pytest.raises(ValueError, match="missing TotalCharges"):
        clean_telco_data(raw_df)
