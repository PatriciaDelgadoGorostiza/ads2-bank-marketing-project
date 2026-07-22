import pandas as pd
import pytest

from src.dataset import clean_bank_marketing_data


@pytest.mark.unit
def test_clean_bank_marketing_data_encodes_target_and_removes_duration():
    raw_df = pd.DataFrame(
        {
            "age": [35, 48],
            "duration": [120, 240],
            "y": ["no", "yes"],
        }
    )

    cleaned_df = clean_bank_marketing_data(raw_df)

    assert cleaned_df["y_binary"].tolist() == [0, 1]
    assert "duration" not in cleaned_df.columns


@pytest.mark.unit
def test_clean_bank_marketing_data_rejects_unexpected_target_values():
    raw_df = pd.DataFrame(
        {
            "age": [35],
            "duration": [120],
            "y": ["maybe"],
        }
    )

    with pytest.raises(ValueError, match="Unexpected target values"):
        clean_bank_marketing_data(raw_df)