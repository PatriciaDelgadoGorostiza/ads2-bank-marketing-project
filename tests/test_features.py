import pytest

from src.features import (
    FEATURE_COLUMNS,
    create_business_features,
    prepare_inference_features,
)


@pytest.mark.unit
def test_create_business_features_matches_notebook_logic(
    minimal_bank_marketing_input,
):
    featured_df = create_business_features(
        minimal_bank_marketing_input
    )

    assert featured_df.loc[0, "PreviouslyContacted"] == 0
    assert featured_df.loc[0, "HasAnyLoan"] == 1
    assert featured_df.loc[0, "BalanceGroup"] == "Medium"


@pytest.mark.unit
def test_prepare_inference_features_returns_expected_model_input_columns(
    minimal_bank_marketing_input,
):
    X = prepare_inference_features(
        minimal_bank_marketing_input
    )

    assert list(X.columns) == FEATURE_COLUMNS
    assert "y" not in X.columns
    assert "y_binary" not in X.columns
    assert "duration" not in X.columns

    assert "PreviouslyContacted" in X.columns
    assert "HasAnyLoan" in X.columns
    assert "BalanceGroup" in X.columns