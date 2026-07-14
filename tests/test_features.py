import pytest

from src.features import create_business_features, prepare_inference_features


@pytest.mark.unit
def test_create_business_features_matches_notebook_logic(minimal_telco_input):
    featured_df = create_business_features(minimal_telco_input)

    assert featured_df.loc[0, "TenureGroup"] == "7-12 months"
    assert featured_df.loc[0, "AverageMonthlyCharges"] == 70.0
    assert featured_df.loc[0, "MonthlyChargesDelta"] == 10.0
    assert featured_df.loc[0, "NumberOfAddOnServices"] == 3


@pytest.mark.unit
def test_prepare_inference_features_returns_model_input_columns_without_target(
    minimal_telco_input,
):
    X = prepare_inference_features(minimal_telco_input)

    assert "Churn" not in X.columns
    assert "ChurnBinary" not in X.columns
    assert "TenureGroup" in X.columns
    assert "NumberOfAddOnServices" in X.columns
