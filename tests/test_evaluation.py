import pandas as pd
import pytest

from src.evaluation import calculate_business_kpis


@pytest.mark.unit
def test_calculate_business_kpis_uses_probability_based_contact_decision():
    y_true = pd.Series([1, 0, 1])
    y_score = [0.80, 0.20, 0.60]
    business_params = {
        "retention_offer_cost": 50,
        "customer_value_if_retained": 300,
        "retention_success_rate": 0.30,
    }

    result = calculate_business_kpis(y_true, y_score, business_params)

    assert result["contacted_customers"] == 2
    assert result["true_churn_customers_contacted"] == 2
    assert result["non_churn_customers_contacted"] == 0
    assert result["missed_churn_customers"] == 0
    assert result["expected_net_value"] == 26.0
    assert result["realized_net_value_for_backtest"] == 80.0
