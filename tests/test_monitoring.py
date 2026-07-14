import json

import pytest

from src.serving.monitoring import initialize_monitoring_db, load_prediction_logs, log_prediction


@pytest.mark.unit
def test_initialize_monitoring_db_creates_sqlite_file(tmp_path):
    db_path = tmp_path / "monitoring" / "predictions.db"

    result_path = initialize_monitoring_db(db_path)

    assert result_path == db_path
    assert db_path.exists()


@pytest.mark.unit
def test_log_prediction_stores_request_and_prediction(tmp_path):
    db_path = tmp_path / "predictions.db"
    request_payload = {
        "customerID": "customer-1",
        "tenure": 12,
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
    }
    prediction_payload = {
        "churn_probability": 0.73,
        "churn_prediction": 1,
        "model_name": "Test model",
        "model_version": "test-version",
        "threshold": 0.5,
    }

    log_prediction(request_payload, prediction_payload, db_path=db_path)
    logs = load_prediction_logs(db_path)

    assert len(logs) == 1
    row = logs.iloc[0]
    assert row["customer_id"] == "customer-1"
    assert row["tenure"] == 12
    assert row["monthly_charges"] == 80.0
    assert row["contract"] == "Month-to-month"
    assert row["churn_probability"] == 0.73
    assert row["churn_prediction"] == 1
    assert json.loads(row["request_json"])["customerID"] == "customer-1"
