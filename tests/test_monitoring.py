import json

import pytest

from src.serving.monitoring import (
    initialize_monitoring_db,
    load_prediction_logs,
    log_prediction,
)


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
        "age": 40,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    prediction_payload = {
        "subscription_probability": 0.73,
        "subscription_prediction": 1,
        "model_name": "XGBoost Default",
        "model_version": "test-version",
        "threshold": 0.5,
    }

    log_prediction(
        request_payload,
        prediction_payload,
        db_path=db_path,
    )

    logs = load_prediction_logs(db_path)

    assert len(logs) == 1

    row = logs.iloc[0]

    assert row["subscription_probability"] == pytest.approx(0.73)
    assert row["subscription_prediction"] == 1
    assert row["model_name"] == "XGBoost Default"
    assert row["model_version"] == "test-version"
    assert row["threshold"] == pytest.approx(0.5)

    stored_request = json.loads(row["request_json"])

    assert stored_request["age"] == 40
    assert stored_request["job"] == "management"
    assert stored_request["balance"] == 1500
    assert stored_request["contact"] == "cellular"