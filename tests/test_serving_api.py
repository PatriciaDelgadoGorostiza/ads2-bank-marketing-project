import pytest
import pandas as pd
from fastapi.testclient import TestClient

import src.serving.api as api_module
from src.serving.api import app, get_serving_manifest


@pytest.fixture
def api_client():
    get_serving_manifest.cache_clear()
    return TestClient(app)


@pytest.fixture
def api_customer_payload():
    return {
        "customerID": "api-test-customer",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
    }


@pytest.mark.integration
def test_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_predict_endpoint_returns_one_customer_prediction(
    monkeypatch,
    api_client,
    api_customer_payload,
):
    test_manifest = {
        "model_name": "Test model",
        "model_class": "TestClassifier",
        "created_at": "test-version",
        "threshold": 0.5,
        "model_path": "models/test_model.joblib",
        "preprocessor_path": "models/test_preprocessor.joblib",
    }
    test_predictions = pd.DataFrame(
        [
            {
                "customerID": "api-test-customer",
                "churn_probability": 0.73,
                "churn_prediction": 1,
            }
        ]
    )
    monkeypatch.setattr(api_module, "get_serving_manifest", lambda: test_manifest)
    monkeypatch.setattr(api_module, "run_inference_pipeline", lambda **kwargs: test_predictions)
    monkeypatch.setattr(api_module, "log_prediction", lambda **kwargs: None)

    response = api_client.post("/predict", json=api_customer_payload)

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["customerID"] == "api-test-customer"
    assert prediction["churn_probability"] == 0.73
    assert prediction["churn_prediction"] == 1
    assert prediction["threshold"] == 0.5
    assert prediction["model_name"] == "Test model"
    assert prediction["model_version"] == "test-version"


@pytest.mark.integration
def test_predict_endpoint_still_returns_prediction_when_monitoring_fails(
    monkeypatch,
    api_client,
    api_customer_payload,
):
    test_manifest = {
        "model_name": "Test model",
        "model_class": "TestClassifier",
        "created_at": "test-version",
        "threshold": 0.5,
        "model_path": "models/test_model.joblib",
        "preprocessor_path": "models/test_preprocessor.joblib",
    }
    test_predictions = pd.DataFrame(
        [
            {
                "customerID": "api-test-customer",
                "churn_probability": 0.73,
                "churn_prediction": 1,
            }
        ]
    )

    def fail_monitoring_log(**kwargs):
        raise OSError("Monitoring database is not writable")

    monkeypatch.setattr(api_module, "get_serving_manifest", lambda: test_manifest)
    monkeypatch.setattr(api_module, "run_inference_pipeline", lambda **kwargs: test_predictions)
    monkeypatch.setattr(api_module, "log_prediction", fail_monitoring_log)

    response = api_client.post("/predict", json=api_customer_payload)

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.73


@pytest.mark.integration
def test_openapi_schema_exposes_predict_endpoint(api_client):
    response = api_client.get("/openapi.json")

    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


@pytest.mark.integration
def test_model_info_returns_503_for_unimplemented_remote_manifest(monkeypatch):
    monkeypatch.setenv("MODEL_MANIFEST_URI", "azure://example/model_manifest.json")
    get_serving_manifest.cache_clear()
    client = TestClient(app)

    response = client.get("/model-info")

    assert response.status_code == 503
    assert "Remote model artifact loading is not implemented yet" in response.json()["detail"]
    get_serving_manifest.cache_clear()
