from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.pipeline import run_inference_pipeline
from src.serving.artifacts import load_serving_manifest
from src.serving.monitoring import log_prediction
from src.serving.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Local FastAPI service for one-customer real-time churn predictions. "
        "The OpenAPI schema documents the request and response contract."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_serving_manifest() -> dict:
    """Load and cache the model manifest once per API process.

    The actual artifact source is delegated to `src.serving.artifacts`:

    - local Docker Compose demo: read `models/model_manifest.json`;
    - future cloud deployment: read `MODEL_MANIFEST_URI` and download artifacts.
    """
    return load_serving_manifest()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple liveness response."""
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return metadata about the model currently used for serving."""
    try:
        manifest = get_serving_manifest()
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        model_name=manifest["model_name"],
        model_class=manifest["model_class"],
        model_version=manifest["created_at"],
        threshold=manifest["threshold"],
        model_path=str(manifest["model_path"]),
        preprocessor_path=str(manifest["preprocessor_path"]),
        mlflow_run_id=manifest.get("mlflow_run_id"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn probability for one customer."""
    try:
        manifest = get_serving_manifest()
        input_df = pd.DataFrame([customer.model_dump()])
        predictions = run_inference_pipeline(
            input_data=input_df,
            model_path=manifest["model_path"],
            preprocessor_path=manifest["preprocessor_path"],
            threshold=manifest["threshold"],
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = predictions.iloc[0]
    response = PredictionResponse(
        customerID=prediction.get("customerID"),
        churn_probability=float(prediction["churn_probability"]),
        churn_prediction=int(prediction["churn_prediction"]),
        threshold=float(manifest["threshold"]),
        model_name=manifest["model_name"],
        model_version=manifest["created_at"],
    )
    try:
        log_prediction(
            request_payload=customer.model_dump(),
            prediction_payload=response.model_dump(),
        )
    except Exception as exc:
        logger.warning(f"Prediction monitoring log failed: {exc}")
    return response
