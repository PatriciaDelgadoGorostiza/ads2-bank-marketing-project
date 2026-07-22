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
    title="Bank Marketing Subscription Prediction API",
    description=(
        "Local FastAPI service for one-customer term-deposit subscription predictions. "
        "The OpenAPI schema documents the request and response contract."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_serving_manifest() -> dict:
    """Load and cache the model manifest once per API process."""
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
    """Predict term-deposit subscription probability for one customer."""
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
        subscription_probability=float(prediction["subscription_probability"]),
        subscription_prediction=int(prediction["subscription_prediction"]),
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