from pydantic import BaseModel, ConfigDict


class CustomerFeatures(BaseModel):
    """Input contract for one Bank Marketing subscription prediction request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


class PredictionResponse(BaseModel):
    """Output contract for one term-deposit subscription prediction."""

    subscription_probability: float
    subscription_prediction: int
    threshold: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Response returned by the API health endpoint."""

    status: str


class ModelInfoResponse(BaseModel):
    """Metadata describing the model currently used by the API."""

    model_name: str
    model_class: str
    model_version: str
    threshold: float
    model_path: str
    preprocessor_path: str
    mlflow_run_id: str | None = None