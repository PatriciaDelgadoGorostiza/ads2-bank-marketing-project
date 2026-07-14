from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(path: Path):
    """Load a serialized model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict(model, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Create class predictions and positive-class probabilities for a fitted classifier."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Churn prediction requires a model with predict_proba.")

    churn_probability = np.asarray(model.predict_proba(X))[:, 1]
    churn_prediction = (churn_probability >= threshold).astype(int)

    return pd.DataFrame(
        {
            "churn_probability": churn_probability,
            "churn_prediction": churn_prediction,
        },
        index=X.index,
    )
