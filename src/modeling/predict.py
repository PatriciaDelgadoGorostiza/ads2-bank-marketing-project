from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(path: Path):
    """Load a serialized model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def predict(
    model,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create subscription predictions and positive-class probabilities."""

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Subscription prediction requires a model with predict_proba."
        )

    subscription_probability = np.asarray(
        model.predict_proba(X)
    )[:, 1]

    subscription_prediction = (
        subscription_probability >= threshold
    ).astype(int)

    return pd.DataFrame(
        {
            "subscription_probability": subscription_probability,
            "subscription_prediction": subscription_prediction,
        },
        index=X.index,
    )