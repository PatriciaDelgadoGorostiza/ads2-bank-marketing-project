from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

BASE_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_baseline_models(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Create the baseline model set from Notebook 04.

    These models are intentionally simple and are used to compare increasing
    model complexity during initial model development.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_default_xgboost(random_state: int = RANDOM_STATE) -> XGBClassifier:
    """Create the default XGBoost classifier used before hyperparameter tuning.

    This is a useful fallback model, but the production-style training pipeline
    should usually receive explicit tuned parameters after model selection.
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost_with_params(params: dict, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """Create an XGBoost classifier from tuned parameter values.

    The input `params` should contain only the hyperparameters selected during
    model development or hyperparameter optimization. Stable base parameters
    such as objective, eval metric, random state, and n_jobs are added here.
    """
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    return XGBClassifier(**base_params, **params)


def fit_models(models: dict[str, object], X_train, y_train) -> dict[str, object]:
    """Fit a dictionary of models and return the fitted models.

    This helper keeps the baseline training loop from Notebook 04 reusable
    without hiding the fact that each model is still fitted independently.
    """
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
    return fitted_models


def safe_model_name(name: str) -> str:
    """Convert a model name into a filesystem-friendly filename fragment."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def save_model(
    model,
    output_dir: Path,
    model_class_name: str | None = None,
    suffix: str = "model",
    timestamp: str | None = None,
) -> Path:
    """Save a fitted model as a timestamped joblib artifact.

    The optional `timestamp` argument allows related artifacts, for example a
    model and its fitted preprocessor, to share the same filename prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_class_name is None:
        model_class_name = model.__class__.__name__

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = output_dir / f"{timestamp}_{safe_model_name(model_class_name)}_{suffix}.joblib"
    joblib.dump(model, model_path)
    return model_path


def load_model(path: Path):
    """Load a joblib model artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
