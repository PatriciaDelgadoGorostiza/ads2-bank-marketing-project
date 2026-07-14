from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from src.dataset import clean_telco_data, load_raw_data
from src.evaluation import DEFAULT_BUSINESS_PARAMS, evaluate_classifier
from src.features import (
    create_business_features,
    create_train_test_split,
    load_preprocessor,
    prepare_inference_features,
    preprocess_train_test,
    save_preprocessor,
    save_processed_outputs,
    split_features_and_target,
    transform_with_preprocessor,
)
from src.modeling.predict import predict
from src.modeling.train import (
    build_default_xgboost,
    build_xgboost_with_params,
    load_model,
    save_model,
)


@dataclass
class TrainingPipelineResult:
    """Return object for the training pipeline.

    The local model and preprocessor paths are needed for the inference pipeline.
    The optional MLflow run ID links the same artifacts and metrics to experiment
    tracking.
    """

    model_path: Path
    preprocessor_path: Path
    manifest_path: Path
    metrics: dict
    mlflow_run_id: str | None = None


def _log_training_run_to_mlflow(
    model,
    model_name: str,
    model_params: dict | None,
    model_path: Path,
    preprocessor_path: Path,
    metrics: dict,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
) -> str:
    """Log training parameters, metrics, and artifacts to MLflow.

    MLflow stores the run metadata and copies of the generated artifacts. The
    local files in `models/` are still returned by the training pipeline because
    the following inference step can use them directly.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    metric_values = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, Real) and key not in {"threshold"}
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "telco_churn",
                "stage": "training_pipeline",
                "model_name": model_name,
                "model_class": model.__class__.__name__,
            }
        )
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model.__class__.__name__)
        if model_params:
            mlflow.log_params(model_params)
        mlflow.log_params(DEFAULT_BUSINESS_PARAMS)
        mlflow.log_metrics(metric_values)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        return run.info.run_id


def run_training_pipeline(
    raw_data_path: Path = RAW_DATA_DIR / "Telco-Customer-Churn.csv",
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    model=None,
    model_params: dict | None = None,
    model_name: str = "XGBoost",
    save_processed_data: bool = True,
    log_to_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "telco-churn",
    manifest_path: Path | None = None,
) -> TrainingPipelineResult:
    """Run the Telco Churn training pipeline from raw data to saved artifacts.

    The pipeline does not perform hyperparameter optimization. It trains one
    defined model configuration and creates the artifacts needed for inference.

    Parameters
    ----------
    model:
        Optional preconfigured model object. Use this when a caller already
        built the estimator elsewhere.
    model_params:
        Optional XGBoost hyperparameters, for example the selected parameters
        from hyperparameter optimization. If `model` is not provided and
        `model_params` is given, the pipeline builds an XGBoost model from these
        parameters. If neither is provided, it falls back to default XGBoost.
    log_to_mlflow:
        If True, metrics, selected parameters, and generated artifacts are logged
        to the configured MLflow experiment.
    """
    if model is not None and model_params is not None:
        raise ValueError("Pass either a model object or model_params, not both.")

    raw_df = load_raw_data(raw_data_path)
    cleaned_df = clean_telco_data(raw_df)
    featured_df = create_business_features(cleaned_df)

    X, y = split_features_and_target(featured_df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    X_train_processed, X_test_processed, feature_names, preprocessor = preprocess_train_test(
        X_train,
        X_test,
    )

    if save_processed_data:
        save_processed_outputs(
            featured_df=featured_df,
            X_train_processed=X_train_processed,
            X_test_processed=X_test_processed,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=processed_data_dir,
        )

    if model is None:
        if model_params is None:
            model = build_default_xgboost()
            model_params_to_log = None
        else:
            model = build_xgboost_with_params(model_params)
            model_params_to_log = model_params
    else:
        model_params_to_log = model_params

    model.fit(X_train_processed, y_train)
    metrics = evaluate_classifier(
        model=model,
        model_name=model_name,
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        threshold=0.5,
        business_params=DEFAULT_BUSINESS_PARAMS,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocessor_path = save_preprocessor(preprocessor, models_dir, timestamp=timestamp)
    model_path = save_model(
        model=model,
        output_dir=models_dir,
        model_class_name=model.__class__.__name__,
        suffix="model",
        timestamp=timestamp,
    )

    mlflow_run_id = None
    if log_to_mlflow:
        if mlflow_tracking_uri is None:
            mlflow_tracking_uri = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
        mlflow_run_id = _log_training_run_to_mlflow(
            model=model,
            model_name=model_name,
            model_params=model_params_to_log,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metrics=metrics,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
            run_name=f"{timestamp}_{model.__class__.__name__}_training_pipeline",
        )

    if manifest_path is None:
        manifest_path = models_dir / "model_manifest.json"
    manifest = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "created_at": timestamp,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "threshold": 0.5,
        "mlflow_run_id": mlflow_run_id,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return TrainingPipelineResult(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        manifest_path=manifest_path,
        metrics=metrics,
        mlflow_run_id=mlflow_run_id,
    )


def load_model_manifest(manifest_path: Path = MODELS_DIR / "model_manifest.json") -> dict:
    """Load the local serving manifest that points to model and preprocessor artifacts."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model manifest not found: {manifest_path}. "
            "Run the training pipeline first to create serving artifacts."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    model_path = Path(manifest["model_path"])
    preprocessor_path = Path(manifest["preprocessor_path"])

    if not model_path.exists():
        model_path = manifest_path.parent / model_path.name
    if not preprocessor_path.exists():
        preprocessor_path = manifest_path.parent / preprocessor_path.name

    manifest["model_path"] = model_path
    manifest["preprocessor_path"] = preprocessor_path
    return manifest


def run_inference_pipeline(
    input_data: pd.DataFrame | Path,
    model_path: Path,
    preprocessor_path: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run inference for new Telco observations using saved training artifacts.

    The function expects raw Telco-style input rows. It recreates deterministic
    business features, applies the fitted preprocessor from training, loads the
    fitted model, and returns probabilities plus class predictions.
    """
    if isinstance(input_data, Path):
        input_df = pd.read_csv(input_data)
    else:
        input_df = input_data.copy()

    preprocessor = load_preprocessor(preprocessor_path)
    model = load_model(model_path)

    X = prepare_inference_features(input_df)
    X_processed = transform_with_preprocessor(preprocessor, X)
    predictions = predict(model, X_processed, threshold=threshold)

    if "customerID" in input_df.columns:
        predictions.insert(0, "customerID", input_df["customerID"].values)

    return predictions.reset_index(drop=True)


def load_model_params(params_path: Path) -> dict:
    """Load selected model parameters from the JSON artifact created in Notebook 05."""
    if not params_path.exists():
        raise FileNotFoundError(f"Model parameter file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    return model_config["params"]
