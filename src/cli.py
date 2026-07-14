from pathlib import Path

import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.pipeline import load_model_params, run_inference_pipeline, run_training_pipeline

app = typer.Typer(help="Run the Telco Churn training and inference pipelines.")


@app.command("train")
def train_command(
    raw_data_path: Path = RAW_DATA_DIR / "Telco-Customer-Churn.csv",
    params_path: Path | None = None,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    model_name: str = "XGBoost tuned",
    save_processed_data: bool = False,
    log_to_mlflow: bool = True,
) -> None:
    """Train one model configuration and save model plus preprocessor artifacts."""
    model_params = load_model_params(params_path) if params_path is not None else None
    result = run_training_pipeline(
        raw_data_path=raw_data_path,
        processed_data_dir=processed_data_dir,
        models_dir=models_dir,
        model_params=model_params,
        model_name=model_name,
        save_processed_data=save_processed_data,
        log_to_mlflow=log_to_mlflow,
    )

    typer.echo(f"Model artifact: {result.model_path}")
    typer.echo(f"Preprocessor artifact: {result.preprocessor_path}")
    typer.echo(f"Model manifest: {result.manifest_path}")
    typer.echo(f"F1: {result.metrics['f1']:.4f}")
    typer.echo(f"ROC-AUC: {result.metrics['roc_auc']:.4f}")
    if result.mlflow_run_id is not None:
        typer.echo(f"MLflow run ID: {result.mlflow_run_id}")


@app.command("predict")
def predict_command(
    input_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Run batch inference from a CSV file and save predictions as CSV."""
    predictions = run_inference_pipeline(
        input_data=input_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold=threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    typer.echo(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    app()
