# Bank Marketing Subscription MLOps Demo

This repository contains a notebook-first MLOps prototype for predicting whether
a bank customer is likely to subscribe to a term deposit after a telephone
marketing campaign.

The project starts with exploratory model development in notebooks and then
evolves into a small local ML application:

- data exploration, cleaning, and feature engineering;
- model training, evaluation, tuning, and error slicing;
- experiment tracking with MLflow;
- reusable Python modules;
- reproducible training and inference pipelines;
- automated tests;
- local FastAPI serving and a Streamlit frontend;
- Docker and Docker Compose;
- GitHub Actions CI;
- a basic monitoring and drift demo.

The main goal is not to build the best possible model, but to demonstrate a
locally executable and understandable MLOps workflow.

## Business Problem

Telephone marketing campaigns require time and resources.

This prototype estimates whether a customer is likely to subscribe to a term
deposit based on historical customer and campaign information. The prediction
could help prioritize customers for future campaigns.

The model predicts subscription likelihood. It does not prove that contacting a
customer causes a subscription.

## Repository Structure

```text
.
├── app/                    # Streamlit frontend
├── data/
│   ├── raw/                # Raw input data
│   ├── interim/            # Cleaned intermediate data
│   └── processed/          # Processed train/test data
├── models/                 # Generated model artifacts
├── monitoring/             # Local prediction logs
├── notebooks/              # Main project notebooks
├── references/             # Background material
├── reports/
│   ├── figures/            # Generated figures
│   └── model_results/      # Metrics, predictions, and tuning results
├── src/                    # Reusable Python code
│   ├── modeling/           # Training and prediction helpers
│   └── serving/            # FastAPI and monitoring code
├── tests/                  # Unit and integration tests
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-api.txt
├── requirements-streamlit.txt
└── README.md
```

## Data

The project uses the UCI Bank Marketing dataset.

Expected local raw-data path:

```text
data/raw/bank-full.csv
```

The dataset contains 45,211 records from telephone marketing campaigns conducted
by a Portuguese banking institution.

The target variable indicates whether a customer subscribed to a term deposit.
It is imbalanced: approximately 11.7% of customers subscribed and 88.3% did not.

The variable `duration` is removed before training because it is only known
after the telephone call. Using it for customer prioritization would cause data
leakage.

The explicit category `unknown` and the value `pdays = -1` are preserved because
they have a documented meaning in the dataset.

## Model

Logistic Regression, Decision Tree, and XGBoost were compared.

XGBoost achieved the highest test F1 and was selected as the main model. Grid
Search, Random Search, and Bayesian optimization were also tested, but none of
the tuned configurations improved the test F1 of the default XGBoost model.

Selected model:

```text
XGBoost Default
```

Main test metrics:

```text
Accuracy:  0.8934
Precision: 0.5983
Recall:    0.2703
F1:        0.3724
ROC-AUC:   0.7910
```

The main limitation is the low Recall: the model misses many actual
subscribers.

## Environment Setup

Use a dedicated Python environment for the project.

After activating the environment, install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The smaller files `requirements-api.txt` and
`requirements-streamlit.txt` are used for the Docker images.

## Notebook Workflow

The main notebooks are intended to be followed in order:

```text
01_data_exploration.ipynb
02_data_cleaning.ipynb
03_feature_engineering.ipynb
04_baseline_model_training.ipynb
05_hyperparameter_optimization.ipynb
06_error_slicing.ipynb
07_mlflow_experiment_tracking.ipynb
08_refactoring_code.ipynb
09_reproducible_pipeline_execution.ipynb
10_tests_for_code_data_and_model.ipynb
11_local_model_serving_api_streamlit.ipynb
12_containerization_with_docker.ipynb
13_github_actions_ci_demo.ipynb
14_monitoring_and_drift_demo.ipynb
```

The notebooks cover the complete workflow from data exploration and model
development to testing, serving, containerization, CI, and monitoring.

Reusable project logic is located in `src/`.

## Training Pipeline

Run the training pipeline from the repository root:

```bash
python -m src.cli train \
  --params-path reports/model_results/05_selected_xgboost_params.json \
  --no-save-processed-data \
  --log-to-mlflow
```

The pipeline cleans the data, creates features, fits the preprocessor, trains
the selected model, evaluates it, and saves the model and preprocessor in:

```text
models/
```

The model and preprocessor are saved together because inference data must use
the same preprocessing rules as the training data.

## MLflow

Start the local MLflow UI from the repository root:

```bash
python -m mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow records model parameters, metrics, predictions, figures, and model
artifacts.

The local MLflow database and artifacts are ignored by Git.

## Tests

Run the complete test suite:

```bash
python -m pytest
```

The tests cover selected data-cleaning, feature-engineering, evaluation,
prediction, API, CLI, and monitoring behavior.

The current test suite contains 14 passing tests.

## Local API Serving

Start the FastAPI service:

```bash
uvicorn src.serving.api:app --reload --port 8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /health
GET  /model-info
POST /predict
```

The API uses `models/model_manifest.json` to locate the current model and
preprocessor.

## Streamlit Frontend

Start FastAPI first and then run:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

Streamlit sends customer information to the FastAPI service and displays the
returned subscription prediction.

## Docker Compose

Build and run FastAPI and Streamlit together:

```bash
docker compose up --build
```

Open:

```text
FastAPI Swagger: http://127.0.0.1:8000/docs
Streamlit:       http://127.0.0.1:8501
```

Stop the services:

```bash
docker compose down
```

Docker packages the application and its dependencies so that it can run more
consistently in different environments.

## GitHub Actions

The CI workflow is stored in:

```text
.github/workflows/ci.yml
```

It runs the tests, builds the Docker images, starts the API container, checks
the `/health` endpoint, and publishes the API image to GitHub Container Registry on `main`.

The workflow demonstrates CI and container delivery. It does not perform a real
cloud deployment.

## Monitoring and Drift Demo

Successful API predictions are logged locally in:

```text
monitoring/predictions.db
```

Notebook `14_monitoring_and_drift_demo.ipynb` compares current inference data
with reference data and demonstrates basic data-quality, drift, and prediction
monitoring.

If no API logs are available, it creates a synthetic sample for demonstration.

Drift is an investigation signal. It does not automatically prove that model
performance has become worse.

## Limitations

This project is a local prototype, not a production system.

It does not include cloud deployment, authentication, a central model registry,
automated retraining, rollback, or production alerting. Model Recall is also
limited, and part of the monitoring demonstration uses synthetic data.

## Generated Artifacts and Git

The following local artifacts are ignored by Git where appropriate:

```text
models/
mlruns/
mlflow.db
monitoring/
generated interim and processed data
```
Source code, notebooks, tests, configuration, and workflow files remain under
version control.