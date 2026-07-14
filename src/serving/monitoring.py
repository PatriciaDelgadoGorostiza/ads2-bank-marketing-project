import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MONITORING_DIR


DEFAULT_MONITORING_DB_PATH = MONITORING_DIR / "predictions.db"
MONITORING_DB_PATH_ENV = "MONITORING_DB_PATH"


def get_monitoring_db_path() -> Path:
    """Return the SQLite database path used for local prediction logging."""
    return Path(os.getenv(MONITORING_DB_PATH_ENV, DEFAULT_MONITORING_DB_PATH))


def initialize_monitoring_db(db_path: Path | None = None) -> Path:
    """Create the local monitoring database and prediction log table if needed."""
    db_path = db_path or get_monitoring_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                customer_id TEXT,
                request_json TEXT NOT NULL,
                tenure REAL,
                monthly_charges REAL,
                total_charges REAL,
                contract TEXT,
                internet_service TEXT,
                payment_method TEXT,
                churn_probability REAL NOT NULL,
                churn_prediction INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                threshold REAL NOT NULL
            )
            """
        )
        connection.commit()

    return db_path


def _as_float_or_none(value: Any) -> float | None:
    """Convert monitoring feature values to floats where possible."""
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_prediction(
    request_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Append one incoming request and its prediction to the monitoring log.

    The full request is stored as JSON for traceability. A small set of important
    features is also stored as regular columns so simple drift checks can query
    them without parsing JSON first.
    """
    db_path = initialize_monitoring_db(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                timestamp_utc,
                customer_id,
                request_json,
                tenure,
                monthly_charges,
                total_charges,
                contract,
                internet_service,
                payment_method,
                churn_probability,
                churn_prediction,
                model_name,
                model_version,
                threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                request_payload.get("customerID"),
                json.dumps(request_payload, sort_keys=True),
                _as_float_or_none(request_payload.get("tenure")),
                _as_float_or_none(request_payload.get("MonthlyCharges")),
                _as_float_or_none(request_payload.get("TotalCharges")),
                request_payload.get("Contract"),
                request_payload.get("InternetService"),
                request_payload.get("PaymentMethod"),
                float(prediction_payload["churn_probability"]),
                int(prediction_payload["churn_prediction"]),
                prediction_payload["model_name"],
                prediction_payload["model_version"],
                float(prediction_payload["threshold"]),
            ),
        )
        connection.commit()


def load_prediction_logs(db_path: Path | None = None) -> pd.DataFrame:
    """Load prediction logs from the local SQLite monitoring database."""
    db_path = db_path or get_monitoring_db_path()
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id", connection)
