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
                request_json TEXT NOT NULL,
                age REAL,
                balance REAL,
                campaign REAL,
                pdays REAL,
                previous REAL,
                job TEXT,
                contact TEXT,
                month TEXT,
                subscription_probability REAL NOT NULL,
                subscription_prediction INTEGER NOT NULL,
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
    """Append one Bank Marketing request and its prediction to the monitoring log."""
    db_path = initialize_monitoring_db(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                timestamp_utc,
                request_json,
                age,
                balance,
                campaign,
                pdays,
                previous,
                job,
                contact,
                month,
                subscription_probability,
                subscription_prediction,
                model_name,
                model_version,
                threshold
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(request_payload, sort_keys=True),
                _as_float_or_none(request_payload.get("age")),
                _as_float_or_none(request_payload.get("balance")),
                _as_float_or_none(request_payload.get("campaign")),
                _as_float_or_none(request_payload.get("pdays")),
                _as_float_or_none(request_payload.get("previous")),
                request_payload.get("job"),
                request_payload.get("contact"),
                request_payload.get("month"),
                float(prediction_payload["subscription_probability"]),
                int(prediction_payload["subscription_prediction"]),
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
        return pd.read_sql_query(
            "SELECT * FROM prediction_logs ORDER BY id",
            connection,
        )