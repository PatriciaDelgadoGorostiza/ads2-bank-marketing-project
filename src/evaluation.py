from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def evaluate_classifier(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a fitted binary classifier with model metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_test_score = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_score >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_test_score)
        brier_score = brier_score_loss(y_test, y_test_score)
    else:
        y_test_score = None
        roc_auc = None
        brier_score = None

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    result = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": test_f1,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "generalization_gap": train_f1 - test_f1,
        "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            zero_division=0,
        ),
        "y_pred": y_test_pred,
        "y_score": y_test_score,
    }

    return result
