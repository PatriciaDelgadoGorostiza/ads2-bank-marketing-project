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

DEFAULT_BUSINESS_PARAMS = {
    "retention_offer_cost": 50,
    "customer_value_if_retained": 300,
    "retention_success_rate": 0.30,
}


def calculate_business_kpis(
    y_true: pd.Series,
    y_score,
    business_params: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Estimate campaign value from individual churn probabilities."""
    if business_params is None:
        business_params = DEFAULT_BUSINESS_PARAMS

    offer_cost = business_params["retention_offer_cost"]
    customer_value = business_params["customer_value_if_retained"]
    success_rate = business_params["retention_success_rate"]

    business_df = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).reset_index(drop=True),
            "churn_probability": y_score,
        }
    )
    business_df["expected_retention_value"] = (
        business_df["churn_probability"] * customer_value * success_rate
    )
    business_df["expected_net_value"] = business_df["expected_retention_value"] - offer_cost
    business_df["contact_customer"] = business_df["expected_net_value"] > 0

    contacted = business_df[business_df["contact_customer"]]
    true_churn_contacted = int((contacted["y_true"] == 1).sum())
    non_churn_contacted = int((contacted["y_true"] == 0).sum())
    missed_churn = int(((business_df["y_true"] == 1) & (~business_df["contact_customer"])).sum())

    expected_gross_value = contacted["expected_retention_value"].sum()
    campaign_cost = len(contacted) * offer_cost
    expected_net_value = expected_gross_value - campaign_cost
    realized_net_value = true_churn_contacted * customer_value * success_rate - campaign_cost

    return {
        "contacted_customers": int(len(contacted)),
        "true_churn_customers_contacted": true_churn_contacted,
        "non_churn_customers_contacted": non_churn_contacted,
        "missed_churn_customers": missed_churn,
        "expected_net_value": float(expected_net_value),
        "realized_net_value_for_backtest": float(realized_net_value),
    }


def evaluate_classifier(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    business_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted binary classifier with model and business metrics."""
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

    if business_params is not None:
        if y_test_score is None:
            raise ValueError("Business KPI calculation requires prediction probabilities.")
        result.update(calculate_business_kpis(y_test, y_test_score, business_params))

    return result
