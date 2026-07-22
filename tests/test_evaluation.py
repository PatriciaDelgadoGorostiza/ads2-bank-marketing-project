import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_classifier


class FixedProbabilityModel:
    """Small deterministic classifier used only for evaluation tests."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = X["score"].to_numpy(dtype=float)

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X["score"].to_numpy(dtype=float) >= 0.5
        ).astype(int)


@pytest.mark.unit
def test_evaluate_classifier_returns_expected_classification_metrics():
    model = FixedProbabilityModel()

    X_train = pd.DataFrame(
        {
            "score": [0.90, 0.80, 0.20, 0.10],
        }
    )
    y_train = pd.Series([1, 1, 0, 0])

    X_test = pd.DataFrame(
        {
            "score": [0.90, 0.80, 0.20, 0.10],
        }
    )
    y_test = pd.Series([1, 0, 1, 0])

    result = evaluate_classifier(
        model=model,
        model_name="Test Classifier",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=0.5,
    )

    assert result["model_name"] == "Test Classifier"
    assert result["accuracy"] == pytest.approx(0.50)
    assert result["precision"] == pytest.approx(0.50)
    assert result["recall"] == pytest.approx(0.50)
    assert result["f1"] == pytest.approx(0.50)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["brier_score"] == pytest.approx(0.325)
    assert result["train_f1"] == pytest.approx(1.00)
    assert result["test_f1"] == pytest.approx(0.50)
    assert result["generalization_gap"] == pytest.approx(0.50)
    assert result["threshold"] == pytest.approx(0.50)