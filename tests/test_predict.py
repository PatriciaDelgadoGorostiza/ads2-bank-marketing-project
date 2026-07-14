import pandas as pd
import pytest

from src.modeling.predict import predict


class DummyProbabilityModel:
    def predict_proba(self, X):
        return [[0.8, 0.2], [0.4, 0.6], [0.51, 0.49]]


@pytest.mark.unit
def test_predict_uses_positive_class_probability_and_threshold():
    X = pd.DataFrame({"feature": [1, 2, 3]})

    predictions = predict(DummyProbabilityModel(), X, threshold=0.5)

    assert predictions["churn_probability"].tolist() == [0.2, 0.6, 0.49]
    assert predictions["churn_prediction"].tolist() == [0, 1, 0]
