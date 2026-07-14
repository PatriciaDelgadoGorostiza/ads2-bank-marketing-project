import pandas as pd
import pytest


@pytest.fixture
def minimal_telco_input() -> pd.DataFrame:
    """Small Telco-style input row without the target column."""
    return pd.DataFrame(
        [
            {
                "customerID": "A",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 80.0,
                "TotalCharges": 700.0,
            }
        ]
    )
