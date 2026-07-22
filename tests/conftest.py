import pandas as pd
import pytest


@pytest.fixture
def minimal_bank_marketing_input() -> pd.DataFrame:
    """Small Bank Marketing input sample without target or leakage columns."""

    return pd.DataFrame(
        [
            {
                "age": 40,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            }
        ]
    )