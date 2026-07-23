import os

import requests
import streamlit as st

# In local development, the FastAPI service runs on localhost.
# In Docker Compose, API_URL is set to http://api:8000.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def build_customer_payload() -> tuple[dict, bool]:
    """Collect one Bank Marketing customer record from the Streamlit form."""
    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            job = st.selectbox(
                "Job",
                [
                    "admin.",
                    "blue-collar",
                    "entrepreneur",
                    "housemaid",
                    "management",
                    "retired",
                    "self-employed",
                    "services",
                    "student",
                    "technician",
                    "unemployed",
                    "unknown",
                ],
                index=4,
            )
            marital = st.selectbox("Marital status", ["divorced", "married", "single"], index=1)
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"], index=2)
            default = st.selectbox("Credit in default", ["no", "yes"])
            balance = st.number_input("Account balance", value=1500)
            housing = st.selectbox("Housing loan", ["no", "yes"], index=1)
            loan = st.selectbox("Personal loan", ["no", "yes"])

        with col2:
            contact = st.selectbox("Contact type", ["cellular", "telephone", "unknown"])
            day = st.number_input("Last contact day", min_value=1, max_value=31, value=15)
            month = st.selectbox(
                "Last contact month",
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                index=4,
            )
            campaign = st.number_input("Contacts during this campaign", min_value=1, value=2)
            pdays = st.number_input("Days since previous contact", min_value=-1, value=-1)
            previous = st.number_input("Previous contacts", min_value=0, value=0)
            poutcome = st.selectbox("Previous campaign outcome", ["failure", "other", "success", "unknown"], index=3)

        submitted = st.form_submit_button("Predict subscription")

    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }
    return payload, submitted


st.set_page_config(page_title="Bank Marketing Demo", layout="wide")
st.title("Bank Marketing Subscription Prediction")
st.caption("This demo sends one customer record to the local FastAPI model service.")

payload, submitted = build_customer_payload()

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not call prediction API: {exc}")
    else:
        probability = prediction["subscription_probability"]
        st.metric("Subscription probability", f"{probability:.1%}")
        st.metric(
            "Prediction",
            "Likely to subscribe" if prediction["subscription_prediction"] == 1 else "Unlikely to subscribe",
        )
        st.write(
            {
                "threshold": prediction["threshold"],
                "model_name": prediction["model_name"],
                "model_version": prediction["model_version"],
            }
        )

with st.expander("API connection"):
    st.write(f"API URL: `{API_URL}`")
    st.write("Start the API before using this app.")