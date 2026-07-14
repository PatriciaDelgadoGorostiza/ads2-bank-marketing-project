import os

import requests
import streamlit as st

# In local development, the FastAPI service runs on localhost.
# In a deployed setup, this should come from an environment variable, for example:
# API_URL=https://telco-churn-api.example.com
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def build_customer_payload() -> dict:
    """Collect one customer record from the Streamlit form.

    Streamlit returns Python values from the widgets. We convert these values
    into the JSON-like dictionary expected by the FastAPI `/predict` endpoint.
    """
    with st.form("customer_form"):
        # The API accepts customerID as optional metadata. It is useful for
        # tracing a prediction back to the submitted customer record.
        customer_id = st.text_input("Customer ID", value="demo-customer")

        # Columns keep the form readable. They do not affect the API payload.
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )

        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)

        submitted = st.form_submit_button("Predict churn")

    # The keys in this payload must match the Pydantic schema in
    # src/serving/schemas.py. This is the contract between Streamlit and FastAPI.
    payload = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return payload, submitted


st.set_page_config(page_title="Telco Churn Demo", layout="wide")
st.title("Telco Churn Prediction")

st.caption("This demo sends one customer record to the local FastAPI model service.")

payload, submitted = build_customer_payload()

if submitted:
    try:
        # Streamlit acts as a client. It does not load model artifacts directly.
        # The model inference logic stays behind the FastAPI boundary.
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()
    except requests.RequestException as exc:
        # This usually means the FastAPI service is not running or not reachable.
        st.error(f"Could not call prediction API: {exc}")
    else:
        probability = prediction["churn_probability"]
        # Show a compact business-facing result first, then metadata below.
        st.metric("Churn probability", f"{probability:.1%}")
        st.metric("Prediction", "Churn" if prediction["churn_prediction"] == 1 else "No churn")
        st.write(
            {
                "customerID": prediction["customerID"],
                "threshold": prediction["threshold"],
                "model_name": prediction["model_name"],
                "model_version": prediction["model_version"],
            }
        )

with st.expander("API connection"):
    st.write(f"API URL: `{API_URL}`")
    st.write("Start the API before using this app.")
