"""Streamlit transaction-risk scoring interface."""

from __future__ import annotations

import streamlit as st

from src.predict import predict_amounts
from src.utils import MODEL_DIR, SEQUENCE_LENGTH

st.set_page_config(page_title="Banking Fraud Detection", page_icon="🏦", layout="centered")
st.title("🏦 Banking Fraud Detection System")
st.caption("LSTM transaction sequence risk scoring")

if not (MODEL_DIR / "fraud_lstm_model.h5").is_file():
    st.warning("The trained model is not available yet. Place the dataset in data/ and run `python -m src.train`.")
    st.stop()

with st.form("transaction_form"):
    customer_id = st.text_input("Customer ID", placeholder="e.g. CUST-1042")
    current_amount = st.number_input("Current Transaction Amount", min_value=0.0, value=125.0, step=1.0)
    history_input = st.text_input(
        "Previous Transaction History (9 comma-separated amounts)",
        placeholder="24.50, 31.00, 19.99, 54.00, 42.00, 18.50, 25.00, 39.00, 47.00, 125.00",
    )
    submitted = st.form_submit_button("Assess transaction", type="primary", use_container_width=True)

if submitted:
    try:
        history = [float(value.strip()) for value in history_input.split(",") if value.strip()]
        if len(history) != SEQUENCE_LENGTH - 1:
            raise ValueError(f"Enter {SEQUENCE_LENGTH - 1} previous amounts; the current amount is appended as transaction {SEQUENCE_LENGTH}.")
        if any(amount < 0 for amount in history):
            raise ValueError("Transaction amounts cannot be negative.")
        result = predict_amounts(history + [float(current_amount)])
    except ValueError as error:
        st.error(str(error))
    else:
        probability = float(result["fraud_probability"])
        st.subheader(f"Assessment for {customer_id or 'customer'}")
        st.metric("Fraud probability", f"{probability:.1%}", delta=result["risk_level"])
        st.progress(probability, text=f"{probability:.1%} estimated fraud risk")
        st.bar_chart({"Probability": {"Fraud": probability, "Genuine": 1 - probability}}, y_label="Probability")
        if result["risk_level"] == "HIGH":
            st.error(f"HIGH RISK. {result['recommendation']}")
        elif result["risk_level"] == "MEDIUM":
            st.warning(f"MEDIUM RISK. {result['recommendation']}")
        else:
            st.success(f"LOW RISK. {result['recommendation']}")