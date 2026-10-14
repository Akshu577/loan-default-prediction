import json

import joblib
import pandas as pd
import streamlit as st

from src.preprocess import preprocess

st.set_page_config(page_title="Loan Default Risk", page_icon="🏦")
st.title("Loan Default Risk Checker")
st.caption("XGBoost model trained on synthetic data. Run `python main.py` first to train it.")


@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_model.pkl")
    with open("models/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold


model, threshold = load_model()

with st.form("applicant"):
    col1, col2 = st.columns(2)
    income = col1.number_input("Annual income ($)", 15000, 250000, 55000, step=1000)
    loan_amount = col1.number_input("Loan amount ($)", 1000, 60000, 15000, step=500)
    debt = col1.number_input("Existing debt ($)", 0, 250000, 12000, step=500)
    credit_score = col2.slider("Credit score", 300, 850, 680)
    late_payments = col2.number_input("Late payments (history)", 0, 20, 1)
    past_loans = col2.number_input("Past loans", 0, 20, 2)
    submitted = st.form_submit_button("Check risk")

if submitted:
    applicant = preprocess(pd.DataFrame([{
        "income": income, "credit_score": credit_score, "loan_amount": loan_amount,
        "debt": debt, "late_payments": late_payments, "past_loans": past_loans,
    }]))
    prob = float(model.predict_proba(applicant)[0, 1])

    st.metric("Probability of default", f"{prob:.1%}")
    if prob >= threshold:
        st.error(f"High risk: above the {threshold:.0%} cutoff. Recommend manual review.")
    else:
        st.success(f"Lower risk: below the {threshold:.0%} cutoff.")
    st.progress(min(prob, 1.0))
