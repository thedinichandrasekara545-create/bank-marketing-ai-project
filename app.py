import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Marketing Predictor")
st.title("Bank Term Deposit Subscription Predictor")
st.write("Enter customer details to estimate whether the customer is likely to subscribe to a term deposit.")

model = joblib.load("bank_marketing_model_compressed.pkl")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Credit in Default?", ["no", "yes"])
balance = st.number_input("Average Yearly Balance", value=1000)
housing = st.selectbox("Housing Loan?", ["no", "yes"])
loan = st.selectbox("Personal Loan?", ["no", "yes"])
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
month = st.selectbox("Last Contact Month", [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
])
campaign = st.number_input("Contacts During Current Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Previous Contact (-1 = never contacted)", min_value=-1, value=-1)
previous = st.number_input("Previous Contacts", min_value=0, value=0)
poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "other", "failure", "success"])

input_data = pd.DataFrame([{
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
    "poutcome": poutcome
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]

    if prediction == 1:
        st.success(f"Likely to subscribe. Probability: {probability:.2%}")
    else:
        st.warning(f"Not likely to subscribe. Probability: {probability:.2%}")
