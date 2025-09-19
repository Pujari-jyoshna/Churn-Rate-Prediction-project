import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction App")

customer_id = st.text_input("Customer ID")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 30)
tenure = st.slider("Tenure (months)", 0, 60, 12)
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
charges = st.number_input("Monthly Charges", min_value=20, max_value=150, value=70)

if st.button("Predict"):
    data = {
        "CustomerID": customer_id,
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Tenure": tenure,
        "InternetService": 0 if internet=="DSL" else 1,
        "Contract": {"Month-to-month":0,"One year":1,"Two year":2}[contract],
        "MonthlyCharges": charges
    }

    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    result = "⚠️ Customer likely to Churn" if prediction[0] == 1 else "✅ Customer will Stay"
    st.subheader(result)
