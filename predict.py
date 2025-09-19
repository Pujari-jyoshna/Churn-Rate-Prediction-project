import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    return "Churn" if prediction[0] == 1 else "No Churn"

if __name__ == "__main__":
    sample = {
        "CustomerID": 2001,
        "Gender": 1,
        "Age": 30,
        "Tenure": 10,
        "InternetService": 0,
        "Contract": 1,
        "MonthlyCharges": 75
    }
    print("Prediction:", predict_churn(sample))
