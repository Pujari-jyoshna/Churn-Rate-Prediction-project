import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import preprocess_data

X, y, scaler = preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Evaluation:\n", classification_report(y_test, y_pred))

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
