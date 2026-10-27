
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")

feature_columns = model.feature_names_in_

training_data["TotalCharges"] = pd.to_numeric(
    training_data["TotalCharges"].str.strip(),
    errors="coerce"
).fillna(0)

X_reference = training_data.drop(columns=["customerID", "Churn"])
X_reference = pd.get_dummies(X_reference, drop_first=True, dtype=int)
feature_columns = X_reference.columns
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "An AI/ML project that predicts whether a telecom customer "
    "is likely to leave the company."
)

st.sidebar.title("About the Project")
st.sidebar.write("Machine Learning Model: Logistic Regression")
st.sidebar.write("Dataset: Telco Customer Churn")
st.sidebar.write("Test Accuracy: 80.70%")
with st.form("customer_form"):
    st.header("Customer Details")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 100000.0, 840.0)

    submitted = st.form_submit_button("Predict Customer Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    customer_encoded = pd.get_dummies(customer, drop_first=True, dtype=int)
    customer_encoded = customer_encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = model.predict(customer_encoded)[0]

    probabilities = model.predict_proba(customer_encoded)[0]
    churn_probability = (
        probabilities[list(model.classes_).index("Yes")] * 100
    )

    st.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.1f}%"
    )

    st.progress(int(round(churn_probability)))

    if prediction == "Yes":
        st.error("Prediction: This customer may leave the company.")
    else:
        st.success("Prediction: This customer may stay with the company.")