import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Adult Income Prediction", layout="centered")

st.title("💼 Adult Income Classification App")
st.write("Predict whether income is >50K or <=50K")

# -----------------------------
# Load Models
# -----------------------------
MODEL_PATH = "model"

models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_PATH, "logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_PATH, "decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_PATH, "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_PATH, "naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_PATH, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_PATH, "xgboost.pkl"))
}

selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Person Details")

age = st.number_input("Age", min_value=17, max_value=90, value=30)

workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])

fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)

education = st.selectbox("Education", [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
    "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
])

education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)

marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent"
])

occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])

relationship = st.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
])

race = st.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
])

sex = st.selectbox("Sex", ["Male", "Female"])

capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

native_country = st.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines",
    "Germany", "Canada", "England", "China",
    "Cuba", "Jamaica", "South", "Italy"
])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Income"):

    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1 or prediction == ">50K":
        st.success("💰 Predicted Income: >50K")
    else:
        st.info("📉 Predicted Income: <=50K")
