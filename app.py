import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Classification App")
st.write("Upload a test dataset CSV file to evaluate the trained models.")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

# ---------------------------------------------------
# Model Selection
# ---------------------------------------------------

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ---------------------------------------------------
# Upload Dataset
# ---------------------------------------------------

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Check required column
    if "income" not in df.columns:
        st.error("Dataset must contain 'income' column.")
    else:
        # Separate features and target
        X = df.drop("income", axis=1)
        y = df["income"]

        # Convert target to binary if needed
        y = y.replace({">50K": 1, "<=50K": 0})

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        y_pred = model.predict(X)

        try:
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        except:
            auc = "Not Available"

        # ---------------------------------------------------
        # Evaluation Metrics
        # ---------------------------------------------------

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.write("## Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC Score", f"{auc if isinstance(auc,str) else f'{auc:.4f}'}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # ---------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------

        st.write("## Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["<=50K", ">50K"],
                    yticklabels=["<=50K", ">50K"])
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        st.pyplot(fig)

        # ---------------------------------------------------
        # Classification Report
        # ---------------------------------------------------

        st.write("## Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

else:
    st.info("Please upload a CSV file to begin evaluation.")
