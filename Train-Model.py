# ============================================
# Adult Income Classification (Correct Columns)
# ============================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("adult.csv")   # change filename if needed

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# ============================================
# 2. Clean Target
# ============================================

df["income"] = df["income"].str.strip()

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# ============================================
# 3. Split Features & Target
# ============================================

X = df.drop("income", axis=1)
y = df["income"]

# ============================================
# 4. Identify Column Types
# ============================================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:", list(num_cols))
print("Categorical Columns:", list(cat_cols))

# ============================================
# 5. Preprocessing
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

# ============================================
# 6. Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# 7. Define Models
# ============================================

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Create model folder
os.makedirs("model", exist_ok=True)

results = []

# ============================================
# 8. Train & Evaluate
# ============================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    # Save model
    joblib.dump(pipeline, f"model/{name}.pkl")
    print(f"{name} saved successfully!")

# ============================================
# 9. Show Results
# ============================================

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
])

print("\nFinal Model Comparison:")
print(results_df)