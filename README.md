# Adult Income Classification using Machine Learning

---

## a. Problem Statement

The objective of this project is to build machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes.

This is a binary classification problem where:

- Class 0 → Income <= 50K
- Class 1 → Income > 50K

The project compares the performance of multiple machine learning models using various evaluation metrics.

---

## b. Dataset Description

Dataset: Adult Income Dataset (Census Income Dataset)

Source: UCI Machine Learning Repository / Kaggle

Number of Instances: 48,842  
Number of Features: 14  

Features include:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country
- etc.

Target Variable:
- income (<=50K or >50K)

The dataset satisfies the assignment requirements:
- More than 12 features
- More than 500 instances
- Binary classification problem

---

## c. Models Used and Evaluation Metrics

The following six machine learning models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble Boosting)

### Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score| MCC
|-------|----------|-----|-----------|--------|----------|-----|
Logistic Regression | 0.854291 | 0.904345 | 0.739366 | 0.609694 | 0.668298 | 0.580438
Decision Tree | 0.812068 | 0.746233 | 0.607635 | 0.619260 | 0.613392 | 0.489307
KNN | 0.831107 | 0.852494 | 0.666193 | 0.598214 | 0.630376 | 0.522602 
Naive Bayes | 0.537540 | 0.735753 | 0.336355 | 0.946429 | 0.496321 | 0.324052
Random Forest | 0.847229 | 0.899472 | 0.714286 | 0.690566 | 0.657487 | 0.562920
XGBoost | 0.871181 | 0.923278 | 0.774680 | 0.655612 | 0.710190 | 0.631726

---

## Observations

1. Logistic Regression performed well due to the linear separability of some features.
2. Decision Tree showed moderate performance but may overfit.
3. KNN performance depends on feature scaling and data distribution.
4. Naive Bayes performed reasonably despite independence assumptions.
5. Random Forest improved performance through ensemble learning and reduced overfitting.
6. XGBoost achieved the best performance due to gradient boosting optimization.

Overall, ensemble methods (Random Forest and XGBoost) outperformed individual models.

---

## Deployment

The project is deployed using Streamlit Community Cloud.

Users can:
- Select a trained model
- Enter feature values
- Get real-time income prediction
