# Adult Income Classification – Machine Learning Project

## 1️⃣ Problem Statement

The objective of this project is to build a Machine Learning classification system that predicts whether an individual's income exceeds \$50K per year based on demographic and employment-related attributes.

The system allows users to upload a test dataset and evaluate multiple trained machine learning models through a Streamlit web application.

---

## 2️⃣ Dataset Description

The dataset used is the Adult Census Income dataset.

### Target Variable:
- `income`
  - <=50K → 0
  - >50K → 1

### Feature Columns:

| Feature | Description |
|----------|-------------|
| age | Age of individual |
| workclass | Employment type |
| fnlwgt | Final sampling weight |
| education | Education level |
| education.num | Education number (numeric encoding) |
| marital.status | Marital status |
| occupation | Occupation type |
| relationship | Family relationship |
| race | Race category |
| sex | Gender |
| capital.gain | Capital gain |
| capital.loss | Capital loss |
| hours.per.week | Weekly working hours |
| native.country | Country of origin |

---

## 3️⃣ Models Used

The following classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All models were trained using a Scikit-learn Pipeline with:
- StandardScaler (for numerical features)
- OneHotEncoder (for categorical features)
- ColumnTransformer for preprocessing

---

## 4️⃣ Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Classification Report

---

## 5️⃣ Model Comparison & Observations

| Model | Strengths | Observations |
|--------|------------|--------------|
| Logistic Regression | Simple & interpretable | Performs well with balanced data |
| Decision Tree | Easy to visualize | May overfit |
| KNN | Simple algorithm | Sensitive to scaling |
| Naive Bayes | Fast & efficient | Assumes feature independence |
| Random Forest | High accuracy | More robust than Decision Tree |
| XGBoost | Strong performance | Best generalization in most cases |

### Observations:
- Random Forest and XGBoost typically achieve higher accuracy and AUC.
- Logistic Regression provides good baseline performance.
- Naive Bayes performs fast but may slightly underperform due to independence assumptions.
- Ensemble methods generally outperform single-tree models.

---

## 6️⃣ Streamlit App Features

The Streamlit web application includes:

✔ Dataset upload option (CSV file – test data only)  
✔ Model selection dropdown  
✔ Display of evaluation metrics  
✔ Confusion matrix visualization  
✔ Classification report table  

---

## 7️⃣ Project Structure

AdultIncomeProject/
│
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ │-- logistic_regression.pkl
│ │-- decision_tree.pkl
│ │-- knn.pkl
│ │-- naive_bayes.pkl
│ │-- random_forest.pkl
│ │-- xgboost.pkl

## 8️⃣ Installation & Running Locally

### Clone Repository

git clone https://github.com/SundaraMoorthySakthivel/AdultIncomePrediction.git
cd Adult-Income-Project

### Install Dependencies

pip install -r requirements.txt

### Run Streamlit App

streamlit run app.py


---

## 9️⃣ Deployment on Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Sign in using GitHub
3. Click “New App”
4. Select repository
5. Choose branch (main)
6. Select app.py
7. Click Deploy

The deployed application link will be generated automatically.

---

## 10 Requirements

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib

---

## 📌 Conclusion

This project demonstrates a complete Machine Learning workflow including:

- Data preprocessing
- Model training
- Performance evaluation
- Model comparison
- Web deployment using Streamlit

The system provides an interactive platform for evaluating classification models on unseen test datasets.