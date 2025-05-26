# 📊 Credit Approval Analysis and Prediction

This project focuses on analyzing and predicting credit approval decisions using combined internal banking data and external credit bureau data. It includes comprehensive data preprocessing, statistical feature selection, and training an XGBoost classification model.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)

---

## 🧾 Introduction

This project utilizes two datasets:
- **Internal bank data**
- **External credit bureau data**

The objective is to:
- Merge both datasets based on a common identifier (`PROSPECTID`)
- Preprocess the combined data
- Select statistically significant features
- Train a robust classification model to predict credit approval levels

---

## 📦 Requirements


Install the required Python packages using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels xgboost

Required Libraries:
numpy
pandas
matplotlib
scikit-learn
scipy
statsmodels
xgboost

⚙️ Usage
Ensure both internal and external datasets are available in .xlsx format. Run the notebook or script in sequence for:
Data loading
Preprocessing
Feature selection
Model training
Evaluation

🧹 Data Preprocessing
Key steps in preprocessing:
Load both datasets from Excel files.
Replace placeholder missing values (-99999.00) with NaN.

Drop:
Rows with missing values in the internal dataset
Columns with over 10,000 NaN values and remaining rows with NaN from the external dataset
Merge the cleaned datasets on the PROSPECTID key.

🎯 Feature Selection
🔸 Categorical Features
Chi-Square Test: Applied to identify statistically significant categorical variables.

🔸 Numerical Features
Variance Inflation Factor (VIF): Used to detect multicollinearity.

ANOVA Test: Applied to assess the significance of numerical features in predicting the target.

🧪 Encoding Techniques
Label Encoding: Applied to EDUCATION.

One-Hot Encoding: Applied to MARITALSTATUS, GENDER, last_prod_enq2, and first_prod_enq2.

The selected and encoded features are consolidated into a final feature set for model training.

🤖 Model Training
Classifier Used: XGBoost
The dataset is split into training and test sets. Hyperparameter tuning is performed using GridSearchCV with the following search space:
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

Final Model Parameters:
objective='multi:softmax'

num_class=4
learning_rate=0.2
max_depth=3
n_estimators=200

📈 Model Evaluation
The model is evaluated using key classification metrics on the test set:

Accuracy: 0.78

📊 Per-Class Performance
Class	Precision	Recall	F1 Score
p1	0.8467	0.7623	0.8023
p2	0.8166	0.9302	0.8697
p3	0.4704	0.2634	0.3377
p4	0.7399	0.7269	0.7333

📌 Conclusion
This project demonstrates an end-to-end pipeline for credit approval prediction using a combination of feature engineering, statistical analysis, and machine learning. While overall accuracy and performance are promising, further improvement may be achieved by:
--Enhancing class p3 performance through SMOTE or cost-sensitive learning
--Incorporating additional features or time-series data
--Experimenting with model ensembles





