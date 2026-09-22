# Credit Scoring & Risk Assessment System

## Skillairo AI Internship — Major Project

**Project:** Credit Scoring & Risk Assessment System  
**Domain:** Artificial Intelligence / Machine Learning  
**Author:** Chetany Kalaneya

---

## 1. Project Overview

The Credit Scoring & Risk Assessment System is an end-to-end machine learning application designed to assess credit risk using applicant financial information.

The project uses the **UCI Statlog German Credit Dataset** and includes:

- Data preprocessing
- Categorical encoding
- Numerical feature processing
- Classification model training
- Cross-validation
- Model comparison
- ROC analysis
- Confusion matrix analysis
- Asymmetric financial cost modeling
- Decision threshold optimization
- Feature importance / decision justification
- Interactive risk assessment
- Batch applicant evaluation

---

## 2. Problem Statement

Financial institutions need systematic methods for evaluating whether an applicant represents a higher or lower credit risk.

The objective of this project is to build a classification system that predicts credit risk from applicant information while considering that different classification errors can have different financial costs.

---

## 3. Dataset

**Dataset:** Statlog (German Credit)

**Source:** UCI Machine Learning Repository

Dataset page:  
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

The supplied implementation maps the bad-credit class to the positive/default class for risk analysis.

---

## 4. Technologies Used

### Programming
- Python

### Machine Learning
- Scikit-learn

### Data Processing
- Pandas
- NumPy
- ColumnTransformer
- RobustScaler
- OneHotEncoder

### Classification Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees

### Evaluation
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- Confusion Matrix

### Risk Optimization
- Asymmetric 5:1 cost matrix
- Decision threshold optimization

### Application
- Interactive web interface
- Batch CSV evaluation

---

## 5. Machine Learning Workflow

```text
Applicant Data
      ↓
Data Preprocessing
      ↓
Numerical Scaling
      ↓
Categorical Encoding
      ↓
Train/Test Split
      ↓
5-Fold Stratified Cross-Validation
      ↓
Multiple Classification Models
      ↓
Performance Evaluation
      ↓
Cost-Sensitive Threshold Optimization
      ↓
Risk Assessment
      ↓
Decision Justification
```

---

## 6. Classification Models

The project evaluates:

### 1. Logistic Regression
Used as an interpretable linear classification baseline.

### 2. Random Forest
An ensemble of decision trees designed to capture non-linear relationships.

### 3. Gradient Boosting
A boosting-based classifier using sequential decision trees.

### 4. Extra Trees
An extremely randomized tree ensemble providing another non-linear classification approach.

---

## 7. Model Evaluation

The project evaluates models using:

- Cross-validation ROC-AUC
- Test ROC-AUC
- Accuracy
- Recall
- F1 Score
- Specificity
- Financial cost

### Champion Model

**Logistic Regression**

Reported test results:

- **Test ROC-AUC:** 0.8087
- **Accuracy:** 74.5%
- **Recall:** 80.0%
- **F1 Score:** 0.6531
- **Specificity:** 72.1%

These values are based on the supplied project evaluation artifacts.

---

## 8. Cost-Sensitive Risk Modeling

The system uses an asymmetric financial cost matrix:

| Actual / Predicted | Good (0) | Bad (1) |
|---|---:|---:|
| Actual Good (0) | 0 DM | 1 DM |
| Actual Bad (1) | 5 DM | 0 DM |

This represents a higher cost for a false-negative decision where a bad-credit applicant is incorrectly approved.

The system evaluates different classification thresholds to find an operating point with lower empirical financial cost.

### Optimized Threshold

**τ = 0.47**

The stored evaluation reports:

- Default average cost: **0.495 DM/applicant**
- Optimized average cost: **0.455 DM/applicant**
- Reported cost reduction: **8.1%**

These results are evaluation results for this project implementation and should not be interpreted as a real-world lending policy.

---

## 9. Decision Justification

The system does not return only a classification.

It provides additional decision-support information including:

- Influential applicant features
- Positive credit factors
- Key risk flags
- Credit assessment outcome
- What-If scenario simulation

This helps demonstrate the internship requirement:

> Justify model decisions.

---

## 10. Application Features

### Risk Assessor
Users can enter applicant information and receive a risk assessment.

### Model Performance
Provides:

- Classifier leaderboard
- Cross-validation results
- Test performance
- ROC curve
- Cost curve
- Feature importance
- Confusion matrix comparison

### Batch Evaluation
Users can upload multiple applicant records using CSV and process them together.

### What-If Analysis
Users can modify selected applicant/loan parameters and recalculate the scenario.

---

## 11. Project Structure

```text
Credit Scoring & Risk Assessment System/
│
├── app/
├── data/
├── models/
├── static/
├── templates/
├── tests/
├── requirements.txt
├── app.py
└── README.md
```

The exact folders may vary slightly depending on the project package version.

---

## 12. How to Run

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Credit-Scoring-Risk-Assessment
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the application

```bash
python app.py
```

Open the local URL shown by the application in your browser.

---

## 13. Security & Responsible Use

This project is an educational internship implementation.

It should not be used as a real lending decision engine without additional:

- Data validation
- Model calibration
- Fairness testing
- Bias assessment
- Regulatory review
- Privacy controls
- Model monitoring
- Human oversight
- Production validation

The displayed risk result is a machine-learning output, not a financial recommendation.

---

## 14. Learning Outcomes

The project demonstrates:

- Classification modeling
- Credit-risk modeling
- Data preprocessing
- Feature engineering
- Stratified cross-validation
- ROC-AUC analysis
- Confusion matrix analysis
- Cost-sensitive learning
- Threshold optimization
- Explainable decision support
- Batch prediction
- Machine learning application development

---

## 15. Internship Submission

**Internship:** Skillairo AI Internship  
**Project Type:** Major Project  
**Submission Deadline:** 20 October 2026

**GitHub Repository:**  
`<ADD YOUR VERIFIED GITHUB LINK HERE>`

**Hosted Application:**  
`<ADD HOSTED LINK IF AVAILABLE>`

---

## 16. Author

**Chetany Kalaneya**

Artificial Intelligence / Machine Learning Enthusiast
