# ML Classification Lab

A hands-on machine learning repository focused on understanding, implementing, and comparing classification algorithms through complete end-to-end projects.

The goal of this repository is not only to train models but to build intuition about:

- Data preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Evaluation metrics
- Model interpretation
- Generalization performance

---

## Project Workflow

Every project follows a structured machine learning pipeline:

```text
EDA
↓
Train / Validation / Test Split
↓
Missing Value Handling
↓
Encoding
↓
Feature Scaling
↓
Baseline Model
↓
Model Comparison
↓
Hyperparameter Tuning
↓
Validation Evaluation
↓
Final Test Evaluation
```

---

## Algorithms Covered

### Linear Models

- Logistic Regression
- Regularized Logistic Regression (L1 / L2)
- SGD Classifier

### Support Vector Machines

- Linear SVM
- Kernel SVM
- Hyperparameter Tuning

### Tree-Based Models

- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier

### Ensemble Methods

- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

---

## Evaluation Metrics

Models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Metric selection depends on the problem.

For imbalanced datasets, F1 Score and Recall receive special attention.

---

## Completed Projects

### 1. Telco Customer Churn Prediction

**Objective**

Predict whether a customer will churn based on demographic and service-related information.

**Highlights**

- Hidden missing value detection
- Feature encoding
- Scaling
- Logistic Regression
- SGD Classifier
- Hyperparameter tuning using RandomizedSearchCV

**Best Model**

SGDClassifier

**Final Test Performance**

| Metric | Score |
|----------|----------|
| Accuracy | 79.75% |
| Precision | 62.79% |
| Recall | 57.86% |
| F1 Score | 60.22% |
| ROC-AUC | 84.62% |

---

## Repository Structure

```text
ML-Classification-Lab/
│── Customer Churn Classifier - Logistic Regression.ipynb
└── README.md
```

---

## Key Learnings

This repository emphasizes understanding:

- Why preprocessing choices matter
- How data leakage occurs
- When scaling is required
- Differences between linear and non-linear models
- The impact of hyperparameter tuning
- Trade-offs between Precision and Recall
- Proper train-validation-test workflows

---

## Future Work

- Feature Selection
- Cross Validation Pipelines
- Threshold Tuning
- Probability Calibration
- Advanced Ensemble Methods
- Real-world Imbalanced Classification Problems

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Author

Smarth Sharma
