# ML Classification Lab

A hands-on machine learning repository focused on understanding, implementing, tuning, and comparing classification algorithms through complete end-to-end projects.

The objective of this repository is not only to train models, but to develop strong intuition about:

* Data preprocessing
* Feature engineering
* Model selection
* Hyperparameter tuning
* Evaluation metrics
* Class imbalance handling
* Model interpretation
* Generalization performance

---

## Project Workflow

Every project follows a structured machine learning pipeline:

```text
EDA
↓
Train / Validation / Test Split
↓
Data Cleaning
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

* Logistic Regression
* Regularized Logistic Regression (L1 / L2)
* SGD Classifier

### Distance-Based Models

* K-Nearest Neighbors (KNN)
* GridSearchCV Tuning
* RandomizedSearchCV Tuning

### Support Vector Machines

* Linear SVM
* Class-Balanced Linear SVM
* RBF Kernel SVM
* Class-Balanced RBF SVM
* GridSearchCV Hyperparameter Optimization

### Tree-Based Models

* Decision Tree Classifier
* Random Forest Classifier
* Extra Trees Classifier

### Ensemble Methods

* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Metric selection depends on the business problem.

For imbalanced classification tasks such as customer churn prediction, F1 Score and Recall are prioritized over Accuracy.

---

# Customer Churn Prediction

Predict whether a customer will churn based on demographic, account, and service-related information.

## Highlights

* Hidden missing value detection
* Data cleaning and preprocessing
* Feature encoding
* Feature scaling
* Logistic Regression
* Regularized Logistic Regression
* SGD Classifier
* K-Nearest Neighbors (KNN)
* Linear Support Vector Machine
* RBF Kernel Support Vector Machine
* Class imbalance handling using class weights
* Hyperparameter tuning using GridSearchCV
* Hyperparameter tuning using RandomizedSearchCV
* Model comparison and validation analysis

---

## Best Model

### Tuned Class-Balanced RBF SVM

```python
SVC(
    kernel='rbf',
    C=1,
    gamma=0.1,
    class_weight='balanced',
    random_state=42
)
```

---

## Final Test Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 74.83% |
| Precision | 51.64% |
| Recall    | 78.57% |
| F1 Score  | 62.32% |
| ROC-AUC   | 82.39% |

---

## Key Findings

### 1. Class Imbalance Handling Was Critical

The largest performance improvements came from introducing balanced class weights.

Both Linear and RBF SVMs showed substantial improvements in Recall and F1 Score after accounting for class imbalance.

### 2. Linear SVM Was Surprisingly Competitive

The tuned class-balanced Linear SVM achieved:

* Validation F1 Score: 60.24%
* Validation Recall: 79.00%

This demonstrated that relatively simple decision boundaries were already effective for the churn dataset.

### 3. RBF SVM Achieved the Best Overall Performance

After tuning:

```python
C = 1
gamma = 0.1
class_weight = "balanced"
```

the RBF SVM achieved the highest validation and test F1 scores.

### 4. Hyperparameter Tuning Matters

Significant performance gains were achieved through systematic hyperparameter optimization using GridSearchCV and RandomizedSearchCV.

---

## Current Model Leaderboard

| Model                     |   F1 Score |
| ------------------------- | ---------: |
| Tuned Balanced RBF SVM    | **62.32%** |
| Tuned Balanced Linear SVM |     60.24% |
| Tuned KNN (Random Search) |     59.29% |
| Baseline Linear SVM       |     56.31% |
| Tuned RBF SVM             |     57.09% |
| Baseline KNN              |     54.17% |
| Baseline RBF SVM          |     54.78% |

---

## Repository Structure

```text
ML-Classification-Lab/
│
├── Customer Churn Classifier - Logistic Regression.ipynb
├── Customer Churn Classifier - KNN.ipynb
├── Customer Churn Classifier - SVM.ipynb
│
└── README.md
```

---

## Key Learnings

This repository emphasizes understanding:

* Why preprocessing choices matter
* How data leakage occurs
* When scaling is required
* Differences between linear and nonlinear models
* The impact of hyperparameter tuning
* The importance of handling class imbalance
* Trade-offs between Precision and Recall
* Proper train-validation-test workflows
* How model complexity affects generalization

---

## Future Work

* Decision Trees
* Random Forests
* Gradient Boosting
* XGBoost
* LightGBM
* Threshold Tuning
* Probability Calibration
* Feature Selection
* Cross-Validation Pipelines
* Advanced Imbalanced Classification Techniques

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

---

## Author

Smarth Sharma
