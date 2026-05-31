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

### Distance-Based Models

- K-Nearest Neighbors (KNN)

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

# Predict whether a customer will churn based on demographic and service-related information.

**Highlights**

- Hidden missing value detection
- Feature encoding
- Feature scaling
- Logistic Regression
- Regularized Logistic Regression
- SGD Classifier
- K-Nearest Neighbors (KNN)
- Hyperparameter tuning using GridSearchCV
- Hyperparameter tuning using RandomizedSearchCV

**Best Model**

KNeighborsClassifier

Best Parameters:

```python
KNeighborsClassifier(
    metric='manhattan',
    n_neighbors=29,
    weights='uniform'
)
```

**Final Test Performance**

| Metric | Score |
|----------|----------|
| Accuracy | 78.43% |
| Precision | 59.29% |
| Recall | 59.29% |
| F1 Score | 59.29% |
| ROC-AUC | 83.39% |

**Key Finding**

Hyperparameter tuning improved KNN significantly, increasing validation F1 score from **54.17%** to **60.22%**. The final model generalized well, achieving a test F1 score of **59.29%** with only a minor drop from validation performance.

---

## Repository Structure

```text
ML-Classification-Lab/
│── Customer Churn Classifier - Logistic Regression.ipynb
│── Customer Churn Classifier - KNN.ipynb
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
