# ML Classification Lab

A hands-on machine learning repository focused on understanding, implementing, tuning, and comparing classification algorithms through complete end-to-end projects.

The objective of this repository is not only to train models, but to develop strong intuition about:

- Data preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Evaluation metrics
- Class imbalance handling
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

- Logistic Regression
- Regularized Logistic Regression (L1 / L2)
- SGD Classifier

### Distance-Based Models

- K-Nearest Neighbors (KNN)
- GridSearchCV Tuning
- RandomizedSearchCV Tuning

### Probabilistic Models

- Gaussian Naive Bayes
- Validation-Based Hyperparameter Tuning
- Probabilistic Classification using Bayes' Theorem

### Support Vector Machines

- Linear SVM
- Class-Balanced Linear SVM
- RBF Kernel SVM
- Class-Balanced RBF SVM
- GridSearchCV Hyperparameter Optimization

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

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Metric selection depends on the business problem.

For imbalanced classification tasks such as customer churn prediction, F1 Score and Recall are prioritized over Accuracy.

---

# Customer Churn Prediction

Predict whether a customer will churn based on demographic, account, and service-related information.

---

## Highlights

- Hidden missing value detection
- Data cleaning and preprocessing
- Feature encoding
- Feature scaling
- Logistic Regression
- Regularized Logistic Regression
- SGD Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Linear Support Vector Machine
- RBF Kernel Support Vector Machine
- Class imbalance handling using class weights
- Hyperparameter tuning using GridSearchCV
- Hyperparameter tuning using RandomizedSearchCV
- Probabilistic classification using Bayes' Theorem
- Model comparison and validation analysis

---

## Best Model

### Tuned Class-Balanced RBF SVM

```python
SVC(
    kernel="rbf",
    C=1,
    gamma=0.1,
    class_weight="balanced",
    random_state=42
)
```

---

## Final Test Performance

| Metric | Score |
|----------|----------|
| Accuracy | 74.83% |
| Precision | 51.64% |
| Recall | 78.57% |
| F1 Score | 62.32% |
| ROC-AUC | 82.39% |

---

## Key Findings

### 1. Class Imbalance Handling Was Critical

The largest performance improvements came from introducing balanced class weights.

Both Linear and RBF SVMs showed substantial improvements in Recall and F1 Score after accounting for class imbalance.

---

### 2. Linear SVM Was Surprisingly Competitive

The tuned class-balanced Linear SVM achieved:

- Validation F1 Score: 60.24%
- Validation Recall: 79.00%

This demonstrated that relatively simple decision boundaries were already effective for the churn dataset.

---

### 3. RBF SVM Achieved the Best Overall Performance

After tuning:

```python
C = 1
gamma = 0.1
class_weight = "balanced"
```

the RBF SVM achieved the highest validation and test F1 Scores.

---

### 4. Hyperparameter Tuning Matters

Significant performance gains were achieved through systematic hyperparameter optimization using GridSearchCV and RandomizedSearchCV.

---

### 5. Naive Bayes Was a Strong Recall-Oriented Model

Gaussian Naive Bayes achieved:

| Metric | Score |
|----------|----------|
| Accuracy | 68.78% |
| Precision | 45.27% |
| Recall | 85.36% |
| F1 Score | 59.16% |
| ROC-AUC | 81.48% |

Key observations:

- Achieved the highest Recall among all models tested so far.
- Successfully identified approximately 85% of all churning customers.
- Produced a strong ROC-AUC despite its simplicity.
- Generated a large number of False Positives, reducing Precision.
- Hyperparameter tuning produced virtually no improvement, indicating that performance was dominated by model assumptions rather than tuning.

The experiment highlighted the trade-off between maximizing Recall and maintaining Precision.

---

## Model Insights

### Logistic Regression

- Strong baseline model.
- Easy to interpret.
- Demonstrated the effectiveness of linear decision boundaries.

### K-Nearest Neighbors

- Benefited significantly from proper scaling.
- Showed sensitivity to hyperparameter choices such as the number of neighbors.
- Provided competitive performance after tuning.

### Gaussian Naive Bayes

- Extremely fast training and inference.
- Uses probability estimation instead of optimization.
- Assumes feature independence.
- Achieved excellent Recall but lower Precision.
- Demonstrated how strong assumptions can still produce competitive results.

### Support Vector Machines

- Delivered the strongest overall performance.
- Benefited heavily from class balancing.
- RBF Kernel captured nonlinear relationships effectively.
- Hyperparameter tuning significantly improved results.

---

## Current Model Leaderboard

### Test Set Results
| Rank | Model | F1 Score |
|------|---------|---------:|
| 🥇 | Tuned Balanced RBF SVM | **62.32%** |
| 🥈 | Tuned Balanced Linear SVM | 60.24% |
| 🥉 | Gaussian Naive Bayes | 59.16% |
| 4 | Tuned KNN (Random Search) | 59.29%* |
| 5 | Tuned RBF SVM | 57.09% |
| 6 | Baseline Linear SVM | 56.31% |
| 7 | Baseline RBF SVM | 54.78% |
| 8 | Baseline KNN | 54.17% |


\* KNN score obtained from a different experimental configuration. Final rankings will be updated once all models are evaluated using a consistent test protocol.

---

## Repository Structure

```text
ML-Classification-Lab/
│
├── Customer Churn Classifier - Logistic Regression.ipynb
├── Customer Churn Classifier - KNN.ipynb
├── Customer Churn Classifier - Naive Bayes.ipynb
├── Customer Churn Classifier - SVM.ipynb
│
└── README.md
```

---

## Key Learnings

This repository emphasizes understanding:

- Why preprocessing choices matter
- How data leakage occurs
- When scaling is required
- Differences between linear and nonlinear models
- Differences between discriminative and probabilistic models
- How Bayes' Theorem is used in classification
- The impact of feature independence assumptions
- The importance of handling class imbalance
- Trade-offs between Precision and Recall
- Proper train-validation-test workflows
- How model complexity affects generalization
- The impact of hyperparameter tuning on different model families

---

## Future Work

### Tree-Based Models

- Decision Trees
- Random Forests
- Extra Trees

### Ensemble Models

- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

### Advanced Topics

- Threshold Tuning
- Probability Calibration
- Feature Selection
- Cross-Validation Pipelines
- Advanced Imbalanced Classification Techniques
- Explainable AI (SHAP / Feature Importance)

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Author

**Smarth Sharma**

Electronics and Communication Engineering (ECE)  
NSUT Delhi

Focused on building strong intuition in Machine Learning through implementation, experimentation, and rigorous model comparison.
