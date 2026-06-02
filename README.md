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


### Ensemble Methods

- XGBoost

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
- Decision Trees
- Ensemble methods like Random Forest
- Boosting models like XGBoost
- Model comparison and validation analysis

---

## Best Model

### Tuned Balanced Random Forest

```python
RandomForestClassifier(
    class_weight="balanced",
    max_depth=10,
    max_features="log2",
    min_samples_leaf=8,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
```

---

## Final Test Performance

| Metric | Score |
|----------|----------:|
| Accuracy | 76.63% |
| Precision | 54.09% |
| Recall | 77.86% |
| F1 Score | **63.84%** |
| ROC-AUC | 77.02% |

---

## Key Findings

### 1. Class Imbalance Handling Was Critical

The largest performance improvements across multiple models came from assigning balanced class weights.

Models using:

```python
class_weight="balanced"
```

consistently achieved higher Recall and F1 Scores by properly accounting for the minority churn class.

---

### 2. Hyperparameter Tuning Significantly Improved Random Forest Performance

The Random Forest improved from:

| Model | F1 Score |
|----------|----------:|
| Baseline Random Forest | 53.39% |
| Tuned Random Forest | **63.84%** |

This represents an improvement of approximately **10.45 percentage points** in F1 Score.

The tuning process successfully reduced overfitting while improving minority-class detection.

---

### 3. Ensemble Learning Outperformed a Single Decision Tree

Although the tuned Decision Tree achieved strong performance, the Random Forest surpassed it on the held-out test set.

Random Forest benefited from averaging multiple decision trees, reducing variance and improving generalization.

| Model | Test F1 |
|----------|----------:|
| Tuned Balanced Decision Tree | 62.73% |
| Tuned Balanced Random Forest | **63.84%** |

---

### 4. Strong Generalization Was Achieved

| Evaluation Stage | F1 Score |
|----------|----------:|
| Cross Validation | 64.09% |
| Validation Set | 62.80% |
| Test Set | 63.84% |

The close agreement between validation and test performance indicates minimal overfitting and strong generalization capability.

---

### 5. Recall Remained a Priority Metric

Customer churn prediction is an imbalanced classification problem.

The tuned Random Forest achieved:

- Recall: **77.86%**
- F1 Score: **63.84%**

This means the model successfully identified a large proportion of customers likely to churn while maintaining reasonable precision.

---


### 6. Naive Bayes Demonstrated the Recall–Precision Tradeoff

Gaussian Naive Bayes achieved excellent Recall while generating more False Positives than other models.

This experiment highlighted how different algorithms optimize different aspects of classification performance.

---

### 7. XGBoost Achieved the Highest ROC-AUC

XGBoost was evaluated using baseline training, hyperparameter tuning, and class-weight balancing.

| Model Variant | F1 Score | ROC-AUC |
|---------------|---------:|---------:|
| Baseline XGBoost | 52.22% | 80.07% |
| Tuned XGBoost | 59.04% | 83.63% |
| Tuned Weighted XGBoost | 61.72% | **84.31%** |

Although XGBoost achieved the highest ROC-AUC score in the repository, the Tuned Balanced Random Forest achieved a higher Test F1 Score and remained the top-performing model according to the selected evaluation metric.

---

## Model Insights

### Logistic Regression

- Strong baseline model.
- Easy to interpret.
- Demonstrated the effectiveness of linear decision boundaries.

### K-Nearest Neighbors

- Benefited significantly from proper scaling.
- Showed sensitivity to hyperparameter choices.
- Provided competitive performance after tuning.

### Gaussian Naive Bayes

- Extremely fast training and inference.
- Uses probability estimation instead of optimization.
- Assumes feature independence.
- Achieved excellent Recall but lower Precision.
- Demonstrated how strong assumptions can still produce competitive results.

### Support Vector Machines

- Strong nonlinear classification performance.
- Benefited significantly from class balancing.
- RBF Kernel captured complex decision boundaries.
- Hyperparameter tuning substantially improved performance.
- Remained among the strongest models tested.

### Decision Tree Classifier

- Highly interpretable and easy to visualize.
- Did not require feature scaling.
- Benefited heavily from class balancing.
- Demonstrated excellent generalization.
- Showed that shallow trees can outperform more sophisticated algorithms when properly tuned.

### Random Forest Classifier

- Best-performing model in the repository.
- Successfully reduced overfitting through hyperparameter tuning.
- Benefited significantly from balanced class weights.
- Achieved the highest overall F1 Score among all tested models.
- Demonstrated strong generalization across cross-validation, validation, and test datasets.
- Improved minority-class detection substantially compared to the baseline model.
- Showed the effectiveness of ensemble learning over a single Decision Tree.

### XGBoost Classifier

- Strongest boosting-based model evaluated.
- Achieved the highest ROC-AUC score in the repository.
- Hyperparameter tuning significantly improved performance.
- Class weighting substantially increased recall on the minority class.
- Generalized well from validation to test data.
- Finished as the second-best model based on Test F1 Score.

---

## Current Model Leaderboard

### Test Set Results

#### Since the dataset was imbalanced the models are evaluated on F1 Score.

| Rank | Model | F1 Score |
|------|---------|---------:|
| 🥇 | Tuned Balanced Random Forest | **63.84%** |
| 🥈 | Tuned Balanced Decision Tree | **62.73%** |
| 🥉 | Tuned Weighted XGBoost | **61.72%** |
| 4 | Tuned Balanced RBF SVM | 62.32% |
| 5 | Tuned Balanced Linear SVM | 60.24% |
| 6 | Tuned KNN (Random Search) | 59.29% |
| 7 | Gaussian Naive Bayes | 59.16% |
| 8 | Tuned RBF SVM | 57.09% |
| 9 | Baseline Linear SVM | 56.31% |
| 10 | Baseline RBF SVM | 54.78% |
| 11 | Baseline KNN | 54.17% |
| 12 | Baseline Random Forest | 53.39% |
| 13 | Baseline XGBoost | 52.22% |



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
├── Customer Churn Classifier - Decision Tree.ipynb
├── Customer Churn Classifier - Random Forest.ipynb
├── Customer Churn Classifier - XGBoost.ipynb
│
└── README.md
```

---

## Key Learnings

This repository emphasizes understanding:

- Why preprocessing choices matter
- How data leakage occurs
- When scaling is required
- Differences between linear, probabilistic, and tree-based models
- How Bayes' Theorem is used in classification
- The impact of feature independence assumptions
- The importance of handling class imbalance
- Trade-offs between Precision and Recall
- Proper train-validation-test workflows
- How model complexity affects generalization
- Why shallow trees often generalize better than deep trees
- The impact of hyperparameter tuning on different model families
- The importance of maintaining an untouched test set

---

## Future Work

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

