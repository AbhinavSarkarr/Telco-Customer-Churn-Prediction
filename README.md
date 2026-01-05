# Telco Customer Churn Prediction

A machine learning project that predicts customer churn in the telecommunications industry using ensemble methods and comprehensive feature engineering. The project includes detailed experiment tracking and model optimization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Experiments](#model-experiments)
- [Results](#results)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

## Overview

Customer churn is a critical metric for telecommunications companies. This project builds predictive models to identify customers likely to churn, enabling proactive retention strategies. The implementation includes comprehensive experiment tracking, feature engineering, and ensemble modeling.

### Key Objectives

- Predict customer churn with high recall to minimize missed at-risk customers
- Implement robust experiment tracking for reproducibility
- Engineer features that capture customer behavior patterns
- Compare multiple ML algorithms and ensemble methods

## Features

- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Custom features including avg_monthly_total and tenure groups
- **Experiment Tracking**: Detailed logging of all experiments in markdown format
- **Ensemble Models**: Voting classifier combining Random Forest, XGBoost, and Logistic Regression
- **Stratified Cross-Validation**: Handling class imbalance in validation
- **Hyperparameter Optimization**: Grid search for optimal model parameters

## Dataset

The Telco Customer Churn dataset contains information about:

| Feature Category | Examples |
|------------------|----------|
| **Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Services** | Phone, Internet, Online Security, Streaming |
| **Account** | Tenure, Contract Type, Payment Method |
| **Billing** | Monthly Charges, Total Charges |
| **Target** | Churn (Yes/No) |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/AbhinavSarkarr/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn==1.5.1
xgboost
imblearn
tensorflow
jupytext
```

## Usage

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open the EDA notebook
notebooks/Exploratory Data Analysis (EDA).ipynb
```

### Training the Model

```python
from data_setup import load_and_preprocess_data
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Create ensemble model
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, solver='saga')),
        ('xgb', XGBClassifier(eval_metric='logloss')),
        ('rf', RandomForestClassifier(random_state=42))
    ],
    voting='soft'
)

# Train and save
voting_clf.fit(X_train, y_train)
pickle.dump(voting_clf, open('customer_churn_model.pkl', 'wb'))
```

## Model Experiments

### Experiment Log Summary

| Experiment | Model | CV Score | Test Accuracy | F1 Score | Key Changes |
|------------|-------|----------|---------------|----------|-------------|
| 1 | Random Forest | 0.8379 | 0.7771 | 0.5780 | Initial baseline |
| 2 | Random Forest | 0.8461 | 0.7771 | 0.5780 | Stratified CV |
| 3 | Random Forest | 0.8525 | 0.7722 | 0.5804 | Feature engineering |
| 4 | Random Forest | 0.8525 | 0.7722 | 0.5804 | Grid search optimization |
| 5 | Voting Ensemble | 0.8458 | **0.7821** | **0.6242** | LR + XGB + RF ensemble |

### Feature Engineering

```python
# Average monthly spending pattern
df['avg_monthly_total'] = df['TotalCharges'] / df['tenure']
df['avg_monthly_total'] = df['avg_monthly_total'].replace([np.inf, -np.inf], 0)

# Tenure grouping by quartiles
df['tenure_group'] = pd.qcut(df['tenure'], q=4, labels=['New', 'Early', 'Mid', 'Long'])
```

### Key Insights

1. **Stratified CV** reduced variance across folds (std: 0.0754 → 0.0043)
2. **Feature engineering** improved CV score by ~0.6%
3. **Ensemble voting** achieved best recall (0.6836) for catching at-risk customers

## Results

### Best Model Performance (Voting Classifier)

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.21% |
| **Precision** | 57.43% |
| **Recall** | 68.36% |
| **F1 Score** | 62.42% |

### Cross-Validation (5-Fold Stratified)

- Mean CV Score: 84.58%
- Standard Deviation: 0.40%

### Feature Importance (Top 5)

1. Contract Type
2. Tenure
3. Monthly Charges
4. Total Charges
5. Internet Service Type

## Technologies

| Technology | Purpose |
|------------|---------|
| **scikit-learn** | ML models and preprocessing |
| **XGBoost** | Gradient boosting classifier |
| **pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib/Seaborn** | Visualizations |
| **imbalanced-learn** | Handling class imbalance |
| **TensorFlow** | Neural network experiments |

## Project Structure

```
Telco-Customer-Churn-Prediction/
├── notebooks/
│   └── Exploratory Data Analysis (EDA).ipynb
├── data_setup.py                    # Data loading and preprocessing
├── logger.py                        # Experiment logging utilities
├── ml_experiments.md                # Experiment tracking log
├── customer_churn_model.pkl         # Trained model
├── encoders.pkl                     # Label encoders
├── requirements.txt                 # Dependencies
├── .gitignore
└── README.md                        # This file
```

## Future Improvements

- [ ] Implement SMOTE for better handling of class imbalance
- [ ] Add neural network models for comparison
- [ ] Create a Flask/FastAPI web interface
- [ ] Deploy model on cloud platform
- [ ] Add SHAP values for model interpretability

## Author

**Abhinav Sarkar**
- GitHub: [@AbhinavSarkarr](https://github.com/AbhinavSarkarr)
- LinkedIn: [abhinavsarkarrr](https://www.linkedin.com/in/abhinavsarkarrr)
- Portfolio: [abhinav-ai-portfolio.lovable.app](https://abhinav-ai-portfolio.lovable.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle for the Telco Customer Churn dataset
- scikit-learn documentation and community
- XGBoost development team

---

<p align="center">
  <strong>Predict customer churn before it happens</strong>
</p>
