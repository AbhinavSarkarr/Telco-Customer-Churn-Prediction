
## Experiment - 2025-02-02 16:47:03

### Model
Random Forest

### Changes Made
Initial run

### Reasoning
Initial Run

### Hyperparameters
```python
random_state: 42
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.7271
- Fold 2: 0.7668
- Fold 3: 0.9045
- Fold 4: 0.8924
- Fold 5: 0.8985

Summary:
- Mean: 0.8379
- Std: 0.0754

#### Test Metrics
- accuracy: 0.7771
- precision: 0.5795
- recall: 0.5764
- f1: 0.5780


### Observations
Words Decent, Needs Optimization.

---

## Experiment - 2025-02-02 16:50:46

### Model
Random Forest

### Changes Made
Tried using Stratified Cross Validation intead of Normal Cross Validation.

### Reasoning
From the Initial run, i analyzed that the first two folds were having lower results a compared to the other three. So there might e chances that the distribution of  classes the first two classes or the last three classes is not balanced that why i am trying to use Stratified CV 

### Hyperparameters
```python
random_state: 42
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.8490
- Fold 2: 0.8447
- Fold 3: 0.8520
- Fold 4: 0.8453
- Fold 5: 0.8393

Summary:
- Mean: 0.8461
- Std: 0.0043

#### Test Metrics
- accuracy: 0.7771
- precision: 0.5795
- recall: 0.5764
- f1: 0.5780


### Observations
No changes in the Metrices but 1% increase in the CV score

---

## Experiment - 2025-02-02 17:13:10

### Model
Random Forest

### Changes Made
Did Feature Engineering created the new features avg_monthly_total = TotalCharges / tenure:   This feature helps capture the customer's spending pattern over time A high average monthly total might indicate premium customers who are less likely to churn Customers with sudden increases in their average monthly charges might be more likely to churn It provides a normalized view of spending that accounts for how long they've been a customer   Replacing infinities with 0:   When tenure is 0 (new customers), division will create infinity Setting these to 0 helps identify brand new customers This is important because new customers often have different churn patterns than established ones   tenure_group using quartiles:   Customer behavior often isn't linear with tenure For example:  New customers (0-25%) might churn due to initial experiences Mid-tenure customers (25-75%) might have different churn reasons Long-term customers (75-100%) might be more stable   Grouping helps the model identify these non-linear patterns better than raw tenure values  Looking at your data:  You have both tenure and TotalCharges Currently, they're treated independently By combining them and creating groups, we give the model more structured information about customer segments

### Reasoning
avg_monthly_total = TotalCharges / tenure:   This feature helps capture the customer's spending pattern over time A high average monthly total might indicate premium customers who are less likely to churn Customers with sudden increases in their average monthly charges might be more likely to churn It provides a normalized view of spending that accounts for how long they've been a customer   Replacing infinities with 0:   When tenure is 0 (new customers), division will create infinity Setting these to 0 helps identify brand new customers This is important because new customers often have different churn patterns than established ones   tenure_group using quartiles:   Customer behavior often isn't linear with tenure For example:  New customers (0-25%) might churn due to initial experiences Mid-tenure customers (25-75%) might have different churn reasons Long-term customers (75-100%) might be more stable   Grouping helps the model identify these non-linear patterns better than raw tenure values  Looking at your data:  You have both tenure and TotalCharges Currently, they're treated independently By combining them and creating groups, we give the model more structured information about customer segments

### Hyperparameters
```python
random_state: 42
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.8527
- Fold 2: 0.8574
- Fold 3: 0.8550
- Fold 4: 0.8532
- Fold 5: 0.8441

Summary:
- Mean: 0.8525
- Std: 0.0045

#### Test Metrics
- accuracy: 0.7722
- precision: 0.5663
- recall: 0.5952
- f1: 0.5804


### Observations
Nothing as such changed

---

## Experiment - 2025-02-02 17:20:38

### Model
Random Forest

### Changes Made
Tried using Grid Search CV

### Reasoning
Wanted to find the optimal params and it provided 

### Hyperparameters
```python
max_depth: 0
min_samples_leaf: 1
'min_samples_split: 2
n_estimators: 100
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.8527
- Fold 2: 0.8574
- Fold 3: 0.8550
- Fold 4: 0.8532
- Fold 5: 0.8441

Summary:
- Mean: 0.8525
- Std: 0.0045

#### Test Metrics
- accuracy: 0.7722
- precision: 0.5663
- recall: 0.5952
- f1: 0.5804


### Observations
No such changes 

---



## Experiment - 2025-02-02 17:35:05

### Model
Random Forest, Linear Regression, Decision Tree

### Changes Made
Create a voting classifier using these three models

### Reasoning
Tried to use an ensemble of model \

### Hyperparameters
```python
hyperparameters = {
    'Logistic Regression': {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'saga',
        'penalty': 'l2',
        'C': 1.0
    },
    'XGBoost': {
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'Random Forest': {
        'random_state': 42
    },
    'Voting Classifier': {
        'voting': 'soft'
    },
    'Cross-Validation': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42,
        'scoring': 'accuracy'
    }
}
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.8478
- Fold 2: 0.8471
- Fold 3: 0.8399
- Fold 4: 0.8514
- Fold 5: 0.8429

Summary:
- Mean: 0.8458
- Std: 0.0040

#### Test Metrics
- accuracy: 0.7821
- precision: 0.5743
- recall: 0.6836
- f1: 0.6242


### Observations
1 percent increase in recall and minor increase in other metrices 

---

## Experiment - 2025-02-02 18:10:00

### Model
Rando Forest

### Changes Made
Tried using "SelectFromModel", a feature importance module from sklear with random forest 

### Reasoning
May be some of the features are leading to low accuracy

### Hyperparameters
```python
None
```

### Results
#### Cross-Validation Scores
Individual scores:
- Fold 1: 0.7271
- Fold 2: 0.7668
- Fold 3: 0.9045
- Fold 4: 0.8924
- Fold 5: 0.8985

Summary:
- Mean: 0.8379
- Std: 0.0754

#### Test Metrics
- accuracy: 0.7821
- precision: 0.5873
- recall: 0.5952
- f1: 0.5912


### Observations
Results got even bad

---
