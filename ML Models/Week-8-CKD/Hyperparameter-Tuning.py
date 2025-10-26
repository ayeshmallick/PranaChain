"""
Week 8 - Hyperparameter Tuning
Filename: 3_hyperparameter_tuning.py
Input: ckd_40k_with_creatinine.csv
Output: Optimized Random Forest model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # THIS WAS MISSING!
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("WEEK 8: Hyperparameter Tuning")
print("="*60)

# Load data
df = pd.read_csv('ckd_40k_with_creatinine.csv')
feature_cols = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']
X = df[feature_cols]
y = (df['class'] == 'ckd').astype(int)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("Hyperparameter Tuning - Random Forest")
print("="*60)

print("\nSearching for optimal hyperparameters...")
print("This will test multiple combinations (may take 15-20 minutes)")

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

total_combinations = (len(param_grid['n_estimators']) *
                     len(param_grid['max_depth']) *
                     len(param_grid['min_samples_split']) *
                     len(param_grid['min_samples_leaf']) *
                     len(param_grid['max_features']))

print(f"Testing {total_combinations} parameter combinations with 5-fold CV...")

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nStarting grid search...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n{'='*60}")
print("BEST PARAMETERS FOUND")
print(f"{'='*60}")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# Test the tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred_tuned)
prec = precision_score(y_test, y_pred_tuned)
rec = recall_score(y_test, y_pred_tuned)
f1 = f1_score(y_test, y_pred_tuned)

print(f"\n{'='*60}")
print("TUNED MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1-Score:       {f1:.4f}")

# Compare with default
print("\nTraining default Random Forest for comparison...")
default_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
default_rf.fit(X_train_scaled, y_train)
y_pred_default = default_rf.predict(X_test_scaled)
acc_default = accuracy_score(y_test, y_pred_default)

improvement = (acc - acc_default) * 100

print(f"\n{'='*60}")
print("COMPARISON: Default vs Tuned")
print(f"{'='*60}")
print(f"Default Random Forest:  {acc_default:.4f} ({acc_default*100:.2f}%)")
print(f"Tuned Random Forest:    {acc:.4f} ({acc*100:.2f}%)")
print(f"Improvement:            {'+' if improvement > 0 else ''}{improvement:.2f}%")

print(f"\n{'='*60}")
print("HYPERPARAMETER TUNING COMPLETE!")
print(f"{'='*60}")
print("Next: Run 4_shap_explainability.py")