"""
Week 8 - Ensemble Model
Filename: 5_ensemble_model.py (or ensemble_model.py)
Input: ckd_40k_with_creatinine.csv
Output: Combined model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier  # ← ALL IMPORTS HERE!
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("WEEK 8: Ensemble Model")
print("=" * 60)

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

print(f"\nDataset: {len(df)} samples")
print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
print(f"Features: {len(feature_cols)} (including Serum Creatinine)")

print("\n" + "=" * 60)
print("Creating ensemble of 3 models:")
print("=" * 60)
print("  1. Random Forest (n_estimators=200, max_depth=20)")
print("  2. Gradient Boosting (n_estimators=100)")
print("  3. Logistic Regression (max_iter=1000)")

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft'  # Use probability averaging
)

print("\nTraining ensemble (this may take 5-10 minutes)...")
ensemble.fit(X_train_scaled, y_train)
print("✓ Ensemble trained")

# Evaluate individual models
print(f"\n{'=' * 60}")
print("INDIVIDUAL MODEL PERFORMANCE")
print(f"{'=' * 60}")

for name, model in ensemble.named_estimators_.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name.upper()}:")
    print(f"  Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Evaluate ensemble
print(f"\n{'=' * 60}")
print("ENSEMBLE PERFORMANCE")
print(f"{'=' * 60}")

y_pred_ensemble = ensemble.predict(X_test_scaled)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
prec_ensemble = precision_score(y_test, y_pred_ensemble)
rec_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

print(f"Accuracy:  {acc_ensemble:.4f} ({acc_ensemble * 100:.2f}%)")
print(f"Precision: {prec_ensemble:.4f}")
print(f"Recall:    {rec_ensemble:.4f}")
print(f"F1-Score:  {f1_ensemble:.4f}")

# Cross-validation
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION ON ENSEMBLE")
print("=" * 60)
print("Performing cross-validation (this may take 5-10 minutes)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\nCV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Classification report
print(f"\n{'=' * 60}")
print("DETAILED CLASSIFICATION REPORT")
print(f"{'=' * 60}")
print(classification_report(y_test, y_pred_ensemble,
                            target_names=['Healthy', 'CKD'],
                            digits=4))

# Final comparison
print(f"\n{'=' * 60}")
print("WEEK 8 FINAL COMPARISON")
print(f"{'=' * 60}")

# Get individual model accuracies
rf_acc = accuracy_score(y_test, ensemble.named_estimators_['rf'].predict(X_test_scaled))
gb_acc = accuracy_score(y_test, ensemble.named_estimators_['gb'].predict(X_test_scaled))
lr_acc = accuracy_score(y_test, ensemble.named_estimators_['lr'].predict(X_test_scaled))

print(f"\nModel                          Test Accuracy")
print("-" * 60)
print(f"Week 6 (no creatinine)         97.6%")
print(f"Week 8 Random Forest           {rf_acc:.4f} ({rf_acc * 100:.2f}%)")
print(f"Week 8 Gradient Boosting       {gb_acc:.4f} ({gb_acc * 100:.2f}%)")
print(f"Week 8 Logistic Regression     {lr_acc:.4f} ({lr_acc * 100:.2f}%)")
print(f"Week 8 ENSEMBLE (combined)     {acc_ensemble:.4f} ({acc_ensemble * 100:.2f}%)")

print(f"\nEnsemble Cross-Validation:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Key metrics summary
print(f"\n{'=' * 60}")
print("KEY METRICS SUMMARY")
print(f"{'=' * 60}")
print(f"Best Test Accuracy:            {acc_ensemble:.4f} ({acc_ensemble * 100:.2f}%)")
print(f"Best Precision:                {prec_ensemble:.4f}")
print(f"Best Recall:                   {rec_ensemble:.4f} (catches {rec_ensemble * 100:.1f}% of CKD)")
print(f"Best F1-Score:                 {f1_ensemble:.4f}")
print(f"Consistency (CV std dev):      ±{cv_scores.std():.4f}")

print(f"\n{'=' * 60}")
print("ENSEMBLE MODEL COMPLETE!")
print(f"{'=' * 60}")
print("✓ Final model ready for deployment")
print("✓ Combines strengths of 3 different algorithms")
print("✓ Validated with 5-fold cross-validation")
print("\nNext: Run 6_create_visualizations.py for comparison charts")