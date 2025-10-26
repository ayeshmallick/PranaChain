# Train Models with Serum Creatinine Feature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Week 8: Training with Serum Creatinine Feature")
print("=" * 60)

# Load data
df = pd.read_csv('ckd_40k_with_creatinine.csv')

# NEW: 11 features now (added 'sc')
feature_cols = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']
X = df[feature_cols]
y = (df['class'] == 'ckd').astype(int)

print(f"Features: {len(feature_cols)} ({feature_cols})")
print(f"NEW FEATURE: Serum Creatinine (sc)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)  # NEW!
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"Training: {name}")
    print(f"{'=' * 40}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    # Train
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_acc': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    print(f"CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# Compare
print(f"\n{'=' * 60}")
print("COMPARISON: With vs Without Serum Creatinine")
print(f"{'=' * 60}")
print(f"\nWeek 6 (10 features, no creatinine): 97.6% accuracy")
print(f"Week 8 (11 features, WITH creatinine): {results['Random Forest']['test_acc'] * 100:.2f}% accuracy")
print(f"\nImprovement: +{(results['Random Forest']['test_acc'] - 0.976) * 100:.2f}%")

# Feature importance
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'=' * 60}")
print("Feature Importance (Random Forest)")
print(f"{'=' * 60}")
for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    marker = "🆕" if row['feature'] == 'sc' else ""
    print(f"{i:2d}. {row['feature']:<10} : {row['importance']:.4f} {marker}")