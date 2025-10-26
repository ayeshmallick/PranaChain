"""
Week 8 - SHAP Explainability (FIXED v2)
Filename: shap_explainability.py
Handles different SHAP versions correctly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("WEEK 8: Model Explainability with SHAP")
print("=" * 60)
print(f"SHAP version: {shap.__version__}")

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

# Train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("✓ Model trained")

# FIXED: Use unscaled data for SHAP (TreeExplainer works better with original features)
print("\nCreating SHAP explainer on UNSCALED data...")
# Retrain on unscaled data for better SHAP interpretability
model_unscaled = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_unscaled.fit(X_train, y_train)
explainer = shap.TreeExplainer(model_unscaled)

# Get sample
n_samples = 100
X_test_sample = X_test.iloc[:n_samples]
y_test_sample = y_test.iloc[:n_samples]

# Compute SHAP values
print(f"Computing SHAP values for {n_samples} test samples...")
shap_values = explainer(X_test_sample)

print("✓ SHAP values computed")
print(f"  SHAP object type: {type(shap_values)}")

# Extract the values for CKD class (class 1)
if hasattr(shap_values, 'values'):
    # Newer SHAP version returns Explanation object
    if len(shap_values.values.shape) == 3:
        # Shape is (n_samples, n_features, n_classes)
        shap_values_ckd = shap_values.values[:, :, 1]
    else:
        # Shape is (n_samples, n_features)
        shap_values_ckd = shap_values.values

    print(f"  SHAP values shape: {shap_values_ckd.shape}")
    print(f"  Features shape: {X_test_sample.shape}")
else:
    # Older SHAP version returns list
    shap_values_ckd = shap_values[1]
    print(f"  SHAP values shape: {shap_values_ckd.shape}")
    print(f"  Features shape: {X_test_sample.shape}")

# Create visualizations
print("\n" + "=" * 60)
print("CREATING SHAP VISUALIZATIONS")
print("=" * 60)

# 1. Summary Plot (Beeswarm)
print("\n1. Creating SHAP summary plot...")
plt.figure(figsize=(10, 8))

if hasattr(shap_values, 'values'):
    # Use the Explanation object directly
    shap.plots.beeswarm(shap_values[:, :, 1], show=False)
else:
    # Use old API
    shap.summary_plot(shap_values_ckd, X_test_sample, show=False)

plt.title("SHAP Feature Importance - Impact on CKD Prediction", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_summary.png")
plt.close()

# 2. Bar Plot
print("\n2. Creating SHAP bar plot...")
plt.figure(figsize=(10, 6))

if hasattr(shap_values, 'values'):
    shap.plots.bar(shap_values[:, :, 1], show=False)
else:
    shap.summary_plot(shap_values_ckd, X_test_sample, plot_type="bar", show=False)

plt.title("Mean Absolute SHAP Values - Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_bar.png")
plt.close()

# 3. Waterfall plot for CKD patient (replaces force plot - more readable)
print("\n3. Creating SHAP waterfall plot for CKD patient...")
ckd_indices = np.where(y_test_sample == 1)[0]

if len(ckd_indices) > 0:
    ckd_idx = ckd_indices[0]

    plt.figure(figsize=(10, 8))

    if hasattr(shap_values, 'values'):
        shap.plots.waterfall(shap_values[ckd_idx, :, 1], show=False)
    else:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_ckd[ckd_idx],
                base_values=explainer.expected_value[1],
                data=X_test_sample.iloc[ckd_idx].values,
                feature_names=feature_cols
            ),
            show=False
        )

    plt.title("SHAP Waterfall - Why This Patient Was Flagged as CKD", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_waterfall_ckd.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: shap_waterfall_ckd.png")
    plt.close()
else:
    print("⚠️ No CKD patients in sample")

# 4. Waterfall plot for healthy patient
print("\n4. Creating SHAP waterfall plot for healthy patient...")
healthy_indices = np.where(y_test_sample == 0)[0]

if len(healthy_indices) > 0:
    healthy_idx = healthy_indices[0]

    plt.figure(figsize=(10, 8))

    if hasattr(shap_values, 'values'):
        shap.plots.waterfall(shap_values[healthy_idx, :, 1], show=False)
    else:
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_ckd[healthy_idx],
                base_values=explainer.expected_value[1],
                data=X_test_sample.iloc[healthy_idx].values,
                feature_names=feature_cols
            ),
            show=False
        )

    plt.title("SHAP Waterfall - Why This Patient Was Classified as Healthy", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_waterfall_healthy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: shap_waterfall_healthy.png")
    plt.close()
else:
    print("⚠️ No healthy patients in sample")

# 5. Feature importance comparison
print("\n5. Creating feature importance comparison...")
plt.figure(figsize=(12, 6))

# SHAP-based importance
if hasattr(shap_values, 'values'):
    shap_importance = np.abs(shap_values.values[:, :, 1]).mean(0)
else:
    shap_importance = np.abs(shap_values_ckd).mean(0)

# Traditional Random Forest importance
rf_importance = model_unscaled.feature_importances_

# Create comparison
x = np.arange(len(feature_cols))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width / 2, rf_importance, width, label='Random Forest', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width / 2, shap_importance, width, label='SHAP', alpha=0.8, color='coral')

ax.set_xlabel('Features', fontweight='bold', fontsize=12)
ax.set_ylabel('Importance', fontweight='bold', fontsize=12)
ax.set_title('Feature Importance: Random Forest vs SHAP', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(feature_cols, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance_comparison.png")
plt.close()

# Explanation
print(f"\n{'=' * 60}")
print("UNDERSTANDING SHAP PLOTS")
print(f"{'=' * 60}")
print("\nSummary Plot (shap_summary.png):")
print("  - Each dot = one patient")
print("  - Red = high feature value, Blue = low feature value")
print("  - Right side = pushes prediction toward CKD")
print("  - Left side = pushes prediction toward healthy")
print("  - Features ordered by importance (top to bottom)")

print("\nBar Plot (shap_bar.png):")
print("  - Shows average absolute impact of each feature")
print("  - Longer bar = more important feature")
print("  - Check if Serum Creatinine (sc) ranks high!")

print("\nWaterfall Plots:")
print("  - Shows how each feature contributes to the prediction")
print("  - Starting from base value (average prediction)")
print("  - Red = pushes toward CKD")
print("  - Blue = pushes toward healthy")
print("  - Arrow size = strength of contribution")

print("\nFeature Importance Comparison:")
print("  - Blue bars = Random Forest native importance")
print("  - Orange bars = SHAP-based importance")
print("  - SHAP is more accurate (based on actual predictions)")

print(f"\n{'=' * 60}")
print("SHAP ANALYSIS COMPLETE!")
print(f"{'=' * 60}")
print("Files created:")
print("  - shap_summary.png (beeswarm plot)")
print("  - shap_bar.png (mean absolute values)")
print("  - shap_waterfall_ckd.png (individual CKD explanation)")
print("  - shap_waterfall_healthy.png (individual healthy explanation)")
print("  - feature_importance_comparison.png (RF vs SHAP)")
print("\nNext: Run 5_ensemble_model.py")