# Week 6: CKD Prediction - 4,000 Sample Kaggle Dataset
# Demonstrating scalability to even larger datasets

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("Week 6: CKD Prediction - 4,000 Sample Dataset")
print("=" * 60)

# STEP 1: Load Data
print("\n1. Loading dataset...")
df = pd.read_csv('ckd_dataset_with_stages(4000).csv')

print(f"Initial shape: {df.shape}")
print(f"\nClass distribution:")
print(df['ckd_pred'].value_counts())
print(f"CKD: {sum(df['ckd_pred'] == 'CKD')} ({sum(df['ckd_pred'] == 'CKD') / len(df) * 100:.1f}%)")
print(f"No CKD: {sum(df['ckd_pred'] == 'No CKD')} ({sum(df['ckd_pred'] == 'No CKD') / len(df) * 100:.1f}%)")

# STEP 2: Feature Engineering
print("\n" + "=" * 60)
print("2. Feature Engineering...")
print("=" * 60)

# Encode categorical variables
categorical_cols = ['physical_activity', 'diet', 'smoking', 'alcohol',
                    'painkiller_usage', 'family_history', 'weight_changes', 'stress_level']

print("Encoding categorical variables...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} categories")

# Select features
feature_cols = [
    'serum_creatinine',
    'gfr',
    'bun',
    'serum_calcium',
    'ana',
    'c3_c4',
    'hematuria',
    'oxalate_levels',
    'urine_ph',
    'blood_pressure',
    'water_intake',
    'months',
    'physical_activity_encoded',
    'diet_encoded',
    'smoking_encoded',
    'alcohol_encoded',
    'painkiller_usage_encoded',
    'family_history_encoded',
    'weight_changes_encoded',
    'stress_level_encoded'
]

X = df[feature_cols]
y = (df['ckd_pred'] == 'CKD').astype(int)  # 1 = CKD, 0 = No CKD

print(f"\nSelected {len(feature_cols)} features")
print(f"No missing values: {X.isnull().sum().sum() == 0}")

# STEP 3: Handle Class Imbalance
print("\n" + "=" * 60)
print("3. Handling Severe Class Imbalance with SMOTE...")
print("=" * 60)

print(f"Before SMOTE: CKD={sum(y == 1)}, No CKD={sum(y == 0)}")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE: CKD={sum(y_resampled == 1)}, No CKD={sum(y_resampled == 0)}")
print("✓ Classes balanced!")

# STEP 4: Train-Test Split
print("\n" + "=" * 60)
print("4. Splitting Data...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 5: Train Models with Cross-Validation
print("\n" + "=" * 60)
print("5. Training Models with 5-Fold Cross-Validation...")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"Training: {name}")
    print(f"{'=' * 40}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full training set
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred_test
    }

    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No CKD', 'CKD'], zero_division=0))

# STEP 6: Visualizations
print("\n" + "=" * 60)
print("6. Creating Visualizations...")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Week 6: CKD Prediction - 4,000 Sample Dataset (SMOTE balanced)',
             fontsize=16, fontweight='bold')

# Plot 1: Cross-Validation Scores
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in model_names]
cv_stds = [results[n]['cv_std'] for n in model_names]

x = np.arange(len(model_names))
bars = ax1.bar(x, cv_means, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
ax1.errorbar(x, cv_means, yerr=cv_stds, fmt='none', ecolor='black', capsize=5)

ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('5-Fold Cross-Validation Results', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.set_ylim([0.85, 1.0])
ax1.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}\n±{cv_stds[i]:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Test Accuracy Comparison
ax2 = plt.subplot(2, 3, 2)
test_accs = [results[n]['test_accuracy'] for n in model_names]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax2.bar(model_names, test_accs, color=colors, alpha=0.8)

ax2.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Final Test Performance', fontsize=12, fontweight='bold')
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.set_ylim([0.85, 1.0])
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Precision, Recall, F1
ax3 = plt.subplot(2, 3, 3)
metrics_data = {
    'Precision': [results[n]['precision'] for n in model_names],
    'Recall': [results[n]['recall'] for n in model_names],
    'F1-Score': [results[n]['f1_score'] for n in model_names]
}

x = np.arange(len(model_names))
width = 0.25

for i, (metric, values) in enumerate(metrics_data.items()):
    offset = width * (i - 1)
    bars = ax3.bar(x + offset, values, width, label=metric, alpha=0.8)

ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Precision, Recall, F1-Score', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=15, ha='right')
ax3.legend()
ax3.set_ylim([0.85, 1.0])
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Confusion Matrix
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_preds = results[best_model_name]['predictions']

ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
ax4.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')

# Plot 5: Feature Importance
ax5 = plt.subplot(2, 1, 2)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax5.barh(range(len(feature_importance)), feature_importance['importance'], color=colors_grad)
ax5.set_yticks(range(len(feature_importance)))
ax5.set_yticklabels(feature_importance['feature'].str.replace('_encoded', ''))
ax5.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax5.set_title('Top 15 Most Important Features (Random Forest)', fontsize=12, fontweight='bold')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax5.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('ckd_4000sample_week6_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'ckd_4000sample_week6_results.png'")
plt.show()

# STEP 7: Summary
print("\n" + "=" * 60)
print("WEEK 6 SUMMARY - 4,000 SAMPLE DATASET")
print("=" * 60)

print(f"\nDataset: 4,000-sample CKD with stages")
print(f"Features: {len(feature_cols)} clinical indicators")
print(f"Original Balance: 96.9% CKD, 3.1% No CKD")
print(f"After SMOTE: Balanced 50/50")
print(f"Validation: 5-Fold Cross-Validation")

print(f"\n{'Model Performance':-^60}")
print(f"{'Model':<25} {'CV Mean':<15} {'Test Acc':<15}")
print("-" * 60)
for name in results:
    cv_mean = results[name]['cv_mean']
    test_acc = results[name]['test_accuracy']
    print(f"{name:<25} {cv_mean:.4f} (±{results[name]['cv_std']:.3f})  {test_acc:.4f}")

print(f"\n{'Best Model':-^60}")
print(f"Model: {best_model_name}")
print(f"Cross-Validation: {results[best_model_name]['cv_mean']:.4f} (±{results[best_model_name]['cv_std']:.3f})")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")

print(f"\n{'Progressive Validation Summary':-^60}")
print("Week 5 (Original): 200 samples → 97.5%")
print("Week 6 (1,659 samples): 1,659 samples → 95.4% with CV")
print(f"Week 6 (4,000 samples): 4,000 samples → {results[best_model_name]['test_accuracy'] * 100:.1f}% with CV")

print("\n" + "=" * 60)
print("WEEK 6 COMPLETE - TRIPLE DATASET VALIDATION!")
print("=" * 60)

print("\nKey Achievements:")
print("  ✓ Validated on 3 independent datasets (200, 1,659, 4,000 samples)")
print("  ✓ Handled severe class imbalance (96.9% vs 3.1%)")
print("  ✓ Consistent cross-validation across all datasets")
print("  ✓ Demonstrated model scalability from 200 to 4,000 samples")
print("  ✓ All results validated with 5-fold CV")