# CKD Prediction: Training on 40K Dataset
# Features: Hemo, Pcv, Sg, Grf, Rbcc, Al, Dm, Htn, Sod, Bp

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("CKD Prediction Model - 40,000 Sample Dataset")
print("=" * 60)

# STEP 1: Load Data
print("\n1. Loading dataset...")
df = pd.read_csv('ckd_40k_dataset.csv')

print(f"Dataset loaded: {df.shape}")
print(f"Samples: {len(df)}")
print(f"Features: {len(df.columns) - 1}")

print(f"\nClass distribution:")
class_counts = df['class'].value_counts()
print(class_counts)
for cls, count in class_counts.items():
    print(f"  {cls}: {count} ({count / len(df) * 100:.1f}%)")

# STEP 2: Prepare Features
print("\n" + "=" * 60)
print("2. Preparing Features...")
print("=" * 60)

feature_cols = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp']
X = df[feature_cols]
y = (df['class'] == 'ckd').astype(int)  # 1=CKD, 0=notckd

print(f"Features: {feature_cols}")
print(f"Target: 1=CKD ({sum(y == 1)}), 0=Not CKD ({sum(y == 0)})")
print(f"No missing values: {X.isnull().sum().sum() == 0}")

# STEP 3: Train-Test Split
print("\n" + "=" * 60)
print("3. Splitting Data...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples ({len(X_train) / len(df) * 100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test) / len(df) * 100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized")

# STEP 4: Train Models with Cross-Validation
print("\n" + "=" * 60)
print("4. Training Models with 5-Fold Cross-Validation...")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"Training: {name}")
    print(f"{'=' * 40}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Train on full training set
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred_test
    }

    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"  Testing Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Not CKD', 'CKD'], zero_division=0))

# STEP 5: Visualizations
print("\n" + "=" * 60)
print("5. Creating Visualizations...")
print("=" * 60)

fig = plt.figure(figsize=(18, 10))
fig.suptitle('CKD Prediction - 40,000 Sample Dataset Results', fontsize=16, fontweight='bold')

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
ax1.set_ylim([0.92, 1.0])
ax1.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}\n±{cv_stds[i]:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Test Accuracy Comparison
ax2 = plt.subplot(2, 3, 2)
test_accs = [results[n]['test_acc'] for n in model_names]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax2.bar(model_names, test_accs, color=colors, alpha=0.8)

ax2.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Final Test Performance', fontsize=12, fontweight='bold')
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.set_ylim([0.92, 1.0])
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
    'F1-Score': [results[n]['f1'] for n in model_names]
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
ax3.set_ylim([0.92, 1.0])
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Confusion Matrix
best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_preds = results[best_model_name]['predictions']

ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
ax4.set_title(f'Confusion Matrix\n{best_model_name}', fontsize=12, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')

# Plot 5: Class Distribution
ax5 = plt.subplot(2, 3, 5)
colors_pie = ['#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax5.pie(class_counts, labels=class_counts.index,
                                   autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax5.set_title('Dataset Class Distribution\n(40,000 samples)', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Plot 6: Feature Importance
ax6 = plt.subplot(2, 3, 6)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax6.barh(range(len(feature_importance)), feature_importance['importance'], color=colors_grad)
ax6.set_yticks(range(len(feature_importance)))
ax6.set_yticklabels(feature_importance['feature'])
ax6.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax6.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax6.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('ckd_40k_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: ckd_40k_results.png")
plt.show()

# STEP 6: Final Summary
print("\n" + "=" * 60)
print("FINAL SUMMARY - 40K DATASET")
print("=" * 60)

print(f"\nDataset: 40,000 CKD patient records")
print(f"Features: 10 clinical indicators (Hemo, PCV, SG, GFR, RBC, Al, DM, HTN, Sod, BP)")
print(f"Class Balance: {class_counts['ckd']} CKD, {class_counts['notckd']} Not CKD")
print(f"Validation: 5-Fold Cross-Validation")

print(f"\n{'Model Performance':-^60}")
print(f"{'Model':<25} {'CV Accuracy':<15} {'Test Accuracy':<15}")
print("-" * 60)
for name in results:
    cv_mean = results[name]['cv_mean']
    test_acc = results[name]['test_acc']
    print(f"{name:<25} {cv_mean:.4f} (±{results[name]['cv_std']:.3f})  {test_acc:.4f}")

print(f"\n{'Best Model: {best_model_name}':-^60}")
print(f"Cross-Validation: {results[best_model_name]['cv_mean']:.4f} (±{results[best_model_name]['cv_std']:.3f})")
print(f"Test Accuracy: {results[best_model_name]['test_acc']:.4f} ({results[best_model_name]['test_acc'] * 100:.2f}%)")
print(f"Precision: {results[best_model_name]['precision']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")

print(f"\n{'Top Features (Random Forest)':-^60}")
for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<10} : {row['importance']:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nFiles generated:")
print("  - ckd_40k_results.png (visualization)")
print("\nReady for presentation!")