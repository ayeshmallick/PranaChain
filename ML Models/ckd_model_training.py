# CKD Prediction Model - Week 5 Deliverable
# Using Cleaned Dataset (200 samples, 29 features)
# Fixed: Removed data leakage features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("CKD Prediction Model - Week 5 Deliverable")
print("Using Cleaned Dataset (200 samples, 29 features)")
print("=" * 60)

# STEP 1: Load the cleaned dataset
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('ckd_dataset_clean.csv')

print(f"Initial shape: {df.shape}")
print(f"Class distribution:")
print(df['class'].value_counts())

# STEP 2: Data Preprocessing
print("\n" + "=" * 60)
print("2. Preprocessing Data...")
print("=" * 60)


# Universal conversion function for handling all edge cases
def convert_to_numeric(value_str, default=0):
    value_str = str(value_str).strip()

    # Excel month conversions
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}
    for month_name, month_num in month_map.items():
        if month_name in value_str:
            try:
                first_num = int(value_str.split('-')[0])
                return (first_num + month_num) / 2
            except:
                return default

    # Handle comparison operators
    if '>=' in value_str:
        return float(value_str.replace('>=', '').strip())
    elif '<=' in value_str:
        return float(value_str.replace('<=', '').strip())
    elif '>' in value_str:
        return float(value_str.replace('>', '').strip())
    elif '<' in value_str:
        return float(value_str.replace('<', '').strip())

    # Handle ranges
    if '-' in value_str and not any(m in value_str for m in month_map.keys()):
        try:
            parts = value_str.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return default

    # Try direct conversion
    try:
        return float(value_str)
    except:
        return default


# Convert age
df['age_numeric'] = df['age'].apply(lambda x: convert_to_numeric(x, default=40))

# Convert specific gravity
df['sg_numeric'] = df['sg'].apply(lambda x: convert_to_numeric(x, default=1.015))

# Convert albumin
df['al_numeric'] = df['al'].apply(lambda x: convert_to_numeric(x, default=0))

# Convert sugar
df['su_numeric'] = df['su'].apply(lambda x: convert_to_numeric(x, default=0))

# Convert other object columns to numeric
numeric_convert_cols = ['bgr', 'bu', 'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc']

for col in numeric_convert_cols:
    df[col] = df[col].apply(lambda x: convert_to_numeric(x, default=np.nan))

# Handle grf
df['grf_numeric'] = df['grf'].apply(lambda x: convert_to_numeric(x, default=np.nan))

# Fill remaining missing values with median
numeric_cols = ['age_numeric', 'sg_numeric', 'al_numeric', 'su_numeric', 'bgr', 'bu',
                'sod', 'sc', 'pot', 'hemo', 'pcv', 'rbcc', 'wbcc', 'grf_numeric']

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} missing values with median: {median_val:.2f}")

print(f"\nPreprocessing complete!")
print(f"Missing values after cleaning: {df[numeric_cols].isnull().sum().sum()}")

# STEP 3: Select features for modeling (REMOVED DATA LEAKAGE)
print("\n" + "=" * 60)
print("3. Selecting Features for Modeling...")
print("=" * 60)

# CRITICAL: Remove 'affected' and 'stage_numeric' - these cause data leakage!
# 'affected' directly indicates CKD status (circular reasoning)
# 'stage_numeric' is determined AFTER diagnosis, not before

feature_cols = ['age_numeric', 'bp (Diastolic)', 'bp limit', 'sg_numeric', 'al_numeric', 'su_numeric',
                'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sod', 'sc', 'pot',
                'hemo', 'pcv', 'rbcc', 'wbcc', 'htn', 'dm', 'cad', 'appet',
                'pe', 'ane', 'grf_numeric']  # 26 features, NO leakage

print(f"\n⚠️  REMOVED DATA LEAKAGE FEATURES:")
print("  ✗ 'affected' - directly indicates CKD status (circular reasoning)")
print("  ✗ 'stage_numeric' - determined after diagnosis, not available for prediction")
print(f"\nUsing {len(feature_cols)} legitimate predictive features")

X = df[feature_cols]
y = df['class']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Target classes: {le.classes_}")
print(f"Class distribution: CKD={sum(y_encoded == 0)}, Not CKD={sum(y_encoded == 1)}")

# STEP 4: Split and Scale Data
print("\n" + "=" * 60)
print("4. Splitting and Scaling Data...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(df) * 100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(df) * 100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# STEP 5: Train Models
print("\n" + "=" * 60)
print("5. Training Machine Learning Models...")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

results = {}

for name, model in models.items():
    print(f"\n{'=' * 40}")
    print(f"Training: {name}")
    print(f"{'=' * 40}")

    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'predictions': y_pred_test
    }

    print(f"\nResults:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))

# STEP 6: Visualizations
print("\n" + "=" * 60)
print("6. Creating Visualizations...")
print("=" * 60)

fig = plt.figure(figsize=(16, 10))
fig.suptitle('CKD Prediction - Week 5 Results (No Data Leakage)',
             fontsize=16, fontweight='bold')

# Plot 1: Model Comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
test_accs = [results[n]['test_accuracy'] for n in model_names]
train_accs = [results[n]['train_accuracy'] for n in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = ax1.bar(x - width / 2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width / 2, test_accs, width, label='Test', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0.85, 1.0])
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Confusion Matrix
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_preds = results[best_model_name]['predictions']

ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax2.set_title(f'Confusion Matrix\n{best_model_name}', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')

# Plot 3: Class Distribution
ax3 = plt.subplot(2, 3, 3)
class_counts = df['class'].value_counts()
colors = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax3.pie(class_counts, labels=class_counts.index,
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('Class Distribution\n(200 samples)', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Plot 4: Feature Importance
ax4 = plt.subplot(2, 1, 2)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax4.barh(range(len(feature_importance)), feature_importance['importance'], color=colors_grad)
ax4.set_yticks(range(len(feature_importance)))
ax4.set_yticklabels(feature_importance['feature'])
ax4.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax4.set_title('Top 15 Most Important Features (Random Forest)', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(feature_importance.iterrows()):
    ax4.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig('ckd_model_results_final_week5.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'ckd_model_results_final_week5.png'")
plt.show()

# STEP 7: Summary Report
print("\n" + "=" * 60)
print("WEEK 5 SUMMARY")
print("=" * 60)

print(f"\nProject: Chronic Kidney Disease Prediction")
print(f"Dataset: 200 patient records")
print(f"Features: 26 medical indicators")
print(f"Class Split: 128 CKD patients, 72 healthy patients")

print(f"\n{'Model Results':-^60}")
print(f"{'Model':<25} {'Training':<15} {'Testing':<15}")
print("-" * 60)
for name in results:
    train_acc = results[name]['train_accuracy']
    test_acc = results[name]['test_accuracy']
    print(f"{name:<25} {train_acc*100:.1f}%          {test_acc*100:.1f}%")

print(f"\n{'Most Important Health Indicators':-^60}")
for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    feature_name = row['feature'].replace('_numeric', '').replace('_', ' ').title()
    print(f"{i:2d}. {feature_name:<25} ({row['importance']*100:.1f}%)")