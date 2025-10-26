"""
Week 8 - Create All Visualizations
Filename: 6_create_visualizations.py
Input: ckd_40k_with_creatinine.csv
Output: Comparison charts and plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("WEEK 8: Creating Visualizations")
print("="*60)

# Load and prepare data
df = pd.read_csv('ckd_40k_with_creatinine.csv')
feature_cols = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']
X = df[feature_cols]
y = (df['class'] == 'ckd').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Week 8: CKD Prediction with Serum Creatinine', fontsize=18, fontweight='bold', y=0.995)

# 1. Feature Importance
ax1 = plt.subplot(2, 3, 1)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

colors = ['red' if f == 'sc' else 'steelblue' for f in feature_importance['feature']]
bars = ax1.barh(range(len(feature_importance)), feature_importance['importance'], color=colors, alpha=0.8)
ax1.set_yticks(range(len(feature_importance)))
ax1.set_yticklabels(feature_importance['feature'])
ax1.set_xlabel('Importance', fontweight='bold')
ax1.set_title('Feature Importance\n(Red = NEW Serum Creatinine)', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 2. Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=True,
            xticklabels=['Healthy', 'CKD'], yticklabels=['Healthy', 'CKD'])
ax2.set_title('Confusion Matrix', fontweight='bold')
ax2.set_ylabel('Actual', fontweight='bold')
ax2.set_xlabel('Predicted', fontweight='bold')

# 3. Week 6 vs Week 8 Comparison
ax3 = plt.subplot(2, 3, 3)
weeks = ['Week 6\n(10 features)', 'Week 8\n(11 features)']
accuracies = [97.6, (y_pred == y_test).mean() * 100]
bars = ax3.bar(weeks, accuracies, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_title('Performance Improvement', fontweight='bold')
ax3.set_ylim([95, 100])
ax3.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 4. Creatinine Distribution
ax4 = plt.subplot(2, 3, 4)
ckd_sc = df[df['class'] == 'ckd']['sc']
healthy_sc = df[df['class'] == 'notckd']['sc']

ax4.hist(healthy_sc, bins=50, alpha=0.6, label='Healthy', color='green', edgecolor='black')
ax4.hist(ckd_sc, bins=50, alpha=0.6, label='CKD', color='red', edgecolor='black')
ax4.axvline(1.3, color='orange', linestyle='--', linewidth=2, label='Normal Upper Limit')
ax4.set_xlabel('Serum Creatinine (mg/dL)', fontweight='bold')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Serum Creatinine Distribution', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Model Metrics Comparison
ax5 = plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]
bars = ax5.bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8, edgecolor='black')
ax5.set_ylabel('Score', fontweight='bold')
ax5.set_title('All Performance Metrics', fontweight='bold')
ax5.set_ylim([0.95, 1.0])
ax5.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Top 5 Features Comparison
ax6 = plt.subplot(2, 3, 6)
top5 = feature_importance.head(5)
colors_top5 = ['red' if f == 'sc' else 'steelblue' for f in top5['feature']]
bars = ax6.barh(range(len(top5)), top5['importance'], color=colors_top5, alpha=0.8, edgecolor='black')
ax6.set_yticks(range(len(top5)))
ax6.set_yticklabels(top5['feature'])
ax6.set_xlabel('Importance', fontweight='bold')
ax6.set_title('Top 5 Most Important Features', fontweight='bold')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top5.iterrows()):
    ax6.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=10)

plt.tight_layout()
plt.savefig('week8_complete_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: week8_complete_analysis.png")
plt.show()

print(f"\n{'='*60}")
print("ALL VISUALIZATIONS COMPLETE!")
print(f"{'='*60}")
print("Files created:")
print("  - week8_complete_analysis.png")
print("\nWeek 8 analysis finished!")