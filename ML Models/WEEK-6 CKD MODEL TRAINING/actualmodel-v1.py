# CKD Prediction - Complete Pipeline with Data Validation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

print("CKD Prediction Model - 40,000 Sample Dataset")
print("=" * 60)

# Load Data
print("\n1. Loading and Validating Data...")
df = pd.read_csv('ckd_40k_realistic.csv')

print(f"Initial shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data Cleaning & Validation
print("\n2. Data Cleaning and Validation...")
print("=" * 60)

# Checking for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f" Found missing values:")
    print(missing_values[missing_values > 0])

    # Handle missing values (if any exist)
    # For numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  Filled {col} with median")

    # For categorical: fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  Filled {col} with mode")
else:
    print("No missing values found")

# Check data types
print(f"\nData types:")
print(df.dtypes)

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"\nFound {duplicates} duplicate rows - removing...")
    df = df.drop_duplicates()
else:
    print("No duplicate rows")

# Validate class column
print(f"\nClass distribution:")
print(df['class'].value_counts())
valid_classes = ['ckd', 'notckd']
invalid_classes = df[~df['class'].isin(valid_classes)]
if len(invalid_classes) > 0:
    print(f"Found {len(invalid_classes)} rows with invalid classes - removing...")
    df = df[df['class'].isin(valid_classes)]
else:
    print(" All class labels are valid")

# Check for outliers
print(f"\nChecking for extreme outliers...")
numeric_cols = ['hemo', 'pcv', 'gfr', 'rbcc', 'sod', 'bp']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  {col}: {outliers} extreme outliers detected (keeping them - could be real cases)")

print(f"\n Data validation complete!")
print(f"Final dataset: {df.shape[0]} samples, {df.shape[1]} columns")

# Prepare Features
print("\n" + "=" * 60)
print("3. Preparing Features...")
print("=" * 60)

feature_cols = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp']

# Verify all features exist
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f" ERROR: Missing features: {missing_features}")
    exit()
else:
    print(f" All {len(feature_cols)} features present")

X = df[feature_cols]
y = (df['class'] == 'ckd').astype(int)

print(f"Features: {feature_cols}")
print(f"Target: 1=CKD ({sum(y == 1)}), 0=Not CKD ({sum(y == 0)})")

# Final validation
print(f"\nFinal validation:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  No missing in X: {X.isnull().sum().sum() == 0}")
print(f"  No missing in y: {y.isnull().sum() == 0}")

# Handle Class Imbalance with SMOTE
print("\n" + "=" * 60)
print("4. Handling Class Imbalance with SMOTE...")
print("=" * 60)

print(f"Before SMOTE: CKD={sum(y == 1)}, Not CKD={sum(y == 0)}")

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"After SMOTE: CKD={sum(y_balanced == 1)}, Not CKD={sum(y_balanced == 0)}")
print(" Classes balanced!")

# Train-Test Split
print("\n" + "=" * 60)
print("5. Splitting Data...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"Training set: {len(X_train)} samples ({len(X_train) / len(X_balanced) * 100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test) / len(X_balanced) * 100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features standardized")

# Train Model with Cross-Validation
print("\n" + "=" * 60)
print("6. Training Random Forest with 5-Fold Cross-Validation...")
print("=" * 60)

model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"Cross-Validation Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train on full training set
model.fit(X_train_scaled, y_train)

# Evaluate on Test Set
print("\n" + "=" * 60)
print("7. Evaluating on Test Set...")
print("=" * 60)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nFinal Results:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)