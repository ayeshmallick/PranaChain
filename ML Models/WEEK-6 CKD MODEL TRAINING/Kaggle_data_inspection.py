import pandas as pd
import numpy as np

# Load the Kaggle dataset
df_kaggle = pd.read_csv('Chronic_Kidney_Dsease_data_kaggle(1700+).csv')

print("KAGGLE DATASET INSPECTION")
print("=" * 60)
print(f"\nShape: {df_kaggle.shape}")
print(f"Samples: {df_kaggle.shape[0]}, Features: {df_kaggle.shape[1]}")

print("\nColumn Names:")
print(df_kaggle.columns.tolist())

print("\nFirst 5 rows:")
print(df_kaggle.head())

print("\nData Types:")
print(df_kaggle.dtypes)

print("\nMissing Values:")
print(df_kaggle.isnull().sum())

print("\nTarget Variable (if exists):")
# Check for common CKD label column names
possible_targets = ['class', 'diagnosis', 'ckd', 'target', 'label', 'CKD', 'Diagnosis']
for col in possible_targets:
    if col in df_kaggle.columns:
        print(f"\nFound target column: '{col}'")
        print(df_kaggle[col].value_counts())
        break

print("\nBasic Statistics:")
print(df_kaggle.describe())

# Check which of your required features are present
required_features = {
    'hemoglobin': ['hemoglobin', 'hemo', 'hgb', 'Hemoglobin'],
    'pcv': ['pcv', 'hematocrit', 'PCV', 'Hematocrit'],
    'specific_gravity': ['sg', 'specific_gravity', 'urine_sg', 'Specific Gravity'],
    'gfr': ['gfr', 'egfr', 'GFR', 'eGFR', 'Glomerular Filtration Rate'],
    'rbc_count': ['rbc', 'rbcc', 'red_blood_cells', 'RBC', 'Red Blood Cells'],
    'albumin': ['albumin', 'al', 'Albumin'],
    'diabetes': ['dm', 'diabetes', 'Diabetes', 'Diabetes Mellitus'],
    'hypertension': ['htn', 'hypertension', 'Hypertension'],
    'sodium': ['sod', 'sodium', 'Sodium', 'Na'],
    'blood_pressure': ['bp', 'blood_pressure', 'systolic', 'diastolic']
}

print("\n" + "=" * 60)
print("FEATURE MATCHING:")
print("=" * 60)

for feature_type, possible_names in required_features.items():
    found = False
    for possible_name in possible_names:
        if possible_name in df_kaggle.columns:
            print(f"✓ {feature_type}: Found as '{possible_name}'")
            found = True
            break
    if not found:
        print(f"✗ {feature_type}: NOT FOUND")