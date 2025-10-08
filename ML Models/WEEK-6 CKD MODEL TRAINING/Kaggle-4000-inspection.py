import pandas as pd
import numpy as np

print("Inspecting 4,000-Sample CKD Dataset")
print("=" * 60)

# Load the dataset
df = pd.read_csv('ckd_dataset_with_stages(4000).csv')

print(f"\nShape: {df.shape}")
print(f"Samples: 4,000")
print(f"Features: 23")

print(f"\nColumn Names:")
print(df.columns.tolist())

print(f"\nTarget Variable - ckd_pred:")
print(df['ckd_pred'].value_counts())

print(f"\nCKD Stage Distribution:")
print(df['ckd_stage'].value_counts().sort_index())

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
print(df.isnull().sum().sum())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nKey Features Present:")
features_we_want = ['gfr', 'serum_creatinine', 'bun', 'blood_pressure', 'family_history',
                     'smoking', 'alcohol', 'physical_activity']
for feat in features_we_want:
    if feat in df.columns:
        print(f"  ✓ {feat}")
    else:
        print(f"  ✗ {feat}")

print(f"\nBasic Statistics:")
print(df.describe())