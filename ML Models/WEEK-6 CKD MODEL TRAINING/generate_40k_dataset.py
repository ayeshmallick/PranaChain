# Generate 40K Synthetic CKD Dataset
# Features: Hemo, Pcv, Sg, Grf, Rbcc, Al, Dm, Htn, Sod, Bp

import pandas as pd
import numpy as np

print("Generating 40,000 Sample CKD Dataset")
print("=" * 60)

np.random.seed(42)

n_samples = 40000
n_ckd = 25000  # 62.5% CKD
n_healthy = 15000  # 37.5% healthy

print(f"Creating {n_samples} samples...")
print(f"  CKD patients: {n_ckd}")
print(f"  Healthy: {n_healthy}")

# Generate CKD patients (abnormal values)
print("\nGenerating CKD patients...")
ckd_data = {
    'hemo': np.random.normal(9.5, 2.0, n_ckd).clip(5.0, 13.0),  # Low hemoglobin
    'pcv': np.random.normal(30, 6, n_ckd).clip(18, 40),  # Low packed cell volume
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020], n_ckd, p=[0.4, 0.4, 0.15, 0.05]),  # Low specific gravity
    'gfr': np.random.normal(35, 18, n_ckd).clip(5, 59),  # Low GFR (kidney function)
    'rbcc': np.random.normal(3.5, 0.8, n_ckd).clip(2.0, 4.5),  # Low red blood cell count
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_ckd, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1]),  # Albumin present
    'dm': np.random.choice([0, 1], n_ckd, p=[0.4, 0.6]),  # Diabetes (1=yes, 0=no)
    'htn': np.random.choice([0, 1], n_ckd, p=[0.2, 0.8]),  # Hypertension (1=yes, 0=no)
    'sod': np.random.normal(138, 5, n_ckd).clip(125, 150),  # Sodium
    'bp': np.random.normal(145, 20, n_ckd).clip(90, 200),  # Blood pressure (systolic)
    'class': ['ckd'] * n_ckd
}

# Generate Healthy patients (normal values)
print("Generating healthy patients...")
healthy_data = {
    'hemo': np.random.normal(14.5, 1.5, n_healthy).clip(12.0, 18.0),  # Normal hemoglobin
    'pcv': np.random.normal(43, 5, n_healthy).clip(36, 54),  # Normal packed cell volume
    'sg': np.random.choice([1.015, 1.020, 1.025], n_healthy, p=[0.3, 0.5, 0.2]),  # Normal specific gravity
    'gfr': np.random.normal(95, 15, n_healthy).clip(60, 120),  # Normal GFR
    'rbcc': np.random.normal(4.8, 0.5, n_healthy).clip(4.0, 6.0),  # Normal red blood cell count
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_healthy, p=[0.85, 0.1, 0.03, 0.01, 0.005, 0.005]),  # Mostly no albumin
    'dm': np.random.choice([0, 1], n_healthy, p=[0.90, 0.10]),  # Diabetes rare
    'htn': np.random.choice([0, 1], n_healthy, p=[0.85, 0.15]),  # Hypertension less common
    'sod': np.random.normal(140, 3, n_healthy).clip(135, 145),  # Normal sodium
    'bp': np.random.normal(118, 12, n_healthy).clip(90, 140),  # Normal blood pressure
    'class': ['notckd'] * n_healthy
}

# Combine datasets
print("\nCombining and shuffling...")
df_ckd = pd.DataFrame(ckd_data)
df_healthy = pd.DataFrame(healthy_data)
df = pd.concat([df_ckd, df_healthy], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✓ Dataset created successfully!")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Target: class (ckd/notckd)")

print(f"\nClass distribution:")
print(df['class'].value_counts())

print(f"\nFeature summary:")
for col in df.columns:
    if col != 'class':
        print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

# Save to CSV
filename = 'ckd_40k_dataset.csv'
df.to_csv(filename, index=False)
print(f"\n✓ Saved as: {filename}")
print(f"\n{'='*60}")
print("Dataset generation complete!")
print("Next: Run model training code on this dataset.")
print(f"{'='*60}")