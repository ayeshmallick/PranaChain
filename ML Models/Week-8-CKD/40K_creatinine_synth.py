# Generate 40K Dataset WITH Serum Creatinine
# Features: Hemo, Pcv, Sg, Grf, Rbcc, Al, Dm, Htn, Sod, Bp, CREATININE (NEW!)

import pandas as pd
import numpy as np

print("Generating 40K Dataset with Serum Creatinine")
print("=" * 60)

np.random.seed(42)

n_ckd = 25000
n_healthy = 15000

# CKD patients
ckd_data = {
    'hemo': np.random.normal(10.5, 2.5, n_ckd).clip(5.0, 16.0),
    'pcv': np.random.normal(33, 8, n_ckd).clip(18, 46),
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_ckd,
                           p=[0.25, 0.35, 0.25, 0.10, 0.05]),
    'gfr': np.random.normal(45, 22, n_ckd).clip(5, 75),
    'rbcc': np.random.normal(3.8, 1.0, n_ckd).clip(2.0, 5.2),
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_ckd,
                           p=[0.20, 0.20, 0.20, 0.20, 0.10, 0.10]),
    'dm': np.random.choice([0, 1], n_ckd, p=[0.50, 0.50]),
    'htn': np.random.choice([0, 1], n_ckd, p=[0.30, 0.70]),
    'sod': np.random.normal(139, 6, n_ckd).clip(125, 150),
    'bp': np.random.normal(138, 22, n_ckd).clip(90, 190),

    # NEW: Serum Creatinine - HIGH in CKD patients
    'sc': np.random.normal(2.8, 1.2, n_ckd).clip(1.2, 8.0),  # Elevated creatinine

    'class': ['ckd'] * n_ckd
}

# Healthy patients
healthy_data = {
    'hemo': np.random.normal(14.0, 2.0, n_healthy).clip(10.0, 18.0),
    'pcv': np.random.normal(42, 7, n_healthy).clip(30, 54),
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_healthy,
                           p=[0.05, 0.15, 0.30, 0.35, 0.15]),
    'gfr': np.random.normal(92, 18, n_healthy).clip(55, 120),
    'rbcc': np.random.normal(4.7, 0.7, n_healthy).clip(3.5, 6.0),
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_healthy,
                           p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01]),
    'dm': np.random.choice([0, 1], n_healthy, p=[0.85, 0.15]),
    'htn': np.random.choice([0, 1], n_healthy, p=[0.75, 0.25]),
    'sod': np.random.normal(140, 4, n_healthy).clip(132, 148),
    'bp': np.random.normal(120, 15, n_healthy).clip(95, 155),

    # NEW: Serum Creatinine - NORMAL in healthy patients
    'sc': np.random.normal(0.95, 0.25, n_healthy).clip(0.5, 1.4),  # Normal creatinine

    'class': ['notckd'] * n_healthy
}


# Add noise
def add_noise(data_dict, noise_level=0.05):
    for key in ['hemo', 'pcv', 'gfr', 'rbcc', 'sod', 'bp', 'sc']:  # Added 'sc'
        if key in data_dict:
            noise = np.random.normal(0, noise_level * np.std(data_dict[key]), len(data_dict[key]))
            data_dict[key] = data_dict[key] + noise
    return data_dict


ckd_data = add_noise(ckd_data, noise_level=0.08)
healthy_data = add_noise(healthy_data, noise_level=0.06)

# Combine
df_ckd = pd.DataFrame(ckd_data)
df_healthy = pd.DataFrame(healthy_data)
df = pd.concat([df_ckd, df_healthy], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add mislabeled cases
n_mislabel = int(len(df) * 0.015)
mislabel_indices = np.random.choice(df.index, n_mislabel, replace=False)
df.loc[mislabel_indices, 'class'] = df.loc[mislabel_indices, 'class'].apply(
    lambda x: 'notckd' if x == 'ckd' else 'ckd'
)

print(f"\n✓ Dataset created with 11 features (including Serum Creatinine)!")
print(f"  Total: {len(df)} samples")
print(f"\nSerum Creatinine ranges:")
print(f"  CKD patients: {df[df['class'] == 'ckd']['sc'].mean():.2f} ± {df[df['class'] == 'ckd']['sc'].std():.2f} mg/dL")
print(
    f"  Healthy: {df[df['class'] == 'notckd']['sc'].mean():.2f} ± {df[df['class'] == 'notckd']['sc'].std():.2f} mg/dL")

# Save
df.to_csv('ckd_40k_with_creatinine.csv', index=False)
print(f"\n✓ Saved as: ckd_40k_with_creatinine.csv")