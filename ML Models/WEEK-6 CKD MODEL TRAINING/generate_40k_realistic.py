# Generate 40K REALISTIC Synthetic CKD Dataset
# Target: 95-98% accuracy (not perfect 100%)
# Features: Hemo, Pcv, Sg, Grf, Rbcc, Al, Dm, Htn, Sod, Bp

import pandas as pd
import numpy as np

print("Generating 40,000 Sample REALISTIC CKD Dataset")
print("=" * 60)

np.random.seed(42)

n_samples = 40000
n_ckd = 25000  # 62.5% CKD
n_healthy = 15000  # 37.5% healthy

print(f"Creating {n_samples} samples with realistic overlap...")
print(f"  CKD patients: {n_ckd}")
print(f"  Healthy: {n_healthy}")

# Generate CKD patients (abnormal values with MORE OVERLAP)
print("\nGenerating CKD patients with realistic variability...")
ckd_data = {
    # Lower hemoglobin but with overlap into normal range (some CKD patients are not anemic)
    'hemo': np.random.normal(10.5, 2.5, n_ckd).clip(5.0, 16.0),  # More variance, some reach normal

    # Lower PCV with overlap
    'pcv': np.random.normal(33, 8, n_ckd).clip(18, 46),  # Some reach normal range

    # Lower specific gravity with realistic distribution
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_ckd,
                           p=[0.25, 0.35, 0.25, 0.10, 0.05]),  # Some normal values

    # Lower GFR but with early-stage patients having near-normal values
    'gfr': np.random.normal(45, 22, n_ckd).clip(5, 75),  # Some overlap with normal

    # Lower RBC with overlap
    'rbcc': np.random.normal(3.8, 1.0, n_ckd).clip(2.0, 5.2),  # Some reach normal

    # Albumin - mix of present and absent (not all CKD has albuminuria)
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_ckd,
                           p=[0.20, 0.20, 0.20, 0.20, 0.10, 0.10]),  # More variable

    # Diabetes - not all CKD patients have diabetes
    'dm': np.random.choice([0, 1], n_ckd, p=[0.50, 0.50]),  # 50/50 split

    # Hypertension - common but not universal
    'htn': np.random.choice([0, 1], n_ckd, p=[0.30, 0.70]),  # 70% have HTN

    # Sodium - mostly normal even in CKD, some abnormal
    'sod': np.random.normal(139, 6, n_ckd).clip(125, 150),  # Wide range

    # Blood pressure - higher but with overlap
    'bp': np.random.normal(138, 22, n_ckd).clip(90, 190),  # Some normal BP

    'class': ['ckd'] * n_ckd
}

# Generate Healthy patients (normal values with SOME OVERLAP into abnormal)
print("Generating healthy patients with realistic variability...")
healthy_data = {
    # Normal hemoglobin but some low-normal (early anemia, iron deficiency, etc.)
    'hemo': np.random.normal(14.0, 2.0, n_healthy).clip(10.0, 18.0),  # Some drop into CKD range

    # Normal PCV with some overlap
    'pcv': np.random.normal(42, 7, n_healthy).clip(30, 54),  # Some lower values

    # Normal specific gravity with variation
    'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_healthy,
                           p=[0.05, 0.15, 0.30, 0.35, 0.15]),  # Some low values

    # Normal GFR but some borderline (Stage 2 CKD, often asymptomatic)
    'gfr': np.random.normal(92, 18, n_healthy).clip(55, 120),  # Some borderline low

    # Normal RBC with some lower values
    'rbcc': np.random.normal(4.7, 0.7, n_healthy).clip(3.5, 6.0),  # Some lower

    # Albumin - mostly absent but some trace amounts (UTI, dehydration)
    'al': np.random.choice([0, 1, 2, 3, 4, 5], n_healthy,
                           p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01]),  # Some present

    # Diabetes - less common but some have it
    'dm': np.random.choice([0, 1], n_healthy, p=[0.85, 0.15]),  # 15% have diabetes

    # Hypertension - less common
    'htn': np.random.choice([0, 1], n_healthy, p=[0.75, 0.25]),  # 25% have HTN

    # Sodium - normal with some variation
    'sod': np.random.normal(140, 4, n_healthy).clip(132, 148),  # Tighter range but some overlap

    # Blood pressure - normal but some elevated (pre-hypertension)
    'bp': np.random.normal(120, 15, n_healthy).clip(95, 155),  # Some reach CKD range

    'class': ['notckd'] * n_healthy
}

# Add noise and borderline cases (simulate measurement error and disease progression)
print("\nAdding realistic noise and borderline cases...")


def add_noise(data_dict, noise_level=0.05):
    """Add random noise to simulate measurement variability"""
    for key in ['hemo', 'pcv', 'gfr', 'rbcc', 'sod', 'bp']:
        if key in data_dict:
            noise = np.random.normal(0, noise_level * np.std(data_dict[key]), len(data_dict[key]))
            data_dict[key] = data_dict[key] + noise
    return data_dict


ckd_data = add_noise(ckd_data, noise_level=0.08)
healthy_data = add_noise(healthy_data, noise_level=0.06)

# Combine datasets
print("Combining and shuffling...")
df_ckd = pd.DataFrame(ckd_data)
df_healthy = pd.DataFrame(healthy_data)
df = pd.concat([df_ckd, df_healthy], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add some mislabeled cases (simulate diagnostic uncertainty ~1-2%)
print("Adding realistic diagnostic uncertainty (1.5% mislabeled)...")
n_mislabel = int(len(df) * 0.015)
mislabel_indices = np.random.choice(df.index, n_mislabel, replace=False)
df.loc[mislabel_indices, 'class'] = df.loc[mislabel_indices, 'class'].apply(
    lambda x: 'notckd' if x == 'ckd' else 'ckd'
)

print(f"\n✓ Dataset created with realistic complexity!")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Mislabeled cases: {n_mislabel} ({n_mislabel / len(df) * 100:.1f}%)")
print(f"  Target: class (ckd/notckd)")

print(f"\nClass distribution:")
print(df['class'].value_counts())

print(f"\nFeature summary (showing overlap):")
for col in ['hemo', 'pcv', 'gfr', 'rbcc']:
    if col in df.columns:
        ckd_vals = df[df['class'] == 'ckd'][col]
        healthy_vals = df[df['class'] == 'notckd'][col]
        print(f"\n  {col}:")
        print(
            f"    CKD: mean={ckd_vals.mean():.2f}, std={ckd_vals.std():.2f}, range=[{ckd_vals.min():.2f}, {ckd_vals.max():.2f}]")
        print(
            f"    Healthy: mean={healthy_vals.mean():.2f}, std={healthy_vals.std():.2f}, range=[{healthy_vals.min():.2f}, {healthy_vals.max():.2f}]")

# Check overlap
print(f"\nFeature Overlap Analysis:")
for col in ['hemo', 'gfr', 'pcv']:
    ckd_min = df[df['class'] == 'ckd'][col].min()
    ckd_max = df[df['class'] == 'ckd'][col].max()
    healthy_min = df[df['class'] == 'notckd'][col].min()
    healthy_max = df[df['class'] == 'notckd'][col].max()

    overlap_start = max(ckd_min, healthy_min)
    overlap_end = min(ckd_max, healthy_max)
    overlap_range = overlap_end - overlap_start

    print(f"  {col}: Overlap range = [{overlap_start:.2f}, {overlap_end:.2f}] (range: {overlap_range:.2f})")

# Save to CSV
filename = 'ckd_40k_realistic.csv'
df.to_csv(filename, index=False)
print(f"\n✓ Saved as: {filename}")
print(f"\n{'=' * 60}")
print("REALISTIC dataset generation complete!")
print("Expected model accuracy: 95-98%")
print("Next: Run train_40k_models.py on ckd_40k_realistic.csv")
print(f"{'=' * 60}")