import pandas as pd
import numpy as np

print("NHANES DATASET INSPECTION")
print("=" * 60)

# File 1: Albumin/Creatinine data
print("\n1. ALBUMIN/CREATININE DATA")
print("-" * 60)
try:
    # Try different separators and handle errors
    df_albumin = pd.read_csv('albumin-creatinine-data.txt', sep='\t', on_bad_lines='skip')
    print(f"Shape: {df_albumin.shape}")
    print(f"\nColumns: {df_albumin.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df_albumin.head())
    print(f"\nData types:")
    print(df_albumin.dtypes)
    print(f"\nMissing values:")
    print(df_albumin.isnull().sum())
except Exception as e:
    print(f"Tab separator failed: {e}")
    try:
        df_albumin = pd.read_csv('albumin-creatinine-data.txt', on_bad_lines='skip')
        print(f"Shape: {df_albumin.shape}")
        print(f"\nColumns: {df_albumin.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df_albumin.head())
    except Exception as e2:
        print(f"Comma separator also failed: {e2}")
        print("\nReading first 20 lines as text:")
        with open('albumin-creatinine-data.txt', 'r') as f:
            for i, line in enumerate(f):
                if i < 20:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break

# File 2: Biochemistry Profile
print("\n" + "=" * 60)
print("\n2. BIOCHEMISTRY PROFILE DATA")
print("-" * 60)
try:
    df_biochem = pd.read_csv('biochemistry-profile.txt', sep='\t', on_bad_lines='skip')
    print(f"Shape: {df_biochem.shape}")
    print(f"\nColumns: {df_biochem.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df_biochem.head())
    print(f"\nData types:")
    print(df_biochem.dtypes)
    print(f"\nMissing values:")
    print(df_biochem.isnull().sum())
except Exception as e:
    print(f"Tab separator failed: {e}")
    try:
        df_biochem = pd.read_csv('biochemistry-profile.txt', on_bad_lines='skip')
        print(f"Shape: {df_biochem.shape}")
        print(f"\nColumns: {df_biochem.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df_biochem.head())
    except Exception as e2:
        print(f"Comma separator also failed: {e2}")
        print("\nReading first 20 lines as text:")
        with open('biochemistry-profile.txt', 'r') as f:
            for i, line in enumerate(f):
                if i < 20:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break

# File 3: Blood Count Data
print("\n" + "=" * 60)
print("\n3. BLOOD COUNT DATA")
print("-" * 60)
try:
    df_blood = pd.read_csv('blood-count-data.txt', sep='\t', on_bad_lines='skip')
    print(f"Shape: {df_blood.shape}")
    print(f"\nColumns: {df_blood.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df_blood.head())
    print(f"\nData types:")
    print(df_blood.dtypes)
    print(f"\nMissing values:")
    print(df_blood.isnull().sum())
except Exception as e:
    print(f"Tab separator failed: {e}")
    try:
        df_blood = pd.read_csv('blood-count-data.txt', on_bad_lines='skip')
        print(f"Shape: {df_blood.shape}")
        print(f"\nColumns: {df_blood.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df_blood.head())
    except Exception as e2:
        print(f"Comma separator also failed: {e2}")
        print("\nReading first 20 lines as text:")
        with open('blood-count-data.txt', 'r') as f:
            for i, line in enumerate(f):
                if i < 20:
                    print(f"Line {i}: {line.strip()}")
                else:
                    break

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)