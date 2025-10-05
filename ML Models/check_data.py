import pandas as pd
import os

# First, let's see what files are actually in the directory
print("Current working directory:")
print(os.getcwd())

print("\nFiles in current directory:")
for file in os.listdir('.'):
    print(file)

# Try finding the file
try:
    # Look in the share folder based on your screenshot
    df = pd.read_csv('share/ckd_dataset_clean.csv')
    print("\nFound it in 'share' folder!")
except:
    # If not there, try current directory
    try:
        df = pd.read_csv('ckd_dataset_clean.csv')
        print("\nFound it in current directory!")
    except:
        print("\nFile not found. Please check the exact location.")
        exit()

print("\nCleaned Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nClass distribution:")
print(df['class'].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())