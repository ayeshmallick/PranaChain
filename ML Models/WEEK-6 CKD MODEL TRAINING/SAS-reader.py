import pandas as pd
import os
import shutil

print("Fixing NHANES File Extensions and Reading")
print("=" * 60)

# Your downloaded files - they're actually SAS files with wrong extension
files_to_fix = {
    'albumin_cr': 'albumin-creatinine-data.txt',
    'biochem': 'biochemistry-profile.txt',
    'blood_count': 'blood-count-data.txt'
}

datasets = {}

for name, txt_file in files_to_fix.items():
    print(f"\n{name}:")

    # Create .xpt version
    xpt_file = txt_file.replace('.txt', '.xpt')

    # Copy file with correct extension
    if os.path.exists(txt_file):
        print(f"  Copying {txt_file} → {xpt_file}")
        shutil.copy(txt_file, xpt_file)

        # Now try reading as XPT
        try:
            df = pd.read_sas(xpt_file, format='xport')
            print(f"  ✓ Success: {len(df)} records, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)[:15]}")
            datasets[name] = df

            # Save as CSV
            csv_file = f'nhanes_{name}.csv'
            df.to_csv(csv_file, index=False)
            print(f"  ✓ Saved as: {csv_file}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print(f"  ✗ File not found: {txt_file}")

print("\n" + "=" * 60)
print(f"Successfully loaded: {len(datasets)} datasets")

if len(datasets) == 3:
    print("\n✓ All 3 files loaded! Now merging...")

    # Check for common ID column
    for name, df in datasets.items():
        print(f"\n{name} columns:")
        print(df.columns.tolist())

    # Find SEQN column (participant ID)
    if all('SEQN' in df.columns for df in datasets.values()):
        print("\n✓ All files have SEQN column - ready to merge!")

        # Merge all three
        df_merged = datasets['albumin_cr'].merge(
            datasets['biochem'], on='SEQN', how='inner'
        ).merge(
            datasets['blood_count'], on='SEQN', how='inner'
        )

        print(f"\n✓ Merged dataset: {len(df_merged)} records, {len(df_merged.columns)} columns")
        print(f"Columns: {df_merged.columns.tolist()}")

        # Save merged
        df_merged.to_csv('nhanes_merged_full.csv', index=False)
        print("\n✓ Saved as: nhanes_merged_full.csv")

        print(f"\nFirst 3 rows:")
        print(df_merged.head(3))
else:
    print("\n✗ Not all files loaded. Check the errors above.")