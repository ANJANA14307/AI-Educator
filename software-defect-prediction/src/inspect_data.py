
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/raw/apachejit_train.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 1. Number of rows and columns
print("Dataset shape:", df.shape)

# 2. Column names
print("\nColumn names:")
print(df.columns.tolist())

# 3. Data types
print("\nData types:")
print(df.dtypes)

# 4. First five records
print("\nFirst five records:")
print(df.head())

# 5. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 6. Missing value percentages
print("\nMissing value percentages:")
print((df.isnull().mean() * 100).round(2))

# 7. Duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# 8. Target distribution
print("\nTarget distribution:")
if "buggy" in df.columns:
    print(df["buggy"].value_counts(dropna=False))

    print("\nTarget percentages:")
    print(df["buggy"].value_counts(normalize=True) * 100)
else:
    print("Target column 'buggy' not found. Check the schema.")

# 9. Statistical summary
print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\n========== INSPECTION COMPLETE ==========")
