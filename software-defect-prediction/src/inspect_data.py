
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
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_raw_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. "
            "Place apachejit_train.csv inside data/raw/."
        )
    return pd.read_csv(path)


def inspect_data(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    missing = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })

    target_distribution = None
    if "buggy" in df.columns:
        target_distribution = df["buggy"].value_counts(dropna=False).rename("count").to_frame()
        target_distribution["percent"] = (
            df["buggy"].value_counts(normalize=True, dropna=False).mul(100).round(2)
        )

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "head": df.head(),
        "missing": missing,
        "duplicates": int(df.duplicated().sum()),
        "numerical_summary": df.describe(include="number").T,
        "target_distribution": target_distribution,
    }


def save_results(df, results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results["missing"].to_csv(REPORTS_DIR / "missing_values.csv")
    results["numerical_summary"].to_csv(REPORTS_DIR / "numerical_summary.csv")
    if results["target_distribution"] is not None:
        results["target_distribution"].to_csv(REPORTS_DIR / "target_distribution.csv")

    output = [
        "ApacheJIT Dataset Inspection",
        "============================",
        f"Rows: {results['shape'][0]}",
        f"Columns: {results['shape'][1]}",
        "\nColumn names:",
        ", ".join(df.columns),
        "\nColumn data types:",
        results["dtypes"].to_string(),
        "\nFirst five records:",
        results["head"].to_string(),
        "\nMissing values:",
        results["missing"].to_string(),
        f"\nDuplicate records: {results['duplicates']}",
        "\nNumerical summary:",
        results["numerical_summary"].to_string(),
    ]
    if results["target_distribution"] is not None:
        output.extend([
            "\nTarget distribution (`buggy`):",
            results["target_distribution"].to_string(),
        ])
    (REPORTS_DIR / "inspection_results.txt").write_text(
        "\n".join(output), encoding="utf-8"
    )


def main():
    df = load_raw_data()
    results = inspect_data(df)
    save_results(df, results)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Dataset dimensions: {results['shape'][0]} rows, {results['shape'][1]} columns")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(results["dtypes"].to_string())
    print("\nFirst five records:")
    print(results["head"].to_string())
    print("\nMissing values:")
    print(results["missing"].to_string())
    print(f"\nDuplicate records: {results['duplicates']}")
    print("\nNumerical summary:")
    print(results["numerical_summary"].to_string())
    if results["target_distribution"] is not None:
        print("\nTarget distribution (`buggy`):")
        print(results["target_distribution"].to_string())
    print("\nInspection files saved in:", REPORTS_DIR)


if __name__ == "__main__":
    main()
