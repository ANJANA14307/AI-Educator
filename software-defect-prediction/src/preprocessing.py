from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_train.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "apachejit_processed.csv"


def load_raw_data(path=RAW_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. "
            "Place apachejit_train.csv inside data/raw/."
        )
    return pd.read_csv(path)


def validate_target(df, target_column="buggy"):
    if target_column not in df.columns:
        raise ValueError(
            f"Required target column '{target_column}' is missing. "
            f"Available columns: {', '.join(df.columns)}"
        )
    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values; no labels will be dropped automatically.")


def convert_target(df, target_column="buggy"):
    validate_target(df, target_column)
    series = df[target_column]
    converted = series.copy()

    if pd.api.types.is_bool_dtype(series):
        converted = series.astype("int64")
    else:
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "defective": 1, "non-defective": 0,
            "buggy": 1, "clean": 0,
        }
        unknown = sorted(set(normalized.unique()) - set(mapping))
        if unknown:
            raise ValueError(
                f"Unexpected target labels: {unknown}. "
                "Accepted labels are 0/1 or True/False."
            )
        converted = normalized.map(mapping).astype("int64")

    invalid = set(converted.unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Target conversion produced invalid labels: {sorted(invalid)}")
    return converted


def handle_missing_values(df):
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if cleaned[column].isna().any():
            median = cleaned[column].median()
            if pd.isna(median):
                raise ValueError(f"Cannot fill missing values in '{column}' because it has no valid values.")
            cleaned[column] = cleaned[column].fillna(median)

    for column in cleaned.select_dtypes(include=["object", "string", "bool"]).columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna("Unknown")
    return cleaned


def remove_duplicate_records(df):
    return df.drop_duplicates().copy()


def separate_features_and_target(df, feature_columns, target_column="buggy"):
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if target_column in feature_columns:
        raise ValueError("The target column must not be included in feature_columns.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def preprocess_data(df):
    validate_target(df)
    cleaned = handle_missing_values(df)
    before_duplicates = len(cleaned)
    cleaned = remove_duplicate_records(cleaned)

    cleaned["buggy"] = convert_target(cleaned)

    from features import get_feature_columns
    feature_columns = get_feature_columns(cleaned)
    X, y = separate_features_and_target(cleaned, feature_columns)

    metadata_columns = [
        column for column in ["commit_id", "project", "fix", "year", "author_date"]
        if column in cleaned.columns
    ]
    processed = cleaned[metadata_columns + feature_columns + ["buggy"]].copy()
    return processed, X, y, before_duplicates - len(cleaned)


def save_processed_dataset(processed, path=PROCESSED_DATA_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)
    return path


def main():
    df = load_raw_data()
    processed, X, y, duplicate_count = preprocess_data(df)
    output_path = save_processed_dataset(processed)
    from features import save_feature_dictionary
    save_feature_dictionary(df)
    print(f"Raw rows: {len(df)}")
    print(f"Processed rows: {len(processed)}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Target labels: {sorted(y.unique().tolist())}")
    print(f"Saved processed dataset: {output_path}")


if __name__ == "__main__":
    main()