from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DICTIONARY_PATH = PROJECT_ROOT / "reports" / "feature_dictionary.csv"


FEATURE_DESCRIPTIONS = {
    "la": "Lines added in the commit.",
    "ld": "Lines deleted in the commit.",
    "nf": "Number of modified files.",
    "nd": "Number of directories modified.",
    "ns": "Number of subsystems modified.",
    "ent": "Entropy of the modified code.",
    "ndev": "Number of developers who changed the files.",
    "age": "Average age of modified files.",
    "nuc": "Number of unique changes to the modified files.",
    "aexp": "Author experience.",
    "arexp": "Author review experience.",
    "asexp": "Author subsystem experience.",
}

EXPECTED_FEATURES = list(FEATURE_DESCRIPTIONS)
EXCLUDED_COLUMNS = {
    "commit_id": "Unique commit identifier; not a predictive feature.",
    "project": "Project name; preserved as metadata.",
    "fix": "Post-change fix status and possible target-adjacent metadata.",
    "year": "Time metadata; may be useful for chronological evaluation.",
    "author_date": "Timestamp metadata; may reveal chronology and is not a change metric.",
}


def get_available_features(df):
    return [column for column in EXPECTED_FEATURES if column in df.columns]


def get_feature_columns(df):
    available = get_available_features(df)
    numeric = [
        column for column in available
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    return numeric


def build_feature_dictionary(df):
    available = set(df.columns)
    rows = []

    for column in df.columns:
        if column in FEATURE_DESCRIPTIONS:
            rows.append({
                "feature_name": column,
                "description": FEATURE_DESCRIPTIONS[column],
                "data_type": str(df[column].dtype),
                "included_in_training": column in get_feature_columns(df),
            })
        elif column in EXCLUDED_COLUMNS:
            rows.append({
                "feature_name": column,
                "description": EXCLUDED_COLUMNS[column],
                "data_type": str(df[column].dtype),
                "included_in_training": False,
            })
        elif column == "buggy":
            rows.append({
                "feature_name": column,
                "description": "Target label: 0 for non-defective, 1 for defective.",
                "data_type": str(df[column].dtype),
                "included_in_training": False,
            })
        else:
            rows.append({
                "feature_name": column,
                "description": "Other dataset column; excluded unless verified as a valid change metric.",
                "data_type": str(df[column].dtype),
                "included_in_training": False,
            })

    return pd.DataFrame(rows)


def save_feature_dictionary(df, path=FEATURE_DICTIONARY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    build_feature_dictionary(df).to_csv(path, index=False)
