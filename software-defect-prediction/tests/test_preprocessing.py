import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import features
import preprocessing


def test_raw_dataset_loading():
    df = preprocessing.load_raw_data()
    assert df.shape == (44834, 18)
    assert "buggy" in df.columns


def test_target_column_validation():
    with pytest.raises(ValueError, match="Required target column"):
        preprocessing.validate_target(pd.DataFrame({"la": [1]}))


def test_target_missing_values_are_rejected():
    df = pd.DataFrame({"buggy": [True, None]})
    with pytest.raises(ValueError, match="missing values"):
        preprocessing.validate_target(df)


def test_target_label_conversion_boolean_and_numeric():
    boolean_df = pd.DataFrame({"buggy": [True, False, True]})
    numeric_df = pd.DataFrame({"buggy": [1, 0, 1]})

    assert preprocessing.convert_target(boolean_df).tolist() == [1, 0, 1]
    assert preprocessing.convert_target(numeric_df).tolist() == [1, 0, 1]


def test_target_rejects_unknown_labels():
    df = pd.DataFrame({"buggy": ["yes", "no"]})
    with pytest.raises(ValueError, match="Unexpected target labels"):
        preprocessing.convert_target(df)


def test_missing_value_handling_uses_median():
    df = pd.DataFrame({"la": [1.0, None, 3.0], "project": ["a", None, "b"]})
    cleaned = preprocessing.handle_missing_values(df)
    assert cleaned["la"].isna().sum() == 0
    assert cleaned["la"].iloc[1] == 2.0
    assert cleaned["project"].isna().sum() == 0


def test_duplicate_handling_removes_duplicate_rows():
    df = pd.DataFrame({"la": [1, 1, 2], "ld": [2, 2, 3]})
    result = preprocessing.remove_duplicate_records(df)
    assert len(result) == 2
    assert result.duplicated().sum() == 0


def test_identifier_columns_are_excluded():
    df = pd.DataFrame({
        "commit_id": ["a", "b"],
        "project": ["p1", "p2"],
        "fix": [False, True],
        "year": [2015, 2016],
        "author_date": [1, 2],
        "la": [10, 20],
        "buggy": [True, False],
    })
    feature_columns = features.get_feature_columns(df)
    assert "la" in feature_columns
    assert "commit_id" not in feature_columns
    assert "project" not in feature_columns
    assert "fix" not in feature_columns
    assert "year" not in feature_columns
    assert "author_date" not in feature_columns
    assert "buggy" not in feature_columns


def test_feature_target_separation_has_no_target_leakage():
    df = pd.DataFrame({"la": [1, 2], "ld": [3, 4], "buggy": [0, 1]})
    X, y = preprocessing.separate_features_and_target(df, ["la", "ld"])
    assert X.columns.tolist() == ["la", "ld"]
    assert y.tolist() == [0, 1]
    assert "buggy" not in X.columns


def test_processed_dataset_creation_and_output_location(tmp_path):
    df = pd.DataFrame({
        "commit_id": ["a", "b"],
        "project": ["p1", "p2"],
        "buggy": [True, False],
        "fix": [False, False],
        "year": [2015, 2016],
        "author_date": [100, 200],
        "la": [10, 20],
    })
    processed, X, y, duplicates = preprocessing.preprocess_data(df)
    output = preprocessing.save_processed_dataset(
        processed, tmp_path / "apachejit_processed.csv"
    )
    assert output == tmp_path / "apachejit_processed.csv"
    assert output.exists()
    assert set(y.unique()) == {0, 1}
    assert "buggy" not in X.columns
    assert len(processed) == 2
