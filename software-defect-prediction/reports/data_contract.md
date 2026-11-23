# ApacheJIT Data Contract

## Processed dataset

- Location: `data/processed/apachejit_processed.csv`
- Loader: `pd.read_csv("data/processed/apachejit_processed.csv")`
- Processed source: `data/raw/apachejit_train.csv`

## Target

- Column name: `buggy`
- Exact labels:
  - `0` = Non-defective
  - `1` = Defective
- Data type: integer

## Model feature columns

The exact ordered feature list is:

1. `la` (`int64`)
2. `ld` (`int64`)
3. `nf` (`int64`)
4. `nd` (`int64`)
5. `ns` (`int64`)
6. `ent` (`float64`)
7. `ndev` (`float64`)
8. `age` (`float64`)
9. `nuc` (`float64`)
10. `aexp` (`int64`)
11. `arexp` (`float64`)
12. `asexp` (`float64`)

```python
feature_columns = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]
X = processed_df[feature_columns]
y = processed_df["buggy"]
```

## Excluded columns

- `commit_id`: unique identifier; excluded to prevent memorization.
- `project`: preserved metadata; not included as a model feature.
- `fix`: possible target-adjacent metadata; excluded.
- `year`: preserved chronological metadata; excluded unless Person 2 deliberately evaluates a time feature.
- `author_date`: timestamp/chronological metadata; excluded.

## Training and test separation

The official training file is `apachejit_train.csv`. The official test files are `apachejit_test_large.csv` and `apachejit_test_small.csv`. Keep these files separate. Do not combine training and future test data. Fit all preprocessing decisions on the training split only and apply the saved feature columns to test data.
