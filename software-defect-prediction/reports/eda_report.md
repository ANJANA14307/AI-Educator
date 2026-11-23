# ApacheJIT EDA Report

## Dataset source

ApacheJIT Just-In-Time Software Defect Prediction dataset, Zenodo record 5907847:
https://zenodo.org/records/5907847

The current local training file is `data/raw/apachejit_train.csv`.

## Dataset dimensions

- Rows: 44,834
- Columns: 18
- Available change features: 12

## Actual column and data quality findings

The actual columns are:
`commit_id`, `project`, `buggy`, `fix`, `year`, `author_date`, `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`.

- Missing values: 0 across all columns.
- Duplicate records: 0.
- Target values: boolean `True` and `False`.

## Target distribution

- Defective (`True`): 22,421 records (50.01%)
- Non-defective (`False`): 22,413 records (49.99%)

## Numerical feature observations

The 12 available change metrics are `la`, `ld`, `nf`, `nd`, `ns`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`, and `asexp`. Their distributions contain wide ranges and high maximum values in several fields. Visual boxplots and histograms are saved in `reports/figures/`.

The numerical summary and missing-value exports are:

- `reports/numerical_summary.csv`
- `reports/missing_values.csv`
- `reports/target_distribution.csv`
- `reports/inspection_results.txt`

## Outlier handling

Extreme values were investigated through visualizations but were not deleted. ApacheJIT change metrics can legitimately contain very large commit or experience values, so automatic deletion would risk removing valid observations.

## Preprocessing decisions

- Required target column: `buggy`.
- Target conversion: `True` to 1 and `False` to 0.
- Missing numeric values are filled with the training median if they occur in future data.
- Missing categorical/string values are filled with `Unknown`.
- Duplicate records are removed only from the training data after inspection.
- No scaling is applied in preprocessing.
- No oversampling or undersampling is applied.

## Feature engineering decisions

The available 12 numerical change metrics are included as model features. `commit_id` is excluded as an identifier. `project`, `fix`, `year`, and `author_date` are preserved in the processed dataset as metadata and excluded from the model feature list. `buggy` is retained only as the target.

## Limitations and assumptions

- The local training file spans 2003 through 2016 according to the `year` column.
- Official test files should remain chronologically separate.
- Median missing-value handling is documented for future robustness, although the actual training file has no missing values.
- Correlations and visual observations do not prove causation.
- Project-specific historical patterns may not generalize to unrelated repositories.
