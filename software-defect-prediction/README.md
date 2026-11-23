# AI-Educator – ApacheJIT Defect Prediction

This project is the data engineering and frontend portion of a Software Defect Prediction system.

## Setup

```powershell
cd software-defect-prediction
py -m pip install -r ..\requirements.txt
```

Place `apachejit_train.csv` in `data/raw/`. Official test CSVs may also be placed in `data/raw/`.

## Run the pipeline

```powershell
py src\inspect_data.py
py src\preprocessing.py
py -m pytest tests -q
```

## Start the dashboard

```powershell
py -m streamlit run frontend\app.py
```

The dashboard uses `data/processed/apachejit_processed.csv` when it exists and otherwise falls back to the raw training CSV.

## Chronological evaluation

Do not combine the training and test datasets. Use `apachejit_train.csv` for training and keep `apachejit_test_large.csv` or `apachejit_test_small.csv` as separate future test data. Apply the same feature columns and preprocessing fitted only on training data.