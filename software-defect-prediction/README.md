# ApacheJIT Software Defect Prediction System

This project predicts whether a software commit is likely to be defective. It uses the ApacheJIT dataset, the 12 commit-change features agreed in the data contract, a trained Gradient Boosting model, FastAPI, and Streamlit.

## Problem statement

Software teams can use commit-level defect risk estimates to prioritize code review and testing. The model returns one of two labels:

- `0` = Non-defective
- `1` = Defective

The system does not claim causality and does not evaluate official future-test datasets unless a separate future evaluation is run.

## Implemented architecture

```text
Dataset
  -> Preprocessing
  -> Feature Engineering
  -> Chronological Model Training
  -> Gradient Boosting Model
  -> FastAPI
  -> Streamlit Dashboard
```

The final model is stored at `models/final_model.joblib`. It contains the preprocessing pipeline and the 12-feature schema.

## Dataset

- Training data: `data/processed/apachejit_processed.csv`
- Raw training data: `data/raw/apachejit_train.csv`
- Official future test, large: `data/raw/apachejit_test_large.csv`
- Official future test, small: `data/raw/apachejit_test_small.csv`

The processed training dataset has 44,834 rows and 18 columns. Future test files are preserved separately and are not used for training or preprocessing fitting.

## Model features

```text
la, ld, nf, nd, ns, ent, ndev, age, nuc, aexp, arexp, asexp
```

`commit_id`, `project`, `fix`, `year`, and `author_date` are metadata. `author_date` supports chronological splitting but is never sent to the model.

## Technology

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- FastAPI and Pydantic
- Streamlit and Plotly
- Pytest

## Project structure

```text
software-defect-prediction/
  backend/main.py       FastAPI application
  frontend/app.py       Streamlit dashboard
  src/train.py          chronological model training
  src/evaluate.py       metrics and SHAP generation
  src/predict.py        reusable single/batch prediction logic
  src/preprocessing.py  data cleaning and target conversion
  src/features.py       approved feature contract
  data/                 raw and processed datasets
  models/               generated final_model.joblib
  reports/              metrics, charts, data contracts, SHAP results
  tests/                preprocessing, model, and API tests
```

## Setup

Use Python 3.11 and a virtual environment.

```powershell
cd E:\AI-Educator\software-defect-prediction
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ..\requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
```

## Training and evaluation

The existing model is trained with a chronological holdout: 35,867 training rows and 8,967 validation rows. Preprocessing is fitted only inside each training pipeline.

```powershell
.venv\Scripts\python.exe src\train.py
.venv\Scripts\python.exe src\evaluate.py
```

Do not retrain on or merge the official future-test CSV files.

## Start the backend

```powershell
cd E:\AI-Educator\software-defect-prediction
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health and model-loading status |
| GET | `/model-info` | Model metadata and validation metrics |
| POST | `/predict` | Single commit prediction |
| POST | `/predict/batch` | Batch JSON prediction |
| POST | `/explain` | Local SHAP-style feature contributions |

### Single prediction

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "@..\sample_input.json"
```

The JSON must contain exactly the 12 feature fields. Extra metadata and missing fields return validation errors.

### Batch prediction

```json
{
  "records": [
    {
      "la": 20,
      "ld": 10,
      "nf": 5,
      "nd": 3,
      "ns": 2,
      "ent": 1.5,
      "ndev": 4,
      "age": 30,
      "nuc": 3,
      "aexp": 2,
      "arexp": 1,
      "asexp": 2
    }
  ]
}
```

For CSV batch prediction, upload a CSV through Streamlit or use `src/predict.py --input-csv`.

```powershell
.venv\Scripts\python.exe src\predict.py --input-csv data\batch.csv --output-csv reports\batch_predictions.csv
```

## Start the dashboard

```powershell
cd E:\AI-Educator\software-defect-prediction
.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Default Streamlit URL: `http://localhost:8501`

The dashboard navigation is organized into three groups:

- **Workspace:** Overview, Predict a Commit, Batch Prediction
- **Analysis:** Model Insights, Dataset Explorer
- **Help:** How It Works

The Overview summarizes the ApacheJIT training data and saved chronological-validation metrics. Single and batch predictions use the existing FastAPI endpoints; individual predictions can also show their SHAP explanation. Model Insights reads saved evaluation reports and labels them as historical validation, not official future-test performance. Dataset Explorer keeps the training data and reserved future-test files in separate views.

## SHAP explainability

SHAP is generated for the selected Gradient Boosting model using transformed model inputs. Global importance is in `reports/shap_feature_importance.csv`. The `/explain` endpoint returns local feature contributions for one commit. SHAP shows model behavior, not proof of causation.

## Testing

```powershell
cd E:\AI-Educator\software-defect-prediction
.venv\Scripts\python.exe -m pytest tests -q
```

## Dataset validation

```powershell
.venv\Scripts\python.exe src\validate_dataset.py
```

## Limitations and future improvements

- Validation metrics are historical holdout metrics, not official future-test performance.
- The project does not yet run final future-test evaluation.
- Performance may vary across repositories and time periods.
- SHAP is descriptive and should not be interpreted as a causal explanation.
- Future work can add calibration, drift monitoring, authentication, deployment configuration, and scheduled re-training.
