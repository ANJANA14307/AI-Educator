
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import predict_one, predict_records

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )
    loaded = joblib.load(MODEL_PATH)
    if "pipeline" not in loaded:
        raise ValueError("Model artifact has no pipeline.")
    if loaded.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema mismatch.")
    return loaded


try:
    artifact = load_model()
except (FileNotFoundError, ValueError, OSError):
    artifact = None

model_pipeline = artifact["pipeline"] if artifact else None
model_name = artifact["model_name"] if artifact else "Unavailable"
threshold = float(artifact["threshold"]) if artifact else 0.5


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="ApacheJIT Software Defect Prediction API",
    description=(
        "Predict whether a software commit is likely "
        "to contain defects using machine learning."
    ),
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    la: float = Field(allow_inf_nan=False)
    ld: float = Field(allow_inf_nan=False)
    nf: float = Field(allow_inf_nan=False)
    nd: float = Field(allow_inf_nan=False)
    ns: float = Field(allow_inf_nan=False)
    ent: float = Field(allow_inf_nan=False)

    ndev: float = Field(allow_inf_nan=False)
    age: float = Field(allow_inf_nan=False)
    nuc: float = Field(allow_inf_nan=False)
    aexp: float = Field(allow_inf_nan=False)
    arexp: float = Field(allow_inf_nan=False)
    asexp: float = Field(allow_inf_nan=False)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PredictionRequest] = Field(min_length=1)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "ApacheJIT Defect Prediction API",
        "status": "running",
        "documentation": "/docs"
    }


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy" if model_pipeline is not None else "degraded",
        "model_loaded": model_pipeline is not None,
        "model_name": model_name,
        "model_path": str(MODEL_PATH)
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.")

    metrics_path = ROOT / "reports" / "evaluation_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

    return {
        "model_name": model_name,
        "features_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "target": "buggy",
        "threshold": threshold,
        "training_rows": artifact["training_rows"],
        "metrics": metrics
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.")
    features = {feature: getattr(request, feature) for feature in FEATURE_COLUMNS}
    return predict_one(features, artifact)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.")
    records = [
        {feature: getattr(record, feature) for feature in FEATURE_COLUMNS}
        for record in request.records
    ]
    return predict_records(records, artifact)


@app.post("/explain")
def explain(request: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable.")

    input_data = pd.DataFrame(
        [{feature: getattr(request, feature) for feature in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS
    )
    transformed = model_pipeline[:-1].transform(input_data)
    classifier = model_pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(explainer.shap_values(transformed))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    row_values = shap_values[0]
    contributions = sorted(
        (
            {"feature": feature, "shap_value": round(float(value), 6)}
            for feature, value in zip(FEATURE_COLUMNS, row_values)
        ),
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    return {
        "model_name": model_name,
        "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[-1]),
        "contributions": contributions
    }
