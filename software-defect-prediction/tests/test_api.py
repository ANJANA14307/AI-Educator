
from fastapi.testclient import TestClient

from backend.main import app, FEATURE_COLUMNS


client = TestClient(app)


SAMPLE = {
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


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["features_count"] == 12
    assert data["features"] == FEATURE_COLUMNS


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json=SAMPLE
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in (0, 1)

    assert data["label"] in (
        "Defective",
        "Non-defective"
    )

    assert "probability" in data

    probabilities = data["probability"]

    assert 0 <= probabilities["defective"] <= 1
    assert 0 <= probabilities["non_defective"] <= 1

    assert abs(
        probabilities["defective"]
        + probabilities["non_defective"]
        - 1
    ) < 1e-5


def test_prediction_rejects_missing_feature():

    payload = SAMPLE.copy()

    payload.pop("la")

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_prediction_rejects_extra_metadata():

    payload = SAMPLE.copy()

    payload["commit_id"] = "test-commit"

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_batch_prediction_endpoint():
    response = client.post(
        "/predict/batch",
        json={"records": [SAMPLE, SAMPLE]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["defective_count"] + data["non_defective_count"] == 2
    assert len(data["results"]) == 2


def test_batch_prediction_rejects_empty_records():
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422


def test_shap_explanation_endpoint():
    response = client.post("/explain", json=SAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert len(data["contributions"]) == 12
    assert {item["feature"] for item in data["contributions"]} == set(FEATURE_COLUMNS)
