"""
SignSense AI - FastAPI Integration Unit Test Suite (Phase 6)
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "SignSense AI Backend"

def test_predict_endpoint_valid_payload():
    # 21 dummy landmarks
    landmarks_payload = [{"x": 0.5 + i*0.01, "y": 0.5 + i*0.01, "z": 0.0} for i in range(21)]
    request_body = {
        "landmarks": landmarks_payload,
        "classifier": "random_forest"
    }

    response = client.post("/api/v1/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_letter" in data
    assert "confidence" in data
    assert "classifier_used" in data
    assert "top_probabilities" in data
    assert len(data["top_probabilities"]) > 0
    assert data["processing_time_ms"] >= 0.0

def test_predict_endpoint_invalid_landmark_count():
    # Only 10 landmarks (invalid)
    landmarks_payload = [{"x": 0.1, "y": 0.1, "z": 0.0} for _ in range(10)]
    request_body = {"landmarks": landmarks_payload}

    response = client.post("/api/v1/predict", json=request_body)
    assert response.status_code == 422 or response.status_code == 400
