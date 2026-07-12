from fastapi.testclient import TestClient

from api.main import app, reset_predictor_cache


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_blank_comments():
    response = client.post("/predict", json={"comment": "   "})

    assert response.status_code == 422


def test_predict_returns_503_when_model_artifacts_are_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("TURISMY_CLASSIFIER_PATH", str(tmp_path / "missing-classifier.joblib"))
    monkeypatch.setenv("TURISMY_REGRESSOR_PATH", str(tmp_path / "missing-regressor.joblib"))
    reset_predictor_cache()

    response = client.post("/predict", json={"comment": "The apartment was clean and central."})

    assert response.status_code == 503
    assert "Model artifacts are not available" in response.json()["detail"]
    reset_predictor_cache()
