"""Inference service (the workload deployed to Kubernetes)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from training import pipeline

APPLICATION = {
    "duration": 24,
    "credit_amount": 4000,
    "installment_commitment": 3,
    "residence_since": 2,
    "age": 35,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_status": "<0",
    "credit_history": "existing paid",
    "purpose": "new car",
    "savings_status": "<100",
    "employment": "1<=X<4",
    "personal_status": "male single",
    "other_parties": "none",
    "property_magnitude": "car",
    "other_payment_plans": "none",
    "housing": "own",
    "job": "skilled",
    "own_telephone": "none",
    "foreign_worker": "yes",
}


@pytest.fixture
def model_dir(credit_csv, tmp_path):
    pipeline.preprocess(credit_csv, tmp_path)
    pipeline.train_model(
        tmp_path / "train.csv", {"model_type": "logistic_regression"}, tmp_path / "model" / "model.joblib"
    )
    (tmp_path / "model" / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "credit-risk-classifier",
                "model_version": "7",
                "run_id": "r" * 32,
                "dataset_version": "v2.0",
            }
        )
    )
    return tmp_path / "model"


@pytest.fixture
def api(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    from serving import app as serving_app

    serving_app.STATE.ready = False
    serving_app.STATE.counters = {k: 0 for k in serving_app.STATE.counters}
    with TestClient(serving_app.app) as client:
        yield client


def test_probes(api):
    assert api.get("/health").json() == {"status": "alive"}
    assert api.get("/ready").json()["model_version"] == "7"


def test_version_reports_model_provenance(api):
    v = api.get("/version").json()
    assert v["model_version"] == "7" and v["dataset_version"] == "v2.0"


def test_predict_batch(api):
    res = api.post("/predict", json={"instances": [APPLICATION, {**APPLICATION, "checking_status": "no checking"}]})
    body = res.json()
    assert res.status_code == 200 and len(body["predictions"]) == 2 and body["model_version"] == "7"
    assert all(
        0 <= p["default_probability"] <= 1 and p["decision"] in ("approve", "review") for p in body["predictions"]
    )


@pytest.mark.parametrize("patch", [{"age": 12}, {"credit_amount": -5}, {"unknown_field": 1}, {"duration": "long"}])
def test_predict_rejects_invalid_input(api, patch):
    assert api.post("/predict", json={"instances": [{**APPLICATION, **patch}]}).status_code == 422


def test_metrics_endpoint_counts_requests(api):
    api.post("/predict", json={"instances": [APPLICATION]})
    text = api.get("/metrics").text
    assert 'inference_predict_requests_total{model_version="7"} 1' in text


def test_missing_model_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from serving import app as serving_app

    with pytest.raises(FileNotFoundError), TestClient(serving_app.app):
        pass
