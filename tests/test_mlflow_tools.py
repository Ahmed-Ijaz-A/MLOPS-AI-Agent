"""MLflow tools against a REAL MLflow tracking store (sqlite) seeded with runs + a registered model."""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

EXP = "credit-risk-classification"
MODEL = "credit-risk-classifier"

SEED = [
    # name, model_type, f1, recall, roc_auc, gate, dataset
    ("lr-v1", "logistic_regression", 0.6122, 0.7317, 0.7615, "passed", "v1.0"),
    ("rf-v2", "random_forest", 0.6466, 0.7167, 0.8019, "passed", "v2.0"),
    ("gb-v2", "gradient_boosting", 0.6600, 0.5333, 0.7904, "failed", "v2.0"),
    ("lr-v2", "logistic_regression", 0.6575, 0.8000, 0.8058, "passed", "v2.0"),
    ("lr-c01-v2", "logistic_regression", 0.6309, 0.7833, 0.8094, "passed", "v2.0"),
]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    uri = "sqlite:///" + (tmp_path_factory.mktemp("mlflow") / "mlflow.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    exp_id = client.create_experiment(EXP)
    client.create_experiment("credit-risk-baselines")
    ids = {}
    for name, mt, f1, rec, auc, gate, ds in SEED:
        run = client.create_run(
            exp_id,
            run_name=name,
            tags={
                "model_type": mt,
                "quality_gate.status": gate,
                "dataset.version": ds,
                "dataset.md5": f"md5-{ds}",
                "git.commit": "abc1234" + "0" * 33,
            },
        )
        for k, v in {"f1": f1, "recall": rec, "roc_auc": auc, "precision": 0.55, "accuracy": 0.75}.items():
            client.log_metric(run.info.run_id, k, v)
        client.log_param(run.info.run_id, "model_type", mt)
        client.set_terminated(run.info.run_id)
        ids[name] = run.info.run_id
    client.create_registered_model(MODEL)
    for name in ("rf-v2", "lr-v2"):
        client.create_model_version(MODEL, f"runs:/{ids[name]}/model", ids[name])
    client.set_registered_model_alias(MODEL, "champion", "1")
    client.set_registered_model_alias(MODEL, "candidate", "2")
    return client, ids


@pytest.fixture
def ex(make_executor, store):
    return make_executor({"mlflow": store[0]})


def test_list_experiments(ex):
    r = ex.execute("mlflow_list_experiments", {})
    names = {e["name"]: e["runs"] for e in r.data["experiments"]}
    assert names[EXP] == 5 and "credit-risk-baselines" in names


def test_list_runs_newest_first_with_provenance(ex):
    r = ex.execute("mlflow_list_runs", {"max_results": 3})
    assert len(r.data["runs"]) == 3 and r.data["runs"][0]["run_name"] == "lr-c01-v2"
    assert r.data["runs"][0]["dataset_version"] == "v2.0"


def test_get_run_details(ex, store):
    run_id = store[1]["lr-v2"]
    r = ex.execute("mlflow_get_run", {"run_id": run_id})
    assert r.data["metrics"]["f1"] == pytest.approx(0.6575) and r.data["params"]["model_type"] == "logistic_regression"
    assert r.data["tags"]["quality_gate.status"] == "passed"


def test_best_run_by_f1_flags_failed_gate(ex):
    r = ex.execute("mlflow_find_best_run", {"metric": "f1"})
    best = r.data["best"]
    assert best["run_name"] == "gb-v2"  # highest raw F1 ...
    caveats = " ".join(r.data["caveats"])
    assert "not promotable" in caveats  # ... but the agent says it is not production-ready
    assert "Offline test metrics alone do not make a model production-ready" in caveats


def test_best_run_restricted_to_gate_passing(ex):
    r = ex.execute("mlflow_find_best_run", {"metric": "f1", "require_quality_gate": True})
    assert r.data["best"]["run_name"] == "lr-v2" and r.data["runner_up"]["run_name"] == "rf-v2"


def test_best_run_by_other_metric(ex):
    r = ex.execute("mlflow_find_best_run", {"metric": "roc_auc"})
    assert r.data["best"]["run_name"] == "lr-c01-v2"


def test_compare_last_five_recommends_gate_passing_candidate(ex):
    r = ex.execute("mlflow_compare_runs", {"last_n": 5, "metric": "f1"})
    assert len(r.data["runs"]) == 5
    assert r.data["leaders_by_metric"]["f1"]["value"] == pytest.approx(0.66)
    recommended = next(x for x in r.data["runs"] if x["run_id"] == r.data["recommended"])
    assert recommended["run_name"] == "lr-v2" and recommended["quality_gate"] == "passed"
    assert any("gate-passing" in j for j in r.data["justification"])
    assert any("DIFFERENT dataset versions" in c for c in r.data["caveats"])


def test_compare_explicit_runs(ex, store):
    ids = store[1]
    r = ex.execute("mlflow_compare_runs", {"run_ids": [ids["lr-v1"], ids["lr-v2"]]})
    assert {x["run_name"] for x in r.data["runs"]} == {"lr-v1", "lr-v2"}


def test_compare_experiments(ex):
    r = ex.execute("mlflow_compare_experiments", {"experiments": [EXP, "credit-risk-baselines"]})
    first = r.data["experiments"][0]
    assert first["runs"] == 5 and first["gate_pass_rate"] == pytest.approx(0.8)
    assert r.data["experiments"][1]["runs"] == 0


def test_model_registry_aliases(ex, store):
    r = ex.execute("mlflow_get_model_registry", {})
    assert r.data["aliases"] == {"champion": "1", "candidate": "2"}
    v1 = next(v for v in r.data["versions"] if v["version"] == "1")
    assert v1["aliases"] == ["champion"] and v1["run_id"] == store[1]["rf-v2"]


def test_unknown_experiment_is_invalid_request(ex):
    r = ex.execute("mlflow_list_runs", {"experiment": "does-not-exist"})
    assert r.status == "invalid_request" and "Existing experiments" in r.error


def test_unknown_registered_model_is_invalid_request(ex):
    assert ex.execute("mlflow_get_model_registry", {"model_name": "ghost"}).status == "invalid_request"


@pytest.mark.parametrize("bad", ["123", "not-a-run-id", "Z" * 32])
def test_malformed_run_id_rejected(ex, bad):
    assert ex.execute("mlflow_get_run", {"run_id": bad}).status == "invalid_request"


def test_unreachable_tracking_server_is_unavailable(make_executor):
    client = MlflowClient(tracking_uri="http://127.0.0.1:9")
    r = make_executor({"mlflow": client}).execute("mlflow_list_experiments", {})
    assert r.status in ("unavailable", "timeout") and r.data is None
