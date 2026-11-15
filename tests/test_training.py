"""Training pipeline stages: validation, preprocessing, models, evaluation, quality gate, provenance."""

from __future__ import annotations

import pandas as pd
import pytest

from training import pipeline
from training.provenance import _read_head_without_git, dataset_provenance, file_md5


def test_validation_passes_on_clean_data(credit_csv):
    report = pipeline.validate_dataset(credit_csv)
    assert report.passed and report.rows == 400 and 0 < report.default_rate < 1


@pytest.mark.parametrize(
    "corrupt,match",
    [
        (lambda df: df.drop(columns=["age"]), "missing columns"),
        (lambda df: df.assign(age=df.age.where(df.index > 20)), "null values"),
        (lambda df: df.assign(age=-5), "outside"),
        (lambda df: df.assign(default=7), "unexpected labels"),
        (lambda df: df.head(50), "rows"),
        (lambda df: df.assign(credit_amount="lots"), "non-numeric"),
    ],
)
def test_validation_rejects_bad_data(credit_csv, tmp_path, corrupt, match):
    bad = tmp_path / "bad.csv"
    corrupt(pd.read_csv(credit_csv)).to_csv(bad, index=False)
    with pytest.raises(pipeline.DataValidationError, match=match):
        pipeline.validate_dataset(bad)


def test_validation_missing_file(tmp_path):
    with pytest.raises(pipeline.DataValidationError, match="not found"):
        pipeline.validate_dataset(tmp_path / "nope.csv")


def test_preprocess_is_stratified_and_deterministic(credit_csv, tmp_path):
    a = pipeline.preprocess(credit_csv, tmp_path / "a")
    b = pipeline.preprocess(credit_csv, tmp_path / "b")
    assert a == b and a["train_rows"] + a["test_rows"] == 400
    assert abs(a["train_default_rate"] - a["test_default_rate"]) < 0.03
    assert (tmp_path / "a" / "train.csv").read_bytes() == (tmp_path / "b" / "train.csv").read_bytes()


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_each_model_type_trains_and_evaluates(credit_csv, tmp_path, model_type):
    pipeline.preprocess(credit_csv, tmp_path)
    model = pipeline.train_model(
        tmp_path / "train.csv", {"model_type": model_type, "n_estimators": 30}, tmp_path / "m.joblib"
    )
    ev = pipeline.evaluate_model(model, tmp_path / "test.csv")
    assert set(ev["metrics"]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert ev["metrics"]["roc_auc"] > 0.6  # synthetic data has real signal
    assert sum(ev["confusion_matrix"].values()) == ev["metrics"]["test_rows"]


def test_unknown_model_type_rejected():
    with pytest.raises(ValueError, match="unknown model_type"):
        pipeline.build_model({"model_type": "magic"})


def test_quality_gate_pass_and_fail():
    thresholds = {"min_f1": 0.6, "min_recall": 0.65, "min_roc_auc": 0.75}
    ok, reasons = pipeline.quality_gate({"f1": 0.66, "recall": 0.8, "roc_auc": 0.81}, thresholds)
    assert ok and len(reasons) == 3
    bad, reasons = pipeline.quality_gate({"f1": 0.59, "recall": 0.53, "roc_auc": 0.79}, thresholds)
    assert not bad and sum("<" in r for r in reasons) == 2


def test_quality_gate_fails_on_missing_metric():
    ok, reasons = pipeline.quality_gate({"f1": 0.9}, {"min_f1": 0.5, "min_recall": 0.5})
    assert not ok and "recall: missing" in reasons


def test_provenance_detects_workspace_drift(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    (tmp_path / "data.csv.dvc").write_text(f"outs:\n- md5: {file_md5(data)}\n  size: 8\n  path: data.csv\n")
    assert dataset_provenance(tmp_path, "data.csv", "v1.0")["in_sync_with_dvc"] is True
    data.write_text("a,b\n1,3\n")
    assert dataset_provenance(tmp_path, "data.csv", "v1.0")["in_sync_with_dvc"] is False


def test_git_head_read_without_git_binary(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert _read_head_without_git(tmp_path) == ("a" * 40, "main")
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text(f"{'b' * 40} refs/heads/main\n")
    assert _read_head_without_git(tmp_path) == ("b" * 40, "main")
