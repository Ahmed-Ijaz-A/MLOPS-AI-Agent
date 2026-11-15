"""ml_training_pipeline - Airflow 3 DAG.

validate_dataset -> preprocess_data -> train_model -> evaluate_and_log -> quality_gate -> register_model

* Retries: every task retries twice (15 s apart) for transient faults. Deterministic failures
  (invalid data, failed quality gate) raise ``AirflowFailException`` so they fail fast and are
  NOT retried - retrying bad data would only waste compute.
* Promotion safety: ``register_model`` runs only if every upstream task succeeded, so a failed
  validation or quality gate can never register / promote a model.
* Failure injection for demos (``dag_run.conf``):
    {"inject_failure": "corrupt_data"}   -> validation fails (nulls + out-of-range ages)
    {"inject_failure": "low_quality"}    -> labels shuffled, quality gate fails, nothing registered
    {"inject_failure": "flaky_training"} -> first training attempt crashes, retry succeeds
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/project"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

log = logging.getLogger("airflow.task")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
    "execution_timeout": timedelta(minutes=15),
}


def _params() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_DIR / "params.yaml").read_text(encoding="utf-8"))


def _workdir() -> Path:
    ctx = get_current_context()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ctx["run_id"])
    path = PROJECT_DIR / "artifacts" / "airflow_runs" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _injection() -> str:
    return str(get_current_context()["params"].get("inject_failure", "none"))


@dag(
    dag_id="ml_training_pipeline",
    description="Validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> register",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["mlops", "training", "credit-risk"],
    params={
        "inject_failure": Param("none", type="string", enum=["none", "corrupt_data", "low_quality", "flaky_training"]),
        "model_type": Param("", type="string", description="Override params.yaml train.model_type (blank = keep)"),
        "reason": Param("manual", type="string"),
    },
    doc_md=__doc__,
)
def ml_training_pipeline():
    @task
    def validate_dataset() -> dict:
        from training.pipeline import DataValidationError, validate_dataset
        from training.provenance import dataset_provenance

        p = _params()
        data_rel = p["dataset"]["path"]
        data_path = PROJECT_DIR / data_rel
        prov = dataset_provenance(PROJECT_DIR, data_rel, p["dataset"].get("version"))
        log.info("dataset provenance: %s", prov)

        if _injection() == "corrupt_data":
            import pandas as pd

            corrupted = _workdir() / "corrupted.csv"
            df = pd.read_csv(data_path)
            df.loc[df.sample(frac=0.1, random_state=1).index, "age"] = -5
            df.loc[df.sample(frac=0.05, random_state=2).index, "credit_amount"] = None
            df.to_csv(corrupted, index=False)
            log.warning("FAILURE INJECTION: validating a corrupted copy at %s", corrupted)
            data_path = corrupted
        elif not prov["in_sync_with_dvc"]:
            raise AirflowFailException(
                f"Workspace data {data_rel} (md5={prov['workspace_md5']}) does not match the DVC-tracked "
                f"version (md5={prov['tracked_md5']}). Run `dvc checkout` before training."
            )

        try:
            report = validate_dataset(data_path)
        except DataValidationError as exc:
            log.error("dataset validation FAILED: %s", exc)
            raise AirflowFailException(f"Dataset validation failed: {exc}") from exc
        log.info("dataset validation passed: %s", report.to_dict())
        return {"data_path": str(data_path), "validation": report.to_dict(), "provenance": prov}

    @task
    def preprocess_data(validated: dict) -> dict:
        from training.pipeline import preprocess

        p = _params()["preprocess"]
        out = _workdir() / "processed"
        info = preprocess(validated["data_path"], out, p["test_size"], p["random_state"])
        return {**validated, "processed_dir": str(out), "preprocess": info}

    @task
    def train_model(prepared: dict) -> dict:
        import pandas as pd

        from training.pipeline import train_model as fit

        ctx = get_current_context()
        params = dict(_params()["train"])
        override = str(ctx["params"].get("model_type") or "").strip()
        if override:
            params["model_type"] = override
        train_csv = Path(prepared["processed_dir"]) / "train.csv"

        injection = _injection()
        if injection == "flaky_training" and ctx["ti"].try_number == 1:
            raise RuntimeError("FAILURE INJECTION: simulated transient worker crash (will be retried)")
        if injection == "low_quality":
            log.warning("FAILURE INJECTION: shuffling training labels to simulate label corruption")
            df = pd.read_csv(train_csv)
            df["default"] = df["default"].sample(frac=1.0, random_state=7).to_numpy()
            train_csv = Path(prepared["processed_dir"]) / "train_shuffled.csv"
            df.to_csv(train_csv, index=False)

        model_path = _workdir() / "model.joblib"
        fit(train_csv, params, model_path)
        log.info("model trained with params %s", params)
        return {**prepared, "model_path": str(model_path), "train_params": params}

    @task
    def evaluate_and_log(trained: dict) -> dict:
        import joblib
        import pandas as pd

        from training import tracking
        from training.pipeline import evaluate_model

        ctx = get_current_context()
        test_csv = Path(trained["processed_dir"]) / "test.csv"
        model = joblib.load(trained["model_path"])
        evaluation = evaluate_model(model, test_csv)
        log.info("evaluation: %s", evaluation)
        run_id = tracking.log_evaluation(
            model=model,
            params=trained["train_params"],
            evaluation=evaluation,
            provenance=trained["provenance"],
            validation=trained["validation"],
            preprocess_info=trained["preprocess"],
            extra_tags={
                "pipeline.orchestrator": "airflow",
                "airflow.dag_id": ctx["dag"].dag_id,
                "airflow.run_id": ctx["run_id"],
                "failure_injection": _injection(),
            },
            sample=pd.read_csv(test_csv),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            run_name=f"airflow-{trained['train_params']['model_type']}",
        )
        log.info("logged MLflow run %s", run_id)
        return {"run_id": run_id, "metrics": evaluation["metrics"]}

    @task(retries=0)
    def quality_gate(evaluated: dict) -> dict:
        from training import tracking
        from training.pipeline import quality_gate as gate

        thresholds = _params()["quality_gate"]
        passed, reasons = gate(evaluated["metrics"], thresholds)
        tracking.record_quality_gate(evaluated["run_id"], passed, reasons, os.environ.get("MLFLOW_TRACKING_URI"))
        if not passed:
            log.error("QUALITY GATE FAILED for run %s: %s", evaluated["run_id"], reasons)
            raise AirflowFailException(f"Quality gate failed: {'; '.join(r for r in reasons if '<' in r)}")
        log.info("quality gate passed: %s", reasons)
        return {**evaluated, "gate_reasons": reasons}

    @task
    def register_model(gated: dict) -> dict:
        from training import tracking

        version = tracking.register_candidate(gated["run_id"], tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
        log.info("registered run %s as model version %s (alias=candidate)", gated["run_id"], version)
        shutil.rmtree(_workdir() / "processed", ignore_errors=True)
        return {"run_id": gated["run_id"], "model_version": version}

    register_model(quality_gate(evaluate_and_log(train_model(preprocess_data(validate_dataset())))))


ml_training_pipeline()
