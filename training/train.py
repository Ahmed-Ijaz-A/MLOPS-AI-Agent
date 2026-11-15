"""Training entry point.

    python -m training.train all          # validate -> preprocess -> train -> evaluate -> gate (+MLflow)
    python -m training.train preprocess   # DVC stage
    python -m training.train fit          # DVC stage
    python -m training.train evaluate     # DVC stage (writes reports/metrics.json)

Hyper-parameters come from params.yaml; any ``--set key=value`` overrides the ``train`` section.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import yaml

from training import pipeline
from training.provenance import dataset_provenance

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("training.train")


def load_params(path: Path = ROOT / "params.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def run_all(
    params: dict[str, Any], *, log_to_mlflow: bool = True, register: bool = True, run_name: str | None = None
) -> dict:
    ds, pp, tp, gate = params["dataset"], params["preprocess"], params["train"], params["quality_gate"]
    data_path = ROOT / ds["path"]
    validation = pipeline.validate_dataset(data_path)
    prep = pipeline.preprocess(data_path, ROOT / "data/processed", pp["test_size"], pp["random_state"])
    model = pipeline.train_model(ROOT / "data/processed/train.csv", tp, ROOT / "models/model.joblib")
    evaluation = pipeline.evaluate_model(model, ROOT / "data/processed/test.csv")
    passed, reasons = pipeline.quality_gate(evaluation["metrics"], gate)
    result: dict[str, Any] = {"metrics": evaluation["metrics"], "quality_gate": passed, "reasons": reasons}
    if log_to_mlflow:
        import pandas as pd

        from training import tracking

        prov = dataset_provenance(ROOT, ds["path"], ds.get("version"))
        run_id = tracking.log_evaluation(
            model=model,
            params=tp,
            evaluation=evaluation,
            provenance=prov,
            validation=validation.to_dict(),
            preprocess_info=prep,
            extra_tags={"pipeline.orchestrator": "cli"},
            sample=pd.read_csv(ROOT / "data/processed/test.csv"),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            run_name=run_name,
        )
        tracking.record_quality_gate(run_id, passed, reasons)
        result["run_id"] = run_id
        if passed and register:
            result["model_version"] = tracking.register_candidate(run_id)
    log.info("result: %s", json.dumps(result, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "preprocess", "fit", "evaluate"])
    ap.add_argument(
        "--set", action="append", default=[], help="override train params, e.g. --set model_type=random_forest"
    )
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--run-name")
    a = ap.parse_args(argv)
    params = load_params()
    for kv in a.set:
        k, v = kv.split("=", 1)
        params["train"][k] = _coerce(v)

    if a.stage == "all":
        res = run_all(params, log_to_mlflow=not a.no_mlflow, register=not a.no_register, run_name=a.run_name)
        return 0 if res["quality_gate"] else 2
    if a.stage == "preprocess":
        pipeline.validate_dataset(ROOT / params["dataset"]["path"])
        pipeline.preprocess(ROOT / params["dataset"]["path"], ROOT / "data/processed", **params["preprocess"])
    elif a.stage == "fit":
        pipeline.train_model(ROOT / "data/processed/train.csv", params["train"], ROOT / "models/model.joblib")
    elif a.stage == "evaluate":
        model = joblib.load(ROOT / "models/model.joblib")
        ev = pipeline.evaluate_model(model, ROOT / "data/processed/test.csv")
        passed, reasons = pipeline.quality_gate(ev["metrics"], params["quality_gate"])
        pipeline.write_json(ROOT / "reports/metrics.json", {**ev["metrics"], "quality_gate_passed": passed})
        log.info("gate passed=%s reasons=%s", passed, reasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
