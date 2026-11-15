"""Populate MLflow with a realistic experiment history (several model types on both dataset versions).

    python scripts/seed_experiments.py

For each dataset version it checks out that DVC version into the workspace, runs the full pipeline
for each configuration (validate -> train -> evaluate -> quality gate -> register if passed), then
restores the latest data. Every run records dataset md5 / version / git commit for provenance.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIGS = [
    {"model_type": "logistic_regression", "C": 1.0},
    {"model_type": "random_forest", "n_estimators": 300, "max_depth": 8},
    {"model_type": "gradient_boosting", "n_estimators": 200, "max_depth": 3},
]
DATA = "data/raw/credit_risk.csv"


def checkout(rev: str) -> None:
    subprocess.run(["git", "checkout", rev, "--", f"{DATA}.dvc"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "dvc", "checkout", DATA, "--force"], cwd=ROOT, check=True)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    from training.train import load_params, run_all

    base = load_params()
    plan = [
        ("data-v1.0", "v1.0", CONFIGS[:1]),
        ("data-v2.0", "v2.0", [*CONFIGS, {"model_type": "logistic_regression", "C": 0.1}]),
    ]
    try:
        for tag, version, configs in plan:
            checkout(tag)
            for cfg in configs:
                params = copy.deepcopy(base)
                params["dataset"]["version"] = version
                params["train"].update(cfg)
                name = f"{cfg['model_type']}-{version}" + (f"-C{cfg['C']}" if "C" in cfg and cfg["C"] != 1.0 else "")
                res = run_all(params, run_name=name)
                print(
                    f"{name:40s} f1={res['metrics']['f1']:.4f} gate={'PASS' if res['quality_gate'] else 'FAIL'} "
                    f"version={res.get('model_version', '-')}"
                )
    finally:
        checkout("HEAD")


if __name__ == "__main__":
    main()
