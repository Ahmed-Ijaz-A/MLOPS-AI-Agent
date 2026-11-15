"""Write a small model into serving/model/ from synthetic data (CI / docker smoke tests only).

Real deployments use scripts/deploy_model.py, which exports a registered MLflow model instead.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fakes import synthetic_credit_frame  # noqa: E402
from training import pipeline  # noqa: E402


def main() -> None:
    out = ROOT / "serving" / "model"
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "d.csv"
        synthetic_credit_frame(500, seed=11).to_csv(csv, index=False)
        pipeline.preprocess(csv, tmp)
        pipeline.train_model(Path(tmp) / "train.csv", {"model_type": "logistic_regression"}, out / "model.joblib")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "credit-risk-classifier",
                "model_version": "0",
                "run_id": None,
                "dataset_version": "synthetic-smoke",
            },
            indent=2,
        )
    )
    print(f"smoke model written to {out}")


if __name__ == "__main__":
    main()
