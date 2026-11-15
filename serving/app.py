"""Credit-risk inference API (the workload deployed on Kubernetes).

    GET  /health    liveness  - process is up
    GET  /ready     readiness - model loaded and able to score
    GET  /version   model name / version / MLflow run / dataset version baked into the image
    POST /predict   score one or more loan applications
    GET  /metrics   Prometheus-format request counters

Configuration comes from the ConfigMap (env vars). The service fails fast at startup if the model
cannot be loaded, so a bad configuration surfaces as CrashLoopBackOff with a clear log line instead
of a pod that silently serves errors.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("inference")

NUMERIC = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
CATEGORICAL = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]


class Application(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    duration: int = Field(..., ge=1, le=120)
    credit_amount: float = Field(..., gt=0, le=1_000_000)
    installment_commitment: int = Field(..., ge=1, le=4)
    residence_since: int = Field(..., ge=1, le=4)
    age: int = Field(..., ge=18, le=100)
    existing_credits: int = Field(..., ge=1, le=10)
    num_dependents: int = Field(..., ge=0, le=10)
    checking_status: str
    credit_history: str
    purpose: str
    savings_status: str
    employment: str
    personal_status: str
    other_parties: str
    property_magnitude: str
    other_payment_plans: str
    housing: str
    job: str
    own_telephone: str
    foreign_worker: str


class PredictRequest(BaseModel):
    instances: list[Application] = Field(..., min_length=1, max_length=256)


class Prediction(BaseModel):
    default_probability: float
    decision: Literal["approve", "review"]


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class _State:
    model: Any = None
    metadata: dict[str, Any] = {}
    ready: bool = False
    counters: dict[str, int] = {"predict_requests": 0, "predict_rows": 0, "predict_errors": 0}
    lock = threading.Lock()


STATE = _State()


def load_model() -> None:
    model_path = Path(os.environ.get("MODEL_PATH", "/app/model/model.joblib"))
    meta_path = Path(os.environ.get("MODEL_METADATA_PATH", str(model_path.with_name("metadata.json"))))
    if not model_path.exists():
        log.error("FATAL: model file not found at MODEL_PATH=%s - check the ConfigMap", model_path)
        raise FileNotFoundError(f"model file not found: {model_path}")
    STATE.model = joblib.load(model_path)
    STATE.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    STATE.ready = True
    log.info(
        "model loaded: %s v%s (run %s)",
        STATE.metadata.get("model_name"),
        STATE.metadata.get("model_version"),
        STATE.metadata.get("run_id"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()  # raising here exits the process -> Kubernetes restarts it (fail fast)
    yield


app = FastAPI(title="Credit-risk inference API", version=os.environ.get("APP_VERSION", "1.0.0"), lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    if not STATE.ready or STATE.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": STATE.metadata.get("model_version")}


@app.get("/version")
def version() -> dict[str, Any]:
    return {
        "app_version": os.environ.get("APP_VERSION", "1.0.0"),
        "model_version_config": os.environ.get("MODEL_VERSION"),
        "decision_threshold": float(os.environ.get("DECISION_THRESHOLD", "0.5")),
        **{
            k: STATE.metadata.get(k)
            for k in (
                "model_name",
                "model_version",
                "run_id",
                "dataset_version",
                "dataset_md5",
                "git_commit",
                "metrics",
                "exported_at",
            )
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not STATE.ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    threshold = float(os.environ.get("DECISION_THRESHOLD", "0.5"))
    started = time.perf_counter()
    try:
        frame = pd.DataFrame([a.model_dump() for a in req.instances])[NUMERIC + CATEGORICAL]
        proba = STATE.model.predict_proba(frame)[:, 1]
    except Exception as exc:
        with STATE.lock:
            STATE.counters["predict_errors"] += 1
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    with STATE.lock:
        STATE.counters["predict_requests"] += 1
        STATE.counters["predict_rows"] += len(frame)
    log.info("scored %d rows in %.1f ms", len(frame), (time.perf_counter() - started) * 1000)
    return PredictResponse(
        model_version=str(STATE.metadata.get("model_version", "unknown")),
        predictions=[
            Prediction(default_probability=round(float(p), 4), decision="review" if p >= threshold else "approve")
            for p in proba
        ],
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    lines = [
        f'inference_{k}_total{{model_version="{STATE.metadata.get("model_version", "unknown")}"}} {v}'
        for k, v in STATE.counters.items()
    ]
    lines.append(f"inference_model_ready {1 if STATE.ready else 0}")
    return "\n".join(lines) + "\n"
