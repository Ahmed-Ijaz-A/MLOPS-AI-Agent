"""Pure-Python pipeline stages. The Airflow DAG, ``dvc repro`` and the CLI all call these, so the
logic that decides whether a model is promotable is identical everywhere and unit-testable."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.data import CATEGORICAL, NUMERIC, TARGET

log = logging.getLogger("training.pipeline")


class DataValidationError(Exception):
    """Dataset failed validation; downstream training must not run."""


class QualityGateError(Exception):
    """Model failed the quality gate; it must not be promoted."""


# --------------------------------------------------------------------------- validation

NUMERIC_RANGES = {
    "duration": (1, 120),
    "credit_amount": (1, 1_000_000),
    "installment_commitment": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 100),
    "existing_credits": (1, 10),
    "num_dependents": (0, 10),
}


@dataclass
class ValidationReport:
    passed: bool
    rows: int
    columns: int
    default_rate: float | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def validate_dataset(path: str | Path, min_rows: int = 200, max_null_fraction: float = 0.0) -> ValidationReport:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"dataset not found at {path} - run `dvc pull` / `dvc checkout`")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV
        raise DataValidationError(f"dataset could not be parsed: {exc}") from exc

    errors: list[str] = []
    warnings: list[str] = []
    expected = [*NUMERIC, *CATEGORICAL, TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if len(df) < min_rows:
        errors.append(f"only {len(df)} rows; at least {min_rows} required")

    present = [c for c in expected if c in df.columns]
    nulls = df[present].isna().mean()
    bad_nulls = nulls[nulls > max_null_fraction]
    if not bad_nulls.empty:
        errors.append("null values in: " + ", ".join(f"{c} ({v:.1%})" for c, v in bad_nulls.items()))

    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any() and not df[col].isna().any():
                errors.append(f"{col} contains non-numeric values")
            out = series[(series < lo) | (series > hi)]
            if len(out):
                errors.append(f"{col}: {len(out)} values outside [{lo}, {hi}]")

    default_rate = None
    if TARGET in df.columns:
        labels = set(df[TARGET].dropna().unique().tolist())
        if not labels <= {0, 1}:
            errors.append(f"target has unexpected labels {sorted(labels)}")
        else:
            default_rate = float(df[TARGET].mean())
            if not 0.05 <= default_rate <= 0.6:
                warnings.append(f"unusual default rate {default_rate:.1%}")

    dupes = int(df.duplicated().sum())
    if dupes:
        warnings.append(f"{dupes} duplicate rows")

    report = ValidationReport(
        passed=not errors,
        rows=len(df),
        columns=df.shape[1],
        default_rate=default_rate,
        errors=errors,
        warnings=warnings,
    )
    log.info("validation report: %s", report.to_dict())
    if errors:
        raise DataValidationError("; ".join(errors))
    return report


# --------------------------------------------------------------------------- preprocessing


def preprocess(path: str | Path, out_dir: str | Path, test_size: float = 0.2, random_state: int = 42) -> dict:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False, lineterminator="\n")
    test.to_csv(out_dir / "test.csv", index=False, lineterminator="\n")
    info = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_default_rate": round(float(train[TARGET].mean()), 4),
        "test_default_rate": round(float(test[TARGET].mean()), 4),
    }
    log.info("preprocess: %s", info)
    return info


# --------------------------------------------------------------------------- training


def build_model(params: dict[str, Any]) -> Pipeline:
    model_type = params.get("model_type", "logistic_regression")
    rs = int(params.get("random_state", 42))
    if model_type == "logistic_regression":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 2000)),
            class_weight="balanced",
            random_state=rs,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
            class_weight="balanced",
            random_state=rs,
            n_jobs=1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_state=rs,
        )
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    features = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("features", features), ("clf", clf)])


def train_model(train_csv: str | Path, params: dict[str, Any], model_path: str | Path) -> Pipeline:
    df = pd.read_csv(train_csv)
    model = build_model(params)
    model.fit(df[[*NUMERIC, *CATEGORICAL]], df[TARGET])
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("trained %s on %d rows -> %s", params.get("model_type"), len(df), model_path)
    return model


# --------------------------------------------------------------------------- evaluation


def evaluate_model(model: Pipeline, test_csv: str | Path, threshold: float = 0.5) -> dict[str, Any]:
    df = pd.read_csv(test_csv)
    X, y = df[[*NUMERIC, *CATEGORICAL]], df[TARGET].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics["test_rows"] = len(df)
    return {"metrics": metrics, "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


# --------------------------------------------------------------------------- quality gate


def quality_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    """Every ``min_<metric>`` threshold must be met. Returns (passed, human-readable reasons)."""
    reasons: list[str] = []
    passed = True
    for key, minimum in thresholds.items():
        if not key.startswith("min_"):
            continue
        metric = key[4:]
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            passed = False
            reasons.append(f"{metric}: missing")
        elif value < minimum:
            passed = False
            reasons.append(f"{metric}={value:.4f} < required {minimum}")
        else:
            reasons.append(f"{metric}={value:.4f} >= {minimum}")
    return passed, reasons


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
