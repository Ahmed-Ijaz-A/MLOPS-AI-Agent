"""Dataset creation for the credit-risk use case (Statlog German Credit, OpenML ``credit-g``).

Two dataset versions model how data evolves in production:

* ``v1.0`` - the initial extract: the first 700 loan applications.
* ``v2.0`` - the full 1000 applications (a new batch of 300 applications appended).

Each version is committed through DVC and tagged in git (``data-v1.0`` / ``data-v2.0``), so any
MLflow run can be traced back to the exact bytes it was trained on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
NUMERIC = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
TARGET = "default"  # 1 = bad credit risk (the positive, costly class)
VERSIONS = {"v1.0": 700, "v2.0": 1000}


def load_credit_g() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    df[TARGET] = (df["class"].astype(str) == "bad").astype(int)
    df = df.drop(columns=["class"])
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    return df[[*NUMERIC, *CATEGORICAL, TARGET]]


def make_dataset(version: str, out_path: str | Path) -> pd.DataFrame:
    if version not in VERSIONS:
        raise ValueError(f"unknown dataset version {version}; choose from {list(VERSIONS)}")
    df = load_credit_g().head(VERSIONS[version]).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, lineterminator="\n")
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Materialise a dataset version")
    p.add_argument("--version", default="v2.0", choices=list(VERSIONS))
    p.add_argument("--out", default="data/raw/credit_risk.csv")
    a = p.parse_args()
    frame = make_dataset(a.version, a.out)
    print(f"wrote {len(frame)} rows ({frame[TARGET].mean():.1%} defaults) to {a.out}")
