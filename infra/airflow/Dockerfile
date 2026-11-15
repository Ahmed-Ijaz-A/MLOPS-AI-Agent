# Airflow 3 image with the training dependencies baked in (no slow _PIP_ADDITIONAL_REQUIREMENTS).
FROM apache/airflow:3.1.0-python3.12

USER airflow
# Same scikit-learn / pandas versions as the agent + inference image so models are portable.
# skops: MLflow 3 serialises sklearn models with skops, but mlflow-skinny does not install it.
# (Root cause of the first DAG failure - found by the agent; see docs/evidence/failure-01-missing-skops.json.)
RUN pip install --no-cache-dir \
    "scikit-learn==1.9.1" \
    "pandas==3.1.1" \
    "numpy==2.5.4" \
    "joblib==1.6.0" \
    "mlflow-skinny==3.18.0" \
    "skops==0.16.0" \
    "pyyaml>=6.0"
