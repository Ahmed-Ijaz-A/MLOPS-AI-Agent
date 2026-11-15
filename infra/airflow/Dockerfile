# Airflow 3 image with the training dependencies baked in (no slow _PIP_ADDITIONAL_REQUIREMENTS).
FROM apache/airflow:3.1.0-python3.12

USER airflow
# Same scikit-learn / pandas versions as the agent + inference image so models are portable.
RUN pip install --no-cache-dir \
    "scikit-learn==1.9.1" \
    "pandas==3.1.1" \
    "numpy==2.5.4" \
    "joblib==1.6.0" \
    "mlflow-skinny==3.18.0" \
    "pyyaml>=6.0"
