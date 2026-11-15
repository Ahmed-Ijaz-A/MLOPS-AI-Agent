FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install "mlflow==3.18.0" && useradd --uid 10001 --create-home mlflow && mkdir -p /mlflow && chown mlflow /mlflow
USER 10001
WORKDIR /mlflow
EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --artifacts-destination /mlflow/artifacts --serve-artifacts \
  --allowed-hosts \"localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000,host.docker.internal,host.docker.internal:5000\""]
