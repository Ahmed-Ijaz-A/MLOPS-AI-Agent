# Credit-risk inference API image (deployed to Kubernetes).
# Build after exporting a registered model:  python scripts/deploy_model.py --version <N>
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY serving/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY serving/app.py /app/serving/app.py
COPY serving/__init__.py /app/serving/__init__.py
COPY serving/model/ /app/model/

ARG MODEL_VERSION=unknown
LABEL org.opencontainers.image.title="credit-risk-inference" \
      org.opencontainers.image.description="Credit-risk model server (FastAPI)" \
      ai.mlops.model-version="${MODEL_VERSION}"

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app && chown -R app /app
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
