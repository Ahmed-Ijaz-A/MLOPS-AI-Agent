"""FastAPI interface for the MLOps agent.

    uvicorn api.main:app --port 8000        ->  web console at http://localhost:8000

Endpoints
    POST /api/chat                          natural-language request -> evidence-based answer + trace
    GET  /api/confirmations                 pending human approvals
    POST /api/confirmations/{id}            approve / decline a pending action (executed by code)
    GET  /api/tools                         tool catalog with permission levels and JSON schemas
    GET  /api/integrations                  live reachability of DVC / MLflow / Airflow / Kubernetes
    GET  /api/audit                         recent audit-log events
    GET  /healthz, /readyz                  liveness / readiness

If ``AGENT_API_KEY`` is set, every /api route requires the ``X-API-Key`` header.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import __version__
from agent.config import get_settings
from agent.logging_utils import configure_logging, get_logger, log_event, new_request_id, request_id_var
from agent.orchestrator import Agent
from agent.schemas import ChatRequest, ChatResponse, ConfirmationDecision

STATIC = Path(__file__).parent / "static"
logger = get_logger("api")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("AGENT_API_KEY", "")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def create_app(agent: Agent | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    app = FastAPI(
        title="MLOps AI Agent",
        version=__version__,
        description="Evidence-grounded agent over DVC, MLflow, Airflow 3 and Kubernetes.",
    )
    state: dict[str, Any] = {"agent": agent}

    def get_agent() -> Agent:
        if state["agent"] is None:
            state["agent"] = Agent(settings)
        return state["agent"]

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        if not rid.replace("_", "").replace("-", "").isalnum() or len(rid) > 64:
            rid = new_request_id()
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        if request.url.path.startswith("/api"):
            log_event(
                logger,
                "http_request",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak stack traces / secrets
        log_event(logger, "unhandled_error", level=40, error=type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": request_id_var.get()})

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> dict[str, Any]:
        a = get_agent()
        return {"status": "ready", "mode": a.mode, "tools": len(a.executor.registry.names())}

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        return get_agent().chat(body.message, session_id=body.session_id, request_id=request_id_var.get())

    @app.get("/api/confirmations", dependencies=[Depends(require_api_key)])
    def pending(session_id: str | None = None) -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in get_agent().executor.confirmations.list(session_id)]

    @app.post("/api/confirmations/{confirmation_id}", dependencies=[Depends(require_api_key)])
    def decide(confirmation_id: str, body: ConfirmationDecision) -> dict[str, Any]:
        if not confirmation_id.startswith("cfm_") or len(confirmation_id) > 40:
            raise HTTPException(status_code=400, detail="malformed confirmation id")
        out = get_agent().decide(confirmation_id, body.approve, session_id=body.session_id)
        if out.get("status") == "rejected":
            raise HTTPException(status_code=409, detail=out["error"])
        return out

    @app.get("/api/tools", dependencies=[Depends(require_api_key)])
    def tools() -> dict[str, Any]:
        return {"count": len(get_agent().executor.registry.names()), "tools": get_agent().executor.registry.catalog()}

    @app.get("/api/audit", dependencies=[Depends(require_api_key)])
    def audit(limit: int = 50) -> list[dict[str, Any]]:
        return get_agent().executor.confirmations.audit.events[-max(1, min(limit, 500)) :]

    @app.get("/api/integrations", dependencies=[Depends(require_api_key)])
    def integrations() -> dict[str, Any]:
        a = get_agent()
        probes = {
            "dvc": ("dvc_get_dataset_version", {}),
            "mlflow": ("mlflow_list_experiments", {}),
            "airflow": ("airflow_list_dags", {}),
            "kubernetes": ("k8s_list_workloads", {}),
        }
        out = {}
        for system, (tool, args) in probes.items():
            r = a.executor.execute(tool, args)
            out[system] = {"status": r.status, "detail": (r.summary or r.error or "")[:200], "ms": r.duration_ms}
        return {"mode": a.mode, "model": settings.anthropic_model if a.mode == "llm" else None, "systems": out}

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


app = create_app()
