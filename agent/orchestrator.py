"""The agent loop.

LLM mode:   Claude plans -> tool_use blocks are executed (in parallel) through the ToolExecutor ->
            results (with evidence ids) go back -> repeat until a final answer. A coverage check
            sends the model back if a system the question requires was never consulted. The final
            answer passes through the grounding verifier.
Offline:    deterministic planner + evidence-only composer (same executor, same safety).
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent import grounding, planner
from agent.config import Settings, get_settings
from agent.llm import LLM, block_attr, block_type, build_llm
from agent.logging_utils import get_logger, log_event, new_request_id, request_id_var
from agent.prompts import COVERAGE_NUDGE, SYSTEM_PROMPT
from agent.safety import PolicyError
from agent.schemas import ChatResponse, ToolCallTrace, ToolResult, ToolStatus
from agent.tools import build_executor
from agent.tools.base import RequestTrace, ToolExecutor

logger = get_logger("orchestrator")

# Which systems a question needs - used by the coverage enforcer.
_SYSTEM_KEYWORDS = {
    "dvc": ("dataset", "data version", "dvc", "provenance"),
    "mlflow": ("best model", "best recent model", "experiment", "metric", "f1", "mlflow", "best run", "runs"),
    "airflow": ("pipeline", "airflow", "dag", "training workflow"),
    "kubernetes": ("deploy", "kubernetes", "k8s", "pod", "running", "prediction api", "inference"),
}


def required_systems(message: str) -> set[str]:
    t = message.lower()
    if "retrain" in t:
        return {"dvc", "mlflow", "airflow", "kubernetes"}
    return {s for s, words in _SYSTEM_KEYWORDS.items() if any(w in t for w in words)}


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


class Agent:
    def __init__(
        self,
        settings: Settings | None = None,
        executor: ToolExecutor | None = None,
        llm: LLM | str | None = "auto",
        max_parallel_tools: int = 6,
    ) -> None:
        self.settings = settings or get_settings()
        self.executor = executor or build_executor(self.settings)
        self.llm: LLM | None = build_llm(self.settings) if llm == "auto" else llm  # type: ignore[assignment]
        self.max_parallel = max_parallel_tools
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "offline"

    # ---------------------------------------------------------------- sessions

    def _session(self, session_id: str | None) -> Session:
        with self._lock:
            sid = session_id or f"s_{uuid.uuid4().hex[:10]}"
            sess = self._sessions.get(sid)
            if sess is None:
                sess = self._sessions[sid] = Session(sid)
            if len(self._sessions) > 200:  # evict the stalest sessions
                for old in sorted(self._sessions.values(), key=lambda s: s.updated)[:50]:
                    self._sessions.pop(old.session_id, None)
            return sess

    # ---------------------------------------------------------------- public API

    def chat(self, message: str, session_id: str | None = None, request_id: str | None = None) -> ChatResponse:
        request_id = request_id or new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        sess = self._session(session_id)
        trace = RequestTrace(request_id=request_id, session_id=sess.session_id)
        log_event(logger, "chat_request", session_id=sess.session_id, mode=self.mode, message=message[:500])
        usage: dict[str, int] = {}
        try:
            if self.llm is None:
                answer = self._run_offline(message, trace)
            else:
                answer, usage = self._run_llm(message, sess, trace)
        except Exception as exc:  # last-resort guard: never crash the API, never invent a result
            log_event(logger, "chat_failed", level=40, error=f"{type(exc).__name__}: {exc}")
            answer = (
                "**Summary** - The agent hit an internal error and could not complete the request. "
                f"No operational results are reported.\n\n**Uncertainties**\n- {type(exc).__name__}: {str(exc)[:300]}"
            )
        finally:
            request_id_var.reset(token)

        report = grounding.verify(answer, trace.results)
        if not report.grounded:
            answer += (
                "\n\n> **Grounding check:** the following values were not found in any tool output and should be "
                f"treated as unverified: {', '.join(report.unsupported_values)}"
            )
        sess.updated = time.time()
        response = ChatResponse(
            request_id=request_id,
            session_id=sess.session_id,
            answer=answer,
            mode=self.mode,
            tool_calls=[_trace(r) for r in trace.results],
            pending_confirmations=[r.confirmation for r in trace.results if r.confirmation is not None],
            systems_consulted=trace.systems,
            grounding=report,
            usage=usage,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        log_event(
            logger,
            "chat_response",
            request_id=request_id,
            session_id=sess.session_id,
            tools=[t.tool for t in response.tool_calls],
            grounded=report.grounded,
            duration_ms=response.duration_ms,
            **usage,
        )
        return response

    def decide(self, confirmation_id: str, approve: bool, session_id: str | None = None) -> dict[str, Any]:
        """Apply a human decision to a pending action. Execution bypasses the LLM entirely."""
        request_id = new_request_id()
        token = request_id_var.set(request_id)
        try:
            trace = RequestTrace(request_id=request_id, session_id=session_id)
            try:
                action, result = self.executor.execute_confirmed(confirmation_id, trace, approve=approve)
            except PolicyError as exc:
                return {"request_id": request_id, "status": "rejected", "error": str(exc)}
            out: dict[str, Any] = {
                "request_id": request_id,
                "confirmation_id": confirmation_id,
                "tool": action.tool,
                "arguments": action.arguments,
                "status": "approved" if approve else "declined",
            }
            if result is not None:
                out["result"] = result.model_dump(mode="json", exclude={"confirmation"})
            # Let the conversation know what the human decided, so follow-up questions have context.
            if action.session_id and action.session_id in self._sessions and self.llm is not None:
                note = (
                    f"[system] Human {'APPROVED' if approve else 'DECLINED'} {confirmation_id} ({action.description})."
                )
                if result is not None:
                    note += f" Execution status: {result.status}. {result.summary or result.error}"
                self._sessions[action.session_id].messages.append({"role": "user", "content": note})
                self._sessions[action.session_id].messages.append(
                    {"role": "assistant", "content": "Noted - I have recorded the outcome of that action."}
                )
            return out
        finally:
            request_id_var.reset(token)

    # ---------------------------------------------------------------- offline

    def _run_offline(self, message: str, trace: RequestTrace) -> str:
        intent, steps = planner.plan(message)
        for step in steps:
            call = step(trace.results) if callable(step) else step
            if call is None:
                continue
            name, args = call
            self.executor.execute(name, args, trace)
        return planner.compose(intent, trace.results)

    # ---------------------------------------------------------------- LLM loop

    def _run_llm(self, message: str, sess: Session, trace: RequestTrace) -> tuple[str, dict[str, int]]:
        assert self.llm is not None
        tools = self.executor.registry.anthropic_tools()
        history = list(sess.messages)
        history.append({"role": "user", "content": message})
        usage: dict[str, int] = {}
        needed = required_systems(message)
        nudges = 0
        final_text = ""

        for _step in range(self.settings.max_agent_steps):
            turn = self.llm.respond(SYSTEM_PROMPT, history, tools)
            for k, v in turn.usage.items():
                usage[k] = usage.get(k, 0) + int(v or 0)
            history.append({"role": "assistant", "content": turn.content})

            if turn.stop_reason == "refusal":
                final_text = "**Summary** - The model declined this request. No actions were taken."
                break

            tool_uses = [b for b in turn.content if block_type(b) == "tool_use"]
            text = "\n".join(block_attr(b, "text", "") for b in turn.content if block_type(b) == "text").strip()

            if not tool_uses:
                missing = needed - set(trace.systems)
                if missing and nudges < 2 and turn.stop_reason != "max_tokens":
                    nudges += 1
                    log_event(logger, "coverage_nudge", missing=sorted(missing))
                    history.append(
                        {"role": "user", "content": COVERAGE_NUDGE.format(missing=", ".join(sorted(missing)))}
                    )
                    continue
                final_text = text
                break

            results = self._execute_parallel(tool_uses, trace)
            history.append({"role": "user", "content": results})
        else:
            final_text = (
                "**Summary** - I reached the maximum number of reasoning steps before finishing. "
                "Partial evidence is listed in the tool trace; no unverified conclusions are drawn."
            )

        if not final_text:
            final_text = "**Summary** - No answer was produced."
        # Persist only the user question + final answer (compact, safe to replay across turns).
        sess.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": final_text}])
        sess.messages[:] = sess.messages[-12:]
        return final_text, usage

    def _execute_parallel(self, tool_uses: list[Any], trace: RequestTrace) -> list[dict[str, Any]]:
        def run(block: Any) -> tuple[Any, ToolResult]:
            return block, self.executor.execute(
                block_attr(block, "name"), dict(block_attr(block, "input") or {}), trace
            )

        if len(tool_uses) == 1:
            pairs = [run(tool_uses[0])]
        else:
            with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                pairs = list(pool.map(run, tool_uses))
        blocks = []
        for block, result in pairs:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block_attr(block, "id"),
                    "content": f"[{result.evidence_id}] " + result.for_llm(),
                    "is_error": result.status not in (ToolStatus.OK, ToolStatus.REQUIRES_CONFIRMATION),
                }
            )
        return blocks


def _trace(r: ToolResult) -> ToolCallTrace:
    return ToolCallTrace(
        evidence_id=r.evidence_id,
        tool=r.tool,
        status=str(r.status),
        duration_ms=r.duration_ms,
        arguments=r.arguments,
        summary=r.summary,
        error=r.error,
    )
