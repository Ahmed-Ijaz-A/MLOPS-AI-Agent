"""LLM providers.

``AnthropicLLM`` calls Claude through the official ``anthropic`` SDK (Messages API + tool use). The
orchestrator owns the agent loop so it can gate actions, enforce coverage and trace every call.

Any object with the same ``respond`` signature works (see ``tests/fakes.py: ScriptedLLM``), which is
how the agent is tested deterministically without network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import Settings
from agent.logging_utils import get_logger, log_event

logger = get_logger("llm")


@dataclass
class LLMTurn:
    content: list[Any]  # SDK content blocks (or dicts in tests) - appended back to history verbatim
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class LLM(Protocol):
    name: str

    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn: ...


def block_type(block: Any) -> str:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", "")


def block_attr(block: Any, name: str, default: Any = None) -> Any:
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3, timeout=180.0)
        self.model = settings.anthropic_model
        self.effort = settings.agent_effort
        self.fallbacks = settings.anthropic_fallbacks

    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            # Server-side fallback: if a safety classifier declines, the API re-runs the request on a
            # suitable fallback model inside the same call.
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        log_event(logger, "llm_call", model=response.model, stop_reason=response.stop_reason, **usage)
        return LLMTurn(
            content=list(response.content), stop_reason=response.stop_reason or "", usage=usage, model=response.model
        )


def build_llm(settings: Settings) -> LLM | None:
    if not settings.use_llm:
        return None
    return AnthropicLLM(settings)
