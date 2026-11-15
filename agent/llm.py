"""LLM providers.

``AnthropicLLM`` calls Claude through the official ``anthropic`` SDK (Messages API + tool use).
``OpenAICompatLLM`` calls any OpenAI-compatible API with tool calling (Gemini, Mistral, Cerebras, Groq, OpenRouter).
The orchestrator owns the agent loop so it can gate actions, enforce coverage and trace every call;
providers only translate messages and tool calls.

Any object with the same ``respond`` signature works (see ``tests/fakes.py: ScriptedLLM``), which is
how the agent is tested deterministically without network access.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import PROVIDER_KEYS, Settings
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


# ----------------------------------------------------------------------------- Gemini (OpenAI-compatible)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_SCHEMA_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "nullable",
    "format",
}


def to_gemini_schema(schema: Any) -> Any:
    """Reduce a Pydantic JSON schema to the OpenAPI subset Gemini function declarations accept.

    Optional fields (``anyOf: [X, null]``) become ``X`` + ``nullable``. Argument validation is not
    weakened: the executor still validates every call against the full Pydantic model.
    """
    if isinstance(schema, list):
        return [to_gemini_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        merged = {**(options[0] if options else {"type": "string"}), "nullable": True}
        if "description" in schema:
            merged["description"] = schema["description"]
        return to_gemini_schema(merged)
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties":
            out[key] = {name: to_gemini_schema(prop) for name, prop in value.items()}
        elif key == "items":
            out[key] = to_gemini_schema(value)
        elif key == "format" and value not in ("enum", "date-time"):
            continue
        else:
            out[key] = value
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}  # free-form dicts (e.g. DAG conf) - validated server-side
    return out


def _openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the orchestrator's (Anthropic-shaped) history into OpenAI chat messages."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            raw = next((b["message"] for b in content if block_type(b) == "provider_raw"), None)
            if raw is not None:  # replay exactly what the model sent (keeps Gemini thought signatures)
                out.append(raw)
                continue
            text = "\n".join(block_attr(b, "text", "") for b in content if block_type(b) == "text")
            calls = [
                {
                    "id": block_attr(b, "id"),
                    "type": "function",
                    "function": {"name": block_attr(b, "name"), "arguments": json.dumps(block_attr(b, "input") or {})},
                }
                for b in content
                if block_type(b) == "tool_use"
            ]
            out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
        else:  # user turn: tool results and/or text
            for b in content:
                if block_type(b) == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])})
                elif block_type(b) == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


_FINISH = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


@dataclass(frozen=True)
class CompatPreset:
    base_url: str
    key_envs: tuple[str, ...]
    default_model: str
    gemini_schema: bool = False  # Gemini accepts only an OpenAPI subset of JSON schema
    extra: dict[str, Any] = field(default_factory=dict)


# OpenAI-compatible providers. All have a free or low-cost tier and support function calling.
PRESETS: dict[str, CompatPreset] = {
    "gemini": CompatPreset(
        GEMINI_BASE_URL,
        PROVIDER_KEYS["gemini"],
        "gemini-flash-latest",
        gemini_schema=True,
        extra={"reasoning_effort": "low"},
    ),
    "mistral": CompatPreset("https://api.mistral.ai/v1", PROVIDER_KEYS["mistral"], "mistral-medium-latest"),
    "cerebras": CompatPreset("https://api.cerebras.ai/v1", PROVIDER_KEYS["cerebras"], "gpt-oss-120b"),
    "groq": CompatPreset("https://api.groq.com/openai/v1", PROVIDER_KEYS["groq"], "openai/gpt-oss-120b"),
    "openrouter": CompatPreset("https://openrouter.ai/api/v1", PROVIDER_KEYS["openrouter"], "openai/gpt-oss-120b:free"),
}


def _plain_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in schema.items() if k != "title"}
    out["properties"] = {
        n: {k: v for k, v in p.items() if k != "title"} for n, p in schema.get("properties", {}).items()
    }
    return out


class OpenAICompatLLM:
    """Any OpenAI-compatible chat-completions API with tool calling (Gemini, Mistral, Cerebras, Groq, ...)."""

    def __init__(self, settings: Settings, provider: str) -> None:
        from openai import OpenAI

        preset = PRESETS[provider]
        self.name = provider
        self.preset = preset
        self.model = settings.llm_model or preset.default_model
        api_key = next((os.environ[e] for e in preset.key_envs if os.environ.get(e)), None)
        self.client = OpenAI(
            api_key=api_key, base_url=settings.llm_base_url or preset.base_url, max_retries=5, timeout=120.0
        )

    def respond(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMTurn:
        convert = to_gemini_schema if self.preset.gemini_schema else _plain_schema
        fn_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": convert(t["input_schema"]),
                },
            }
            for t in tools
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=_openai_messages(system, messages),
            tools=fn_tools,
            tool_choice="auto",
            **self.preset.extra,
        )
        choice = response.choices[0]
        msg = choice.message
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for call in msg.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": call.function.arguments}
            blocks.append({"type": "tool_use", "id": call.id, "name": call.function.name, "input": args})
        raw = msg.model_dump(exclude_none=True)
        for noisy in ("refusal", "annotations", "audio", "function_call", "reasoning", "reasoning_details"):
            raw.pop(noisy, None)
        blocks.append({"type": "provider_raw", "message": raw})
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }
        stop = "tool_use" if msg.tool_calls else _FINISH.get(choice.finish_reason or "stop", "end_turn")
        log_event(logger, "llm_call", provider=self.name, model=self.model, stop_reason=stop, **usage)
        return LLMTurn(content=blocks, stop_reason=stop, usage=usage, model=self.model)


def build_llm(settings: Settings) -> LLM | None:
    provider = settings.resolved_provider
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider in PRESETS:
        return OpenAICompatLLM(settings, provider)
    return None
