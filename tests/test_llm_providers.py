"""LLM provider selection and the Gemini (OpenAI-compatible) translation layer - no network."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from agent.config import PROVIDER_KEYS
from agent.llm import PRESETS, OpenAICompatLLM, _openai_messages, _plain_schema, build_llm, to_gemini_schema
from agent.tools import build_registry


@pytest.mark.parametrize(
    "provider,anthropic,gemini,expected",
    [
        ("auto", "", "", "offline"),
        ("auto", "sk-ant-x", "", "anthropic"),
        ("auto", "", "AIza-x", "gemini"),
        ("auto", "sk-ant-x", "AIza-x", "anthropic"),
        ("gemini", "", "AIza-x", "gemini"),
        ("offline", "sk-ant-x", "AIza-x", "offline"),
    ],
)
def test_provider_resolution(settings, monkeypatch, provider, anthropic, gemini, expected):
    for envs in PROVIDER_KEYS.values():
        for var in envs:
            monkeypatch.delenv(var, raising=False)
    for var, val in (("ANTHROPIC_API_KEY", anthropic), ("GEMINI_API_KEY", gemini)):
        monkeypatch.setenv(var, val) if val else monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert replace(settings, llm_provider=provider).resolved_provider == expected


def test_offline_builds_no_llm(settings):
    assert build_llm(replace(settings, llm_provider="offline")) is None


def test_every_tool_schema_converts_to_gemini_subset():
    allowed = {
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

    def walk(node):
        if isinstance(node, dict):
            assert set(node) <= allowed, set(node) - allowed
            for key, value in node.items():
                if key == "properties":
                    for prop in value.values():
                        walk(prop)
                elif key == "items":
                    walk(value)

    for tool in build_registry().anthropic_tools():
        converted = to_gemini_schema(tool["input_schema"])
        assert converted["type"] == "object"
        walk(converted)


def test_optional_field_becomes_nullable():
    schema = {
        "type": "object",
        "properties": {
            "run_id": {
                "anyOf": [{"type": "string", "pattern": "x"}, {"type": "null"}],
                "default": None,
                "description": "d",
            }
        },
        "additionalProperties": False,
    }
    out = to_gemini_schema(schema)
    assert out["properties"]["run_id"] == {"type": "string", "nullable": True, "description": "d"}
    assert "additionalProperties" not in out


def test_history_translation_round_trip():
    history = [
        {"role": "user", "content": "why is the API down?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "k8s_diagnose_deployment",
                    "input": {"namespace": "mlops-agent"},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "[E1] {...}"}]},
        {"role": "user", "content": "[orchestrator] nudge"},
    ]
    msgs = _openai_messages("SYS", history)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[2]["tool_calls"][0]["function"] == {
        "name": "k8s_diagnose_deployment",
        "arguments": json.dumps({"namespace": "mlops-agent"}),
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "[E1] {...}"}
    assert msgs[4]["role"] == "user"


def test_raw_provider_message_replayed_verbatim():
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c9",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
                "extra_content": {"google": {"thought_signature": "sig"}},
            }
        ],
    }
    msgs = _openai_messages("S", [{"role": "assistant", "content": [{"type": "provider_raw", "message": raw}]}])
    assert msgs[1] == raw  # thought signature preserved for multi-turn function calling


class _FakeCompletions:
    def __init__(self, message, finish="tool_calls"):
        self.message, self.finish, self.kwargs = message, finish, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return NS(
            choices=[NS(message=self.message, finish_reason=self.finish)],
            usage=NS(prompt_tokens=120, completion_tokens=30),
        )


def _gemini_with(message, finish="tool_calls"):
    llm = OpenAICompatLLM.__new__(OpenAICompatLLM)
    llm.name, llm.preset, llm.model = "gemini", PRESETS["gemini"], "gemini-flash-latest"
    completions = _FakeCompletions(message, finish)
    llm.client = NS(chat=NS(completions=completions))
    return llm, completions


def _message(content=None, calls=None):
    dump = {"role": "assistant", "content": content}
    return NS(content=content, tool_calls=calls, model_dump=lambda exclude_none=True: dump)


def test_gemini_tool_calls_parsed_into_blocks():
    call = NS(id="c1", function=NS(name="mlflow_list_runs", arguments='{"max_results": 3}'))
    llm, completions = _gemini_with(_message(calls=[call]))
    turn = llm.respond("SYS", [{"role": "user", "content": "hi"}], build_registry().anthropic_tools())
    assert turn.stop_reason == "tool_use" and turn.usage == {"input_tokens": 120, "output_tokens": 30}
    tool_use = next(b for b in turn.content if b["type"] == "tool_use")
    assert tool_use == {"type": "tool_use", "id": "c1", "name": "mlflow_list_runs", "input": {"max_results": 3}}
    assert completions.kwargs["tools"][0]["type"] == "function"


def test_gemini_final_text_and_bad_json_arguments():
    llm, _ = _gemini_with(_message(content="**Summary** - ok"), finish="stop")
    turn = llm.respond("SYS", [{"role": "user", "content": "hi"}], [])
    assert turn.stop_reason == "end_turn" and turn.content[0] == {"type": "text", "text": "**Summary** - ok"}
    call = NS(id="c2", function=NS(name="x", arguments="{not json"))
    llm, _ = _gemini_with(_message(calls=[call]))
    block = next(b for b in llm.respond("S", [], []).content if b["type"] == "tool_use")
    assert "_unparseable_arguments" in block["input"]  # the executor will reject it as invalid_request


@pytest.mark.parametrize(
    "provider,env",
    [
        ("mistral", "MISTRAL_API_KEY"),
        ("cerebras", "CEREBRAS_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_auto_detects_other_openai_compatible_providers(settings, monkeypatch, provider, env):
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "CEREBRAS_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env, "test-key")
    s = replace(settings, llm_provider="auto")
    assert s.resolved_provider == provider
    llm = build_llm(s)
    assert isinstance(llm, OpenAICompatLLM) and llm.model == PRESETS[provider].default_model
    assert str(llm.client.base_url).rstrip("/") == PRESETS[provider].base_url.rstrip("/")


def test_model_and_endpoint_overrides(settings, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = build_llm(
        replace(
            settings, llm_provider="mistral", llm_model="mistral-large-latest", llm_base_url="http://proxy.local/v1"
        )
    )
    assert llm.model == "mistral-large-latest" and "proxy.local" in str(llm.client.base_url)


def test_non_gemini_providers_keep_full_json_schema():
    schema = build_registry().get("k8s_scale_deployment").json_schema()
    plain = _plain_schema(schema)
    assert plain["properties"]["replicas"]["maximum"] == 50 and "pattern" in plain["properties"]["namespace"]
