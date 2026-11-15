"""Post-hoc grounding verifier.

Extracts operational values from the final answer (decimal metrics, percentages, md5 hashes, git
commits, MLflow run ids) and checks that each one is supported by the raw tool evidence collected for
this request. Unsupported values are flagged in the response, so a hallucinated number can never be
presented silently as fact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from agent.schemas import GroundingReport, ToolResult

_DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d{2,})(?!\.?\d)(?![A-Za-z_])")  # allows a trailing full stop
_PERCENT = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?%")
_HEX = re.compile(r"\b([0-9a-f]{7,40})\b")
_CITATION = re.compile(r"\[E\d+\]")


def _evidence_text(results: Iterable[ToolResult]) -> str:
    parts = []
    for r in results:
        if r.status == "ok":
            parts.append(r.summary)
            parts.append(json.dumps(r.data, default=str))
    return "\n".join(parts).lower()


def _evidence_numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def _supported_number(value: float, decimals: int, numbers: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(n - value) <= tol for n in numbers)


def verify(answer: str, results: list[ToolResult]) -> GroundingReport:
    text = _CITATION.sub(" ", answer)
    evidence = _evidence_text(results)
    numbers = _evidence_numbers(evidence)
    unsupported: list[str] = []
    checked = 0

    for m in _DECIMAL.finditer(text):
        raw = m.group(1)
        checked += 1
        if not _supported_number(float(raw), len(raw.split(".")[1]), numbers):
            unsupported.append(raw)

    for m in _PERCENT.finditer(text):
        raw = m.group(1)
        value = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        checked += 1
        as_fraction = _supported_number(value / 100, decimals + 2, numbers)
        if not (as_fraction or _supported_number(value, decimals, numbers)):
            unsupported.append(raw + "%")

    for m in _HEX.finditer(text.lower()):
        token = m.group(1)
        if token.isdigit():  # plain integers are not identifiers
            continue
        checked += 1
        if token not in evidence:
            unsupported.append(token)

    unsupported = sorted(set(unsupported))
    note = ""
    if unsupported:
        note = "Values not found in any tool output: " + ", ".join(unsupported)
    elif checked == 0:
        note = "No numeric/identifier claims to verify."
    return GroundingReport(grounded=not unsupported, checked_values=checked, unsupported_values=unsupported, note=note)
