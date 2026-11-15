"""Command-line interface.

    python -m agent.main "Why is my prediction API unavailable?"     # one-shot
    python -m agent.main                                              # interactive REPL
    python -m agent.main --offline "..."                              # force the deterministic planner
    python -m agent.main --json "..."                                 # machine-readable output

Pending actions (restart / scale / rollback / trigger) are presented for an explicit y/N decision
that is executed by code - the model can never approve its own actions.
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.config import get_settings
from agent.logging_utils import configure_logging

console = Console()

_STATUS_STYLE = {
    "ok": "green",
    "requires_confirmation": "yellow",
    "denied": "red",
    "unavailable": "red",
    "timeout": "red",
    "error": "red",
    "invalid_request": "magenta",
}


def _render(resp, show_trace: bool = True) -> None:
    if show_trace and resp.tool_calls:
        table = Table(title=f"Tool trace - {resp.request_id} ({resp.mode} mode)", show_lines=False)
        for col in ("id", "tool", "status", "ms", "summary / error"):
            table.add_column(col, overflow="fold")
        for t in resp.tool_calls:
            style = _STATUS_STYLE.get(t.status, "white")
            table.add_row(
                t.evidence_id,
                t.tool,
                f"[{style}]{t.status}[/{style}]",
                f"{t.duration_ms:.0f}",
                (t.summary or t.error or "")[:220],
            )
        console.print(table)
    console.print(Panel(Markdown(resp.answer), title="Agent", border_style="cyan"))
    if resp.grounding:
        g = resp.grounding
        colour = "green" if g.grounded else "red"
        console.print(
            f"[{colour}]grounding: {'OK' if g.grounded else 'UNVERIFIED VALUES'}[/{colour}] "
            f"({g.checked_values} values checked) {g.note}"
        )


def _handle_confirmations(agent, resp, auto_decline: bool) -> None:
    for action in resp.pending_confirmations:
        console.print(
            Panel(
                f"[bold]{action.description}[/bold]\nconfirmation id: {action.confirmation_id}\n"
                f"permission: {action.permission}",
                title="Approval required",
                border_style="yellow",
            )
        )
        if auto_decline or not sys.stdin.isatty():
            decision = False
            console.print("[yellow]Non-interactive session: action declined.[/yellow]")
        else:
            decision = console.input("Approve this action? [y/N] ").strip().lower() in {"y", "yes"}
        outcome = agent.decide(action.confirmation_id, decision, session_id=resp.session_id)
        result = outcome.get("result") or {}
        console.print(
            f"-> {outcome['status']}"
            + (f": {result.get('status')} - {result.get('summary') or result.get('error')}" if result else "")
        )


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlops-agent", description="MLOps AI Agent")
    ap.add_argument("message", nargs="*", help="question; omit for interactive mode")
    ap.add_argument("--offline", action="store_true", help="use the deterministic planner (no LLM)")
    ap.add_argument("--json", action="store_true", help="print the raw JSON response")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--decline-all", action="store_true", help="automatically decline pending actions")
    a = ap.parse_args(argv)
    if a.offline:
        os.environ["LLM_PROVIDER"] = "offline"
        get_settings.cache_clear()

    settings = get_settings()
    configure_logging("WARNING" if not a.json else "ERROR", settings.log_dir)
    from agent.orchestrator import Agent

    agent = Agent(settings)

    def ask(text: str, session_id: str | None) -> str:
        with console.status("[cyan]Agent is gathering evidence...[/cyan]"):
            resp = agent.chat(text, session_id=session_id)
        if a.json:
            print(resp.model_dump_json(indent=2))
        else:
            _render(resp, show_trace=not a.no_trace)
        _handle_confirmations(agent, resp, a.decline_all)
        return resp.session_id

    if a.message:
        ask(" ".join(a.message), None)
        return 0

    console.print(Panel(f"MLOps Agent ({agent.mode} mode). Type 'exit' to quit.", border_style="cyan"))
    session_id = None
    while True:
        try:
            text = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            session_id = ask(text, session_id)
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(cli())


if __name__ == "__main__":  # pragma: no cover
    main()
